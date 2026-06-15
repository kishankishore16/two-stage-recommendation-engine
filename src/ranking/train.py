"""
LightGBM re-ranker training pipeline.

Generates training data via negative sampling, trains a binary
classifier, evaluates it, and serialises the model to disk.
"""

import logging
import time
from typing import Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from src.config import get_config
from src.ranking.features import (
    build_ranking_feature_matrix,
    get_feature_names,
)

logger = logging.getLogger(__name__)


# ── Negative sampling ─────────────────────────────────────────────────────────


def generate_ranking_training_data(
    train_df: pd.DataFrame,
    user_features_df: pd.DataFrame,
    item_features_df: pd.DataFrame,
    num_negatives: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create training instances with positive + negative sampling.

    For every positive interaction in *train_df* we:
      1. Keep the positive  (retrieval_score=1.0, label=1).
      2. Sample ``num_negatives`` random items the user has **not**
         interacted with (retrieval_score=0.0, label=0).

    Parameters
    ----------
    train_df : pd.DataFrame
        Interaction data with columns ``user_id``, ``item_id``, ``label``.
    user_features_df : pd.DataFrame
        User feature table indexed by ``user_id``.
    item_features_df : pd.DataFrame
        Item feature table indexed by ``item_id``.
    num_negatives : int, default 10
        Number of negative samples per positive interaction.

    Returns
    -------
    X : np.ndarray  (n_samples, 23)
    y : np.ndarray  (n_samples,)
    """
    rng = np.random.default_rng(seed=42)
    all_item_ids = item_features_df.index.values

    # Build a set of items per user for fast lookup
    user_pos_items: dict[int, set] = (
        train_df.groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    candidates: list[tuple[int, int, float, int]] = []

    for _, row in train_df.iterrows():
        user_id = int(row["user_id"])
        item_id = int(row["item_id"])

        # Positive sample
        candidates.append((user_id, item_id, 1.0, 1))

        # Negative samples
        pos_set = user_pos_items.get(user_id, set())
        neg_count = 0
        max_attempts = num_negatives * 10  # safety guard
        attempts = 0
        while neg_count < num_negatives and attempts < max_attempts:
            neg_item = int(rng.choice(all_item_ids))
            if neg_item not in pos_set:
                candidates.append((user_id, neg_item, 0.0, 0))
                neg_count += 1
            attempts += 1

    logger.info(
        "Generated %d training candidates (%d pos, %d neg)",
        len(candidates),
        len(train_df),
        len(candidates) - len(train_df),
    )

    return build_ranking_feature_matrix(user_features_df, item_features_df, candidates)


# ── Training entry point ──────────────────────────────────────────────────────


def train_ranker() -> lgb.LGBMClassifier:
    """Train a LightGBM binary classifier for re-ranking.

    Workflow
    --------
    1. Load user/item features and train/test parquet files.
    2. Generate training data with negative sampling.
    3. Fit a ``LGBMClassifier`` using ``RankingConfig`` hyper-parameters.
    4. Evaluate on the test set (AUC-ROC, log-loss).
    5. Print the top-10 feature importances.
    6. Save the trained model to ``ranker_model_file`` via joblib.

    Returns
    -------
    lgb.LGBMClassifier
        The trained model.
    """
    cfg = get_config()
    rc = cfg.ranking
    paths = cfg.paths

    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info("Loading feature tables and interaction data …")
    user_features_df = pd.read_parquet(paths.user_features_file)
    item_features_df = pd.read_parquet(paths.item_features_file)
    train_df = pd.read_parquet(paths.train_file)
    test_df = pd.read_parquet(paths.test_file)

    # Ensure the feature frames are indexed by id for O(1) lookup
    if user_features_df.index.name != "user_id":
        if "user_id" in user_features_df.columns:
            user_features_df = user_features_df.set_index("user_id")
    if item_features_df.index.name != "item_id":
        if "item_id" in item_features_df.columns:
            item_features_df = item_features_df.set_index("item_id")

    # ── 2. Generate training data ─────────────────────────────────────────
    logger.info("Building training feature matrix (negative sampling) …")
    t0 = time.perf_counter()
    X_train, y_train = generate_ranking_training_data(
        train_df, user_features_df, item_features_df, num_negatives=10,
    )
    logger.info(
        "Training matrix: %s  (%.1f s)",
        X_train.shape, time.perf_counter() - t0,
    )

    # ── 3. Generate test data ─────────────────────────────────────────────
    logger.info("Building test feature matrix …")
    test_candidates = [
        (int(r["user_id"]), int(r["item_id"]), 1.0, int(r["label"]))
        for _, r in test_df.iterrows()
    ]
    X_test, y_test = build_ranking_feature_matrix(
        user_features_df, item_features_df, test_candidates,
    )
    logger.info("Test matrix: %s", X_test.shape)

    # ── 4. Train LightGBM ────────────────────────────────────────────────
    logger.info("Training LightGBM classifier …")
    model = lgb.LGBMClassifier(
        n_estimators=rc.n_estimators,
        learning_rate=rc.learning_rate,
        num_leaves=rc.num_leaves,
        max_depth=rc.max_depth,
        reg_alpha=rc.reg_alpha,
        reg_lambda=rc.reg_lambda,
        subsample=rc.subsample,
        colsample_bytree=rc.colsample_bytree,
        min_child_samples=rc.min_child_samples,
        verbose=rc.verbose,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=["auc", "logloss"],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=30, verbose=True),
        ],
    )

    # ── 5. Evaluate ──────────────────────────────────────────────────────
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    ll = log_loss(y_test, y_proba)
    logger.info("Test AUC-ROC: %.4f  |  Log-loss: %.4f", auc, ll)
    print(f"\n{'='*50}")
    print(f"  Test AUC-ROC : {auc:.4f}")
    print(f"  Test Log-loss: {ll:.4f}")
    print(f"{'='*50}\n")

    # ── 6. Feature importance ─────────────────────────────────────────────
    feature_names = get_feature_names()
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print("Top-10 Feature Importances:")
    print("-" * 40)
    for rank, idx in enumerate(sorted_idx[:10], 1):
        print(f"  {rank:>2}. {feature_names[idx]:<25s} {importances[idx]:>6d}")
    print()

    # ── 7. Save model ────────────────────────────────────────────────────
    model_path = paths.ranker_model_file
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Model saved → %s", model_path)

    return model


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    train_ranker()
