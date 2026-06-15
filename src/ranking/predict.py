"""
Inference-time re-ranking with the trained LightGBM model.

Loads the serialised ranker, scores candidate items for a given
user, and returns the top-K recommendations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import get_config
from src.ranking.features import (
    build_cross_features,
    get_feature_names,
    ITEM_FEATURE_NAMES,
    USER_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)


# ── Model loading ─────────────────────────────────────────────────────────────


def load_ranker(model_path: Optional[Path] = None) -> lgb.LGBMClassifier:
    """Load a trained LightGBM ranker from disk.

    Parameters
    ----------
    model_path : Path, optional
        Explicit path to the model file. Falls back to
        ``config.paths.ranker_model_file`` if not provided.

    Returns
    -------
    lgb.LGBMClassifier
        The de-serialised model ready for inference.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    if model_path is None:
        model_path = get_config().paths.ranker_model_file

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Ranker model not found at {model_path}. "
            "Run `python -m src.ranking.train` first."
        )

    model = joblib.load(model_path)
    logger.info("Loaded ranker from %s", model_path)
    return model


# ── Re-ranking ────────────────────────────────────────────────────────────────


def rerank_candidates(
    ranker: lgb.LGBMClassifier,
    user_features: np.ndarray,  # kept for API compat; we look up from df
    candidate_items: List[Dict],
    retrieval_scores: List[float],
    user_features_df: pd.DataFrame,
    item_features_df: pd.DataFrame,
    user_id: int,
) -> List[Dict]:
    """Re-rank retrieval candidates for a single user.

    Parameters
    ----------
    ranker : lgb.LGBMClassifier
        Trained re-ranking model.
    user_features : np.ndarray
        Raw user embedding (unused here but kept for interface symmetry
        with the retrieval stage).
    candidate_items : list of dict
        Each dict must contain at least ``{"item_id": int, ...}``.
        Additional metadata (title, image, …) is passed through.
    retrieval_scores : list of float
        FAISS similarity scores aligned with *candidate_items*.
    user_features_df : pd.DataFrame
        Full user feature table (indexed by ``user_id``).
    item_features_df : pd.DataFrame
        Full item feature table (indexed by ``item_id``).
    user_id : int
        The user to generate recommendations for.

    Returns
    -------
    list of dict
        Top ``num_results`` items sorted by descending ranker score.
        Each dict is the original candidate dict augmented with a
        ``"ranking_score"`` key.
    """
    cfg = get_config()
    num_results = cfg.ranking.num_results

    if not candidate_items:
        logger.warning("No candidate items supplied for user %d", user_id)
        return []

    # Ensure indices
    if user_features_df.index.name != "user_id" and "user_id" in user_features_df.columns:
        user_features_df = user_features_df.set_index("user_id")
    if item_features_df.index.name != "item_id" and "item_id" in item_features_df.columns:
        item_features_df = item_features_df.set_index("item_id")

    # Look up user features (graceful fallback to zeros)
    if user_id in user_features_df.index:
        user_feats = user_features_df.loc[user_id]
    else:
        logger.warning("User %d not found in feature table; using zeros", user_id)
        user_feats = pd.Series(0.0, index=USER_FEATURE_NAMES)

    # Build feature matrix row-by-row
    feature_names = get_feature_names()
    n_candidates = len(candidate_items)
    X = np.zeros((n_candidates, len(feature_names)), dtype=np.float32)

    for i, (cand, ret_score) in enumerate(zip(candidate_items, retrieval_scores)):
        item_id = int(cand["item_id"])

        # Item features
        if item_id in item_features_df.index:
            item_feats = item_features_df.loc[item_id]
        else:
            item_feats = pd.Series(0.0, index=ITEM_FEATURE_NAMES)

        # Cross features
        cross = build_cross_features(user_feats, item_feats, ret_score)

        row = (
            [float(user_feats.get(f, 0.0)) for f in USER_FEATURE_NAMES]
            + [float(item_feats.get(f, 0.0)) for f in ITEM_FEATURE_NAMES]
            + [cross[f] for f in [
                "price_match", "price_tier_match", "brand_match",
                "category_match", "rating_gap", "popularity_score",
                "user_item_price_ratio", "retrieval_score",
            ]]
        )
        X[i] = row

    # Score
    probas = ranker.predict_proba(X)[:, 1]

    # Attach scores and sort
    scored: List[Dict] = []
    for cand, prob in zip(candidate_items, probas):
        entry = {**cand, "ranking_score": float(prob)}
        scored.append(entry)

    scored.sort(key=lambda d: d["ranking_score"], reverse=True)

    return scored[:num_results]
