"""
Cross-feature engineering for the LightGBM re-ranking stage.

Builds a 23-dimensional feature vector per (user, item) pair:
  8 user features + 7 item features + 8 cross-interaction features.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Feature name registry ────────────────────────────────────────────────────

USER_FEATURE_NAMES: List[str] = [
    "user_avg_rating",
    "user_review_count",
    "user_avg_price",
    "user_positive_ratio",
    "user_top_category",
    "user_top_brand",
    "user_price_std",
    "user_activity_span",
]

ITEM_FEATURE_NAMES: List[str] = [
    "item_avg_rating",
    "item_review_count",
    "item_price",
    "item_price_tier",
    "item_brand_encoded",
    "item_category_encoded",
    "item_positive_ratio",
]

CROSS_FEATURE_NAMES: List[str] = [
    "price_match",
    "price_tier_match",
    "brand_match",
    "category_match",
    "rating_gap",
    "popularity_score",
    "user_item_price_ratio",
    "retrieval_score",
]


def get_feature_names() -> List[str]:
    """Return the ordered list of all 23 feature names.

    Order: user features → item features → cross features.
    """
    return USER_FEATURE_NAMES + ITEM_FEATURE_NAMES + CROSS_FEATURE_NAMES


# ── Single-pair cross features ───────────────────────────────────────────────


def build_cross_features(
    user_features: pd.Series,
    item_features: pd.Series,
    retrieval_score: float,
) -> Dict[str, float]:
    """Compute 8 interaction features between one user and one item.

    Parameters
    ----------
    user_features : pd.Series
        Series with the 8 user feature values (indexed by feature name).
    item_features : pd.Series
        Series with the 7 item feature values (indexed by feature name).
    retrieval_score : float
        FAISS similarity / retrieval-stage score.

    Returns
    -------
    dict
        Mapping of 8 cross-feature names to their computed values.
    """
    user_avg_price = float(user_features.get("user_avg_price", 0.0))
    item_price = float(item_features.get("item_price", 0.0))

    # price_match: 1 − |user_avg_price − item_price| / max_price, clipped [0,1]
    max_price = max(user_avg_price, item_price)
    price_match = 1.0 - abs(user_avg_price - item_price) / (max_price + 1e-8)
    price_match = float(np.clip(price_match, 0.0, 1.0))

    # price_tier_match: approximate user preferred tier as rounded mean tier
    user_preferred_tier = int(round(float(user_features.get("user_avg_price", 0.0))))
    # For the user, we approximate their preferred price tier.
    # The item_price_tier is already encoded as an integer tier index.
    item_tier = int(item_features.get("item_price_tier", -1))
    # Use a simple heuristic: compare rounded user_avg_price-based tier proxy
    # with the item's tier.  Since actual tier mapping isn't available here,
    # we compare the raw encoded values directly.
    price_tier_match = 1.0 if user_preferred_tier == item_tier else 0.0

    # brand_match
    user_top_brand = user_features.get("user_top_brand", -1)
    item_brand = item_features.get("item_brand_encoded", -2)
    brand_match = 1.0 if user_top_brand == item_brand else 0.0

    # category_match
    user_top_cat = user_features.get("user_top_category", -1)
    item_cat = item_features.get("item_category_encoded", -2)
    category_match = 1.0 if user_top_cat == item_cat else 0.0

    # rating_gap
    item_avg_rating = float(item_features.get("item_avg_rating", 0.0))
    user_avg_rating = float(user_features.get("user_avg_rating", 0.0))
    rating_gap = item_avg_rating - user_avg_rating

    # popularity_score
    item_review_count = float(item_features.get("item_review_count", 0.0))
    popularity_score = float(np.log1p(item_review_count))

    # user_item_price_ratio
    user_item_price_ratio = item_price / (user_avg_price + 1e-8)

    return {
        "price_match": price_match,
        "price_tier_match": price_tier_match,
        "brand_match": brand_match,
        "category_match": category_match,
        "rating_gap": rating_gap,
        "popularity_score": popularity_score,
        "user_item_price_ratio": user_item_price_ratio,
        "retrieval_score": float(retrieval_score),
    }


# ── Batch feature matrix ─────────────────────────────────────────────────────


def build_ranking_feature_matrix(
    user_features_df: pd.DataFrame,
    item_features_df: pd.DataFrame,
    candidates: List[Tuple[int, int, float, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the full feature matrix for a batch of candidates.

    Parameters
    ----------
    user_features_df : pd.DataFrame
        User feature table indexed by ``user_id``.
    item_features_df : pd.DataFrame
        Item feature table indexed by ``item_id``.
    candidates : list of (user_id, item_id, retrieval_score, label)
        Each tuple represents one candidate interaction to featurise.

    Returns
    -------
    X : np.ndarray, shape (n_candidates, 23)
        Feature matrix.
    y : np.ndarray, shape (n_candidates,)
        Labels (binary).
    """
    feature_names = get_feature_names()
    n_features = len(feature_names)  # 23

    X = np.zeros((len(candidates), n_features), dtype=np.float32)
    y = np.zeros(len(candidates), dtype=np.float32)

    # Pre-compute default vectors for missing users/items
    default_user = pd.Series(0.0, index=USER_FEATURE_NAMES)
    default_item = pd.Series(0.0, index=ITEM_FEATURE_NAMES)

    n_missing_users = 0
    n_missing_items = 0

    for idx, (user_id, item_id, retrieval_score, label) in enumerate(candidates):
        # --- User features (8) ---
        if user_id in user_features_df.index:
            user_feats = user_features_df.loc[user_id]
        else:
            user_feats = default_user
            n_missing_users += 1

        # --- Item features (7) ---
        if item_id in item_features_df.index:
            item_feats = item_features_df.loc[item_id]
        else:
            item_feats = default_item
            n_missing_items += 1

        # --- Cross features (8) ---
        cross = build_cross_features(user_feats, item_feats, retrieval_score)

        # Assemble row: user (8) + item (7) + cross (8) = 23
        row = (
            [float(user_feats.get(f, 0.0)) for f in USER_FEATURE_NAMES]
            + [float(item_feats.get(f, 0.0)) for f in ITEM_FEATURE_NAMES]
            + [cross[f] for f in CROSS_FEATURE_NAMES]
        )
        X[idx] = row
        y[idx] = float(label)

    if n_missing_users > 0:
        logger.warning("Missing user features for %d / %d candidates", n_missing_users, len(candidates))
    if n_missing_items > 0:
        logger.warning("Missing item features for %d / %d candidates", n_missing_items, len(candidates))

    return X, y
