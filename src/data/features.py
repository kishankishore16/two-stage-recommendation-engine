"""
Feature engineering for the Amazon Clothing recommendation engine.

Computes per-user and per-item features strictly from the **training** split
so that no future information leaks into the feature vectors.
"""

import numpy as np
import pandas as pd

from src.config import get_config


# ── User features ────────────────────────────────────────────────────────────


def compute_user_features(
    train_df: pd.DataFrame, item_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute 8 aggregated features per user from training interactions.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training interactions with at least: user_id, rating, timestamp, asin.
    item_df : pd.DataFrame
        Item metadata with at least: asin, price, category, brand.

    Returns
    -------
    pd.DataFrame
        Indexed by ``user_id`` with columns:
        user_avg_rating, user_review_count, user_avg_price,
        user_positive_ratio, user_top_category, user_top_brand,
        user_price_std, user_activity_span
    """
    cfg = get_config()
    threshold = cfg.data.positive_rating_threshold

    # Join price / category / brand from metadata
    merged = train_df.merge(
        item_df[["asin", "price", "category", "brand"]],
        on="asin",
        how="left",
        suffixes=("", "_meta"),
    )
    # If the train_df already had these columns from preprocessing, prefer the
    # metadata copy for canonical values.
    for col in ("price", "category", "brand"):
        meta_col = f"{col}_meta"
        if meta_col in merged.columns:
            merged[col] = merged[meta_col].fillna(merged[col])
            merged.drop(columns=[meta_col], inplace=True)

    grp = merged.groupby("user_id")

    user_avg_rating = grp["rating"].mean().rename("user_avg_rating")
    user_review_count = grp["rating"].count().rename("user_review_count")
    user_avg_price = grp["price"].mean().rename("user_avg_price")
    user_positive_ratio = (
        grp["rating"]
        .apply(lambda s: (s >= threshold).mean())
        .rename("user_positive_ratio")
    )
    user_price_std = grp["price"].std().rename("user_price_std")
    user_activity_span = (
        grp["timestamp"]
        .apply(lambda s: (s.max() - s.min()) / 86400.0)
        .rename("user_activity_span")
    )

    # Mode helpers (label-encoded) ───────────────────────────────────────
    cat_labels, cat_uniques = pd.factorize(merged["category"])
    merged["_cat_enc"] = cat_labels
    user_top_category = (
        merged.groupby("user_id")["_cat_enc"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else 0)
        .rename("user_top_category")
    )

    brand_labels, brand_uniques = pd.factorize(merged["brand"])
    merged["_brand_enc"] = brand_labels
    user_top_brand = (
        merged.groupby("user_id")["_brand_enc"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else 0)
        .rename("user_top_brand")
    )

    features = pd.concat(
        [
            user_avg_rating,
            user_review_count,
            user_avg_price,
            user_positive_ratio,
            user_top_category,
            user_top_brand,
            user_price_std,
            user_activity_span,
        ],
        axis=1,
    )

    # Graceful NaN handling
    features["user_avg_price"] = features["user_avg_price"].fillna(
        features["user_avg_price"].median()
    )
    features["user_price_std"] = features["user_price_std"].fillna(0)
    features["user_activity_span"] = features["user_activity_span"].fillna(0)
    features = features.fillna(0)

    features.index.name = "user_id"
    print(f"  User features: {features.shape}")
    return features


# ── Item features ────────────────────────────────────────────────────────────


def compute_item_features(
    train_df: pd.DataFrame, metadata_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute 7 aggregated features per item from training interactions.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training interactions with at least: item_id, asin, rating.
    metadata_df : pd.DataFrame
        Product metadata with at least: asin, price, brand, category.

    Returns
    -------
    pd.DataFrame
        Indexed by ``item_id`` with columns:
        item_avg_rating, item_review_count, item_price,
        item_price_tier, item_brand_encoded, item_category_encoded,
        item_positive_ratio
    """
    cfg = get_config()
    threshold = cfg.data.positive_rating_threshold
    max_brands = cfg.data.max_brands
    max_categories = cfg.data.max_categories

    # Build item-level table from training interactions
    merged = train_df.merge(
        metadata_df[["asin", "price", "brand", "category"]],
        on="asin",
        how="left",
        suffixes=("", "_meta"),
    )
    for col in ("price", "brand", "category"):
        meta_col = f"{col}_meta"
        if meta_col in merged.columns:
            merged[col] = merged[meta_col].fillna(merged[col])
            merged.drop(columns=[meta_col], inplace=True)

    grp = merged.groupby("item_id")

    item_avg_rating = grp["rating"].mean().rename("item_avg_rating")
    item_review_count = grp["rating"].count().rename("item_review_count")
    item_positive_ratio = (
        grp["rating"]
        .apply(lambda s: (s >= threshold).mean())
        .rename("item_positive_ratio")
    )

    # Aggregate price per item (take median of all review rows in case of
    # slight metadata inconsistencies).
    item_price_raw = grp["price"].median().rename("item_price_raw")

    features = pd.concat(
        [item_avg_rating, item_review_count, item_positive_ratio, item_price_raw],
        axis=1,
    )

    # Min-max normalised price ──────────────────────────────────────────
    p_min = features["item_price_raw"].min()
    p_max = features["item_price_raw"].max()
    if p_max > p_min:
        features["item_price"] = (
            (features["item_price_raw"] - p_min) / (p_max - p_min)
        )
    else:
        features["item_price"] = 0.0

    # Price tiers (quantile-based) ─────────────────────────────────────
    features["item_price_tier"] = pd.qcut(
        features["item_price_raw"],
        q=4,
        labels=[0, 1, 2, 3],
        duplicates="drop",
    ).astype(float)
    features["item_price_tier"] = features["item_price_tier"].fillna(0).astype(int)

    features.drop(columns=["item_price_raw"], inplace=True)

    # Brand encoding (top N, rest → Other) ─────────────────────────────
    # Get the first brand per item from the merged frame
    item_brand = merged.groupby("item_id")["brand"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else "Unknown"
    )
    top_brands = item_brand.value_counts().head(max_brands).index
    item_brand_capped = item_brand.where(item_brand.isin(top_brands), other="Other")
    brand_enc, _ = pd.factorize(item_brand_capped)
    features["item_brand_encoded"] = brand_enc

    # Category encoding (top N, rest → Other) ──────────────────────────
    item_cat = merged.groupby("item_id")["category"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else "Unknown"
    )
    top_cats = item_cat.value_counts().head(max_categories).index
    item_cat_capped = item_cat.where(item_cat.isin(top_cats), other="Other")
    cat_enc, _ = pd.factorize(item_cat_capped)
    features["item_category_encoded"] = cat_enc

    # Final NaN cleanup
    features = features.fillna(0)

    features.index.name = "item_id"
    print(f"  Item features: {features.shape}")
    return features


# ── Full pipeline ────────────────────────────────────────────────────────────


def run_feature_engineering() -> None:
    """Load training data & metadata, compute features, save to parquet."""
    cfg = get_config()
    paths = cfg.paths

    print("=" * 60)
    print("Feature Engineering")
    print("=" * 60)

    print("\n→ Loading training data …")
    train_df = pd.read_parquet(paths.train_file)
    print(f"  Train shape: {train_df.shape}")

    print("\n→ Loading metadata …")
    metadata_df = pd.read_parquet(paths.interactions_file)
    # Deduplicate to one row per asin for metadata lookup
    meta_cols = ["asin", "price", "brand", "category"]
    available_cols = [c for c in meta_cols if c in metadata_df.columns]
    metadata_dedup = metadata_df[available_cols].drop_duplicates(subset=["asin"])
    print(f"  Unique items in metadata: {len(metadata_dedup):,}")

    print("\n→ Computing user features …")
    user_feats = compute_user_features(train_df, metadata_dedup)

    print("\n→ Computing item features …")
    item_feats = compute_item_features(train_df, metadata_dedup)

    # Persist
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    user_feats.to_parquet(paths.user_features_file)
    print(f"\n  ✓ Saved {paths.user_features_file.name}")

    item_feats.to_parquet(paths.item_features_file)
    print(f"  ✓ Saved {paths.item_features_file.name}")

    print("\nFeature engineering complete ✔")


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_feature_engineering()
