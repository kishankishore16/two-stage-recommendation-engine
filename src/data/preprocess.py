"""
Preprocessing pipeline for the Amazon Clothing recommendation engine.

Covers:
  1. Parsing gzipped line-delimited JSON
  2. Loading & cleaning reviews and metadata
  3. Implicit-label creation (rating >= 4 → positive)
  4. Integer ID mapping for users and items
  5. Time-based train/test split
  6. Saving all artefacts as Parquet
"""

import gzip
import json
import re
import ast
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd

from src.config import get_config


# ── Low-level I/O ────────────────────────────────────────────────────────────


def parse_jsonl_gz(path: Path) -> Generator[dict, None, None]:
    """Yield dicts from a gzip-compressed, line-delimited JSON file."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield ast.literal_eval(line)


# ── Loading helpers ──────────────────────────────────────────────────────────


def load_reviews(path: Path) -> pd.DataFrame:
    """Parse review JSONL-gz into a DataFrame.

    Columns returned:
        reviewer_id, asin, rating, timestamp, review_text
    """
    records = []
    for rec in parse_jsonl_gz(path):
        records.append(
            {
                "reviewer_id": rec.get("reviewerID", ""),
                "asin": rec.get("asin", ""),
                "rating": float(rec.get("overall", 0)),
                "timestamp": int(rec.get("unixReviewTime", 0)),
                "review_text": rec.get("reviewText", ""),
            }
        )
    df = pd.DataFrame(records)
    print(f"  Loaded {len(df):,} reviews from {path.name}")
    return df


def load_metadata(path: Path) -> pd.DataFrame:
    """Parse metadata JSONL-gz into a DataFrame.

    Columns returned:
        asin, title, price, brand, category

    Price cleaning:
        - Strips leading '$' and commas.
        - Handles ranges like "$10.00 - $20.00" by taking the midpoint.
        - Non-parseable values become NaN.

    Brand: missing values filled with 'Unknown'.
    Category: the last (leaf) element of the category list, or 'Unknown'.
    """
    records = []
    for rec in parse_jsonl_gz(path):
        raw_price = rec.get("price", "")
        price = _clean_price(raw_price)

        # Extract the leaf category
        cats = rec.get("category", [])
        if isinstance(cats, list) and len(cats) > 0:
            # Flatten nested lists (metadata can be list-of-lists)
            flat = []
            for c in cats:
                if isinstance(c, list):
                    flat.extend(c)
                else:
                    flat.append(c)
            category = flat[-1] if flat else "Unknown"
        else:
            category = "Unknown"

        records.append(
            {
                "asin": rec.get("asin", ""),
                "title": rec.get("title", ""),
                "price": price,
                "brand": rec.get("brand", "Unknown") or "Unknown",
                "category": category,
            }
        )
    df = pd.DataFrame(records)
    df["brand"] = df["brand"].fillna("Unknown")
    print(f"  Loaded {len(df):,} metadata rows from {path.name}")
    return df


# ── Price cleaning ───────────────────────────────────────────────────────────

_PRICE_RANGE_RE = re.compile(
    r"\$?\s*([\d,]+(?:\.\d+)?)\s*[-–]\s*\$?\s*([\d,]+(?:\.\d+)?)"
)
_PRICE_SINGLE_RE = re.compile(r"\$?\s*([\d,]+(?:\.\d+)?)")


def _clean_price(raw: str | float | int | None) -> float | None:
    """Parse a price string into a float, handling ranges and symbols."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if raw > 0 else None

    raw = str(raw).strip()
    if not raw:
        return None

    # Range: take midpoint
    m = _PRICE_RANGE_RE.search(raw)
    if m:
        lo = float(m.group(1).replace(",", ""))
        hi = float(m.group(2).replace(",", ""))
        return (lo + hi) / 2.0

    # Single value
    m = _PRICE_SINGLE_RE.search(raw)
    if m:
        return float(m.group(1).replace(",", ""))

    return None


# ── Label & ID helpers ───────────────────────────────────────────────────────


def create_implicit_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary 'label' column: 1 if rating >= 4.0, else 0."""
    cfg = get_config()
    threshold = cfg.data.positive_rating_threshold
    df = df.copy()
    df["label"] = (df["rating"] >= threshold).astype(int)
    pos = df["label"].sum()
    print(f"  Implicit labels: {pos:,} positive / {len(df) - pos:,} negative")
    return df


def create_id_mappings(
    df: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int]]:
    """Create contiguous integer ID mappings for users and items.

    Returns
    -------
    user_map : dict  {reviewer_id → int user_id}
    item_map : dict  {asin → int item_id}
    """
    unique_users = sorted(df["reviewer_id"].unique())
    unique_items = sorted(df["asin"].unique())

    user_map = {uid: idx for idx, uid in enumerate(unique_users)}
    item_map = {iid: idx for idx, iid in enumerate(unique_items)}

    print(f"  ID mappings: {len(user_map):,} users, {len(item_map):,} items")
    return user_map, item_map


# ── Train/test split ────────────────────────────────────────────────────────


def time_based_split(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* by timestamp at the *train_ratio* percentile.

    No random shuffling — preserves temporal ordering.
    """
    cutoff = np.percentile(df["timestamp"].values, train_ratio * 100)
    train = df[df["timestamp"] <= cutoff].copy()
    test = df[df["timestamp"] > cutoff].copy()
    print(
        f"  Time split @ {train_ratio:.0%}: "
        f"{len(train):,} train / {len(test):,} test"
    )
    return train, test


# ── Full pipeline ────────────────────────────────────────────────────────────


def run_preprocessing() -> None:
    """Execute the complete preprocessing pipeline and persist results."""
    cfg = get_config()
    paths = cfg.paths

    print("=" * 60)
    print("Step 1 — Loading reviews")
    print("=" * 60)
    reviews = load_reviews(paths.reviews_file)

    print("\n" + "=" * 60)
    print("Step 2 — Loading metadata")
    print("=" * 60)
    metadata = load_metadata(paths.metadata_file)

    print("\n" + "=" * 60)
    print("Step 3 — Creating implicit labels")
    print("=" * 60)
    reviews = create_implicit_labels(reviews)

    print("\n" + "=" * 60)
    print("Step 4 — Creating ID mappings")
    print("=" * 60)
    user_map, item_map = create_id_mappings(reviews)
    reviews["user_id"] = reviews["reviewer_id"].map(user_map)
    reviews["item_id"] = reviews["asin"].map(item_map)

    print("\n" + "=" * 60)
    print("Step 5 — Merging reviews with metadata")
    print("=" * 60)
    merged = reviews.merge(metadata, on="asin", how="left")
    print(f"  Merged shape: {merged.shape}")

    print("\n" + "=" * 60)
    print("Step 6 — Time-based train/test split")
    print("=" * 60)
    train_df, test_df = time_based_split(merged, cfg.data.train_ratio)

    # ── Persist ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 7 — Saving artefacts")
    print("=" * 60)

    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    merged.to_parquet(paths.interactions_file, index=False)
    print(f"  ✓ {paths.interactions_file.name}")

    # Save lightweight item metadata
    item_metadata = merged[["item_id", "asin", "title", "price", "category"]].drop_duplicates(subset=["item_id"])
    item_metadata.to_parquet(paths.item_metadata_file, index=False)
    print(f"  ✓ {paths.item_metadata_file.name}")

    train_df.to_parquet(paths.train_file, index=False)
    print(f"  ✓ {paths.train_file.name}")

    test_df.to_parquet(paths.test_file, index=False)
    print(f"  ✓ {paths.test_file.name}")

    # Save ID mappings as parquet (two-column tables)
    user_map_df = pd.DataFrame(
        list(user_map.items()), columns=["reviewer_id", "user_id"]
    )
    user_map_df.to_parquet(paths.user_id_map_file, index=False)
    print(f"  ✓ {paths.user_id_map_file.name}")

    item_map_df = pd.DataFrame(
        list(item_map.items()), columns=["asin", "item_id"]
    )
    item_map_df.to_parquet(paths.item_id_map_file, index=False)
    print(f"  ✓ {paths.item_id_map_file.name}")

    print("\nPreprocessing complete ✔")


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_preprocessing()
