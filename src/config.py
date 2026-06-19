"""
Central configuration for the recommendation engine.

All paths, hyperparameters, and settings in one place.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathConfig:
    """File system paths."""
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "data" / "models"
    faiss_dir: Path = PROJECT_ROOT / "data" / "faiss"

    # Raw data files
    reviews_file: Path = PROJECT_ROOT / "data" / "raw" / "reviews.json.gz"
    metadata_file: Path = PROJECT_ROOT / "data" / "raw" / "metadata.json.gz"

    # Processed data files
    interactions_file: Path = PROJECT_ROOT / "data" / "processed" / "interactions.parquet"
    item_metadata_file: Path = PROJECT_ROOT / "data" / "processed" / "item_metadata.parquet"
    user_features_file: Path = PROJECT_ROOT / "data" / "processed" / "user_features.parquet"
    item_features_file: Path = PROJECT_ROOT / "data" / "processed" / "item_features.parquet"
    train_file: Path = PROJECT_ROOT / "data" / "processed" / "train.parquet"
    test_file: Path = PROJECT_ROOT / "data" / "processed" / "test.parquet"
    user_id_map_file: Path = PROJECT_ROOT / "data" / "processed" / "user_id_map.parquet"
    item_id_map_file: Path = PROJECT_ROOT / "data" / "processed" / "item_id_map.parquet"

    # Model files
    two_tower_model_file: Path = PROJECT_ROOT / "data" / "models" / "two_tower_best.pt"
    ranker_model_file: Path = PROJECT_ROOT / "data" / "models" / "ranker.lgb"
    faiss_index_file: Path = PROJECT_ROOT / "data" / "faiss" / "items.index"
    faiss_id_map_file: Path = PROJECT_ROOT / "data" / "faiss" / "id_map.npy"

    def ensure_dirs(self):
        """Create all required directories."""
        for d in [self.raw_dir, self.processed_dir, self.models_dir, self.faiss_dir]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data preprocessing settings."""
    # Dataset URLs (SNAP Stanford, 2014 5-core subset)
    reviews_url: str = (
        "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
        "reviews_Clothing_Shoes_and_Jewelry_5.json.gz"
    )
    metadata_url: str = (
        "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
        "meta_Clothing_Shoes_and_Jewelry.json.gz"
    )

    # Implicit feedback threshold
    positive_rating_threshold: float = 4.0  # >= 4 stars is positive

    # Train/test split
    train_ratio: float = 0.8  # 80% by time for training

    # Feature engineering
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    price_tiers: List[str] = field(
        default_factory=lambda: ["budget", "mid", "premium", "luxury"]
    )
    price_tier_quantiles: List[float] = field(
        default_factory=lambda: [0.25, 0.50, 0.75]
    )
    max_brands: int = 500  # Top N brands to keep, rest → "Other"
    max_categories: int = 200  # Top N categories to keep


@dataclass
class RetrievalConfig:
    """Two-Tower model hyperparameters."""
    embedding_dim: int = 64
    id_embedding_dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.2
    temperature: float = 0.07

    # Training
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    patience: int = 5  # Early stopping patience
    num_workers: int = 0  # DataLoader workers (0 for Windows compatibility)

    # FAISS
    faiss_nlist: int = 256  # IVF clusters (if > 100K items)
    faiss_nprobe: int = 32  # Search probes
    retrieval_k: int = 200  # Number of candidates to retrieve


@dataclass
class RankingConfig:
    """LightGBM ranker hyperparameters."""
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 20
    verbose: int = -1

    # Re-ranking
    num_candidates: int = 200  # From FAISS
    num_results: int = 10  # Final top-K to return


@dataclass
class ServingConfig:
    """API and infrastructure settings."""
    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Redis
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = 0
    redis_max_connections: int = 50
    redis_user_ttl: int = 3600  # 1 hour
    redis_item_ttl: int = 86400  # 24 hours

    # FAISS
    faiss_index_path: str = os.getenv(
        "FAISS_INDEX_PATH",
        str(PROJECT_ROOT / "data" / "faiss" / "items.index"),
    )


@dataclass
class Config:
    """Master configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    def __post_init__(self):
        self.paths.ensure_dirs()


# ── Singleton ─────────────────────────────────────────────────────────────────
_config: Config | None = None


def get_config() -> Config:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
