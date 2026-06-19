"""
FastAPI application entrypoint for the recommendation service.

Startup lifespan loads all models, indices, and data into ``app.state``
so they are shared across requests without repeated I/O.

Run with::

    uvicorn src.serving.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import get_config
from src.serving.inference import RecommendationPipeline
from src.serving.redis_client import RedisClient
from src.serving.routes import router

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources on startup; tear down on shutdown."""
    cfg = get_config()
    paths = cfg.paths
    serving = cfg.serving

    logger.info("🚀  Starting recommendation service …")

    # ── 1. PyTorch Two-Tower model ────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading Two-Tower model from %s", paths.two_tower_model_file)
    checkpoint = torch.load(
        paths.two_tower_model_file, map_location=device, weights_only=False
    )
    # The checkpoint stores the full model under key "model" or is the model itself.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Delayed import so the module doesn't need to exist at import time.
        from src.retrieval.model import TwoTowerModel  # type: ignore[import-untyped]

        num_users = checkpoint["num_users"]
        num_items = checkpoint["num_items"]
        user_tower_model = TwoTowerModel(num_users=num_users, num_items=num_items).to(device)
        user_tower_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume entire model was saved via torch.save(model, path).
        user_tower_model = checkpoint
    user_tower_model.eval()
    logger.info("Two-Tower model loaded on %s.", device)

    # ── 2. FAISS index + id map ───────────────────────────────────────────
    faiss_index_path = Path(serving.faiss_index_path)
    faiss_id_map_path = paths.faiss_id_map_file
    logger.info("Loading FAISS index from %s", faiss_index_path)
    faiss_index = faiss.read_index(str(faiss_index_path))
    faiss_id_map = np.load(str(faiss_id_map_path))
    logger.info("FAISS index loaded — %d vectors.", faiss_index.ntotal)

    # ── 3. LightGBM ranker ────────────────────────────────────────────────
    logger.info("Loading LightGBM ranker from %s", paths.ranker_model_file)
    ranker_model = joblib.load(paths.ranker_model_file)
    logger.info("Ranker loaded.")

    # ── 4. Redis client ───────────────────────────────────────────────────
    redis_client = RedisClient(
        host=serving.redis_host,
        port=serving.redis_port,
        db=serving.redis_db,
        max_connections=serving.redis_max_connections,
    )
    try:
        await redis_client.connect()
        logger.info("Redis connected.")
    except Exception:
        logger.warning("Redis unavailable — running without cache.", exc_info=True)

    # ── 5. DataFrames ─────────────────────────────────────────────────────
    logger.info("Loading feature DataFrames …")
    user_features_df = pd.read_parquet(paths.user_features_file)
    item_features_df = pd.read_parquet(paths.item_features_file)

    # Ensure integer index named 'user_id' / 'item_id'.
    if "user_id" in user_features_df.columns:
        user_features_df = user_features_df.set_index("user_id")
    if "item_id" in item_features_df.columns:
        item_features_df = item_features_df.set_index("item_id")

    # Load lightweight item metadata
    logger.info("Loading item metadata from %s", paths.item_metadata_file)
    item_metadata_df = pd.read_parquet(paths.item_metadata_file).set_index("item_id")

    logger.info(
        "DataFrames loaded — %d users, %d items.",
        len(user_features_df),
        len(item_features_df),
    )

    # ── 6. Pipeline ───────────────────────────────────────────────────────
    pipeline = RecommendationPipeline(
        user_tower_model=user_tower_model,
        faiss_index=faiss_index,
        faiss_id_map=faiss_id_map,
        ranker_model=ranker_model,
        redis_client=redis_client,
        user_features_df=user_features_df,
        item_features_df=item_features_df,
        item_metadata_df=item_metadata_df,
    )

    # Stash references on app.state for dependency injection in routes.
    app.state.pipeline = pipeline
    app.state.redis_client = redis_client
    app.state.faiss_index = faiss_index
    app.state.user_tower_model = user_tower_model
    app.state.ranker_model = ranker_model

    logger.info("✅  Recommendation service ready.")
    yield  # ── application is running ──

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("Shutting down — closing Redis …")
    await redis_client.close()
    logger.info("Shutdown complete.")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Recommendation Engine API",
    description=(
        "Two-stage recommendation service for Amazon Clothing, Shoes & Jewelry. "
        "Stage 1: Two-Tower (PyTorch) + FAISS retrieval. "
        "Stage 2: LightGBM re-ranking."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all origins during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the route module.
app.include_router(router)

# Mount the static frontend directory at root.
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="static")
