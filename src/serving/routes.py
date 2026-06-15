"""
API route handlers for the recommendation service.

Endpoints
---------
POST /recommend   — Generate personalised recommendations for a user.
GET  /health      — Readiness / health check.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from src.serving.inference import RecommendationPipeline
from src.serving.schemas import (
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
    RecommendedItem,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Dependency helpers ────────────────────────────────────────────────────────


def _get_pipeline(request: Request) -> RecommendationPipeline:
    """Extract the shared pipeline from ``app.state``."""
    pipeline: RecommendationPipeline | None = getattr(
        request.app.state, "pipeline", None
    )
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation pipeline is not initialised yet.",
        )
    return pipeline


# ── POST /recommend ───────────────────────────────────────────────────────────


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(body: RecommendRequest, request: Request) -> RecommendResponse:
    """Return personalised recommendations for the given user.

    * Unknown users receive a popular-item fallback (not a 404) so the
      frontend always gets a valid response.
    * Pipeline or model errors surface as 500s with a descriptive message.
    """
    pipeline = _get_pipeline(request)

    try:
        items, timing = await pipeline.recommend(
            user_id=body.user_id,
            num_candidates=body.num_candidates,
            num_results=body.num_results,
        )
    except Exception as exc:
        logger.exception("Pipeline error for user_id=%s", body.user_id)
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation pipeline failed: {exc}",
        ) from exc

    recommendations = [RecommendedItem(**item) for item in items]

    logger.info(
        "POST /recommend  user=%s  results=%d  total=%.1f ms",
        body.user_id,
        len(recommendations),
        timing["total_time_ms"],
    )

    return RecommendResponse(
        user_id=body.user_id,
        recommendations=recommendations,
        retrieval_time_ms=timing["retrieval_time_ms"],
        ranking_time_ms=timing["ranking_time_ms"],
        total_time_ms=timing["total_time_ms"],
    )


# ── GET /health ───────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """System readiness check.

    Reports Redis connectivity, FAISS index size, and model status.
    """
    redis_ok = False
    faiss_size = 0
    models_loaded = False

    redis_client = getattr(request.app.state, "redis_client", None)
    if redis_client is not None:
        redis_ok = await redis_client.ping()

    faiss_index = getattr(request.app.state, "faiss_index", None)
    if faiss_index is not None:
        faiss_size = faiss_index.ntotal

    user_tower = getattr(request.app.state, "user_tower_model", None)
    ranker = getattr(request.app.state, "ranker_model", None)
    models_loaded = user_tower is not None and ranker is not None

    status = "ok" if (redis_ok and models_loaded and faiss_size > 0) else "degraded"

    return HealthResponse(
        status=status,
        redis_connected=redis_ok,
        faiss_index_size=faiss_size,
        models_loaded=models_loaded,
    )
