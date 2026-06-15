"""
Pydantic models for the recommendation API request / response contracts.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────


class RecommendRequest(BaseModel):
    """Incoming recommendation request."""

    user_id: str = Field(..., description="Unique user identifier (e.g. Amazon reviewer ID).")
    num_candidates: int = Field(
        200,
        ge=10,
        le=1000,
        description="Number of ANN candidates to retrieve from FAISS.",
    )
    num_results: int = Field(
        10,
        ge=1,
        le=100,
        description="Number of final re-ranked results to return.",
    )


# ── Response ──────────────────────────────────────────────────────────────────


class RecommendedItem(BaseModel):
    """A single recommended item with metadata and ranking score."""

    item_id: int = Field(..., description="Internal integer item ID.")
    asin: str = Field(..., description="Amazon Standard Identification Number.")
    title: str = Field(..., description="Product title.")
    score: float = Field(..., description="Final re-ranking score (LightGBM).")
    price: Optional[float] = Field(None, description="Product price in USD.")
    category: Optional[str] = Field(None, description="Product category.")


class RecommendResponse(BaseModel):
    """Full recommendation response with timing diagnostics."""

    user_id: str
    recommendations: List[RecommendedItem]
    retrieval_time_ms: float = Field(
        ..., description="Time spent on FAISS candidate retrieval (ms)."
    )
    ranking_time_ms: float = Field(
        ..., description="Time spent on LightGBM re-ranking (ms)."
    )
    total_time_ms: float = Field(
        ..., description="Wall-clock time for the full pipeline (ms)."
    )


# ── Health ────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """System health / readiness check."""

    status: str = Field(..., description="'ok' or 'degraded'.")
    redis_connected: bool
    faiss_index_size: int = Field(
        ..., description="Number of vectors currently in the FAISS index."
    )
    models_loaded: bool = Field(
        ..., description="True when both User Tower and ranker are ready."
    )
