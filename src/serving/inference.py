"""
Full inference pipeline: user-profile → Two-Tower → FAISS → LightGBM → top-K.

Heavy CPU-bound work (PyTorch forward pass, FAISS search, LightGBM predict) is
dispatched via ``run_in_threadpool`` so the async event loop stays responsive.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from fastapi.concurrency import run_in_threadpool

from src.serving.redis_client import RedisClient

logger = logging.getLogger(__name__)


class RecommendationPipeline:
    """Orchestrates retrieval → ranking for a single request.

    Parameters
    ----------
    user_tower_model : torch.nn.Module
        Pre-trained User Tower (expects a dict of float tensors).
    faiss_index : faiss.Index
        FAISS index containing item embeddings.
    faiss_id_map : np.ndarray
        Array mapping FAISS row indices → internal item IDs.
    ranker_model : lgb.Booster
        Trained LightGBM ranker.
    redis_client : RedisClient
        Async Redis helper for profile / feature caching.
    user_features_df : pd.DataFrame
        In-memory user features (fallback when Redis misses).
        Indexed by ``user_id`` (int).
    item_features_df : pd.DataFrame
        In-memory item features (fallback).
        Indexed by ``item_id`` (int).
    item_metadata_df : pd.DataFrame
        Item metadata (``asin``, ``title``, ``price``, ``category``).
        Indexed by ``item_id`` (int).
    """

    def __init__(
        self,
        user_tower_model: torch.nn.Module,
        faiss_index: faiss.Index,
        faiss_id_map: np.ndarray,
        ranker_model: lgb.Booster,
        redis_client: RedisClient,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
        item_metadata_df: pd.DataFrame,
    ) -> None:
        self.user_tower = user_tower_model
        self.user_tower.eval()

        self.faiss_index = faiss_index
        self.faiss_id_map = faiss_id_map
        self.ranker = ranker_model
        self.redis = redis_client

        self.user_features_df = user_features_df
        self.item_features_df = item_features_df
        self.item_metadata_df = item_metadata_df

        # Pre-compute popular items for cold-start fallback (top by interaction count)
        self._popular_item_ids: list[int] = (
            item_features_df.sort_values("review_count", ascending=False)
            .head(100)
            .index.tolist()
            if "review_count" in item_features_df.columns
            else item_features_df.head(100).index.tolist()
        )

        # Build a mapping from string user_id (e.g. reviewer ID) to int index.
        # Assumes user_features_df has a column ``reviewer_id`` or uses the index.
        self._str_to_int_user: dict[str, int] = {}
        if "reviewer_id" in user_features_df.columns:
            self._str_to_int_user = dict(
                zip(
                    user_features_df["reviewer_id"].astype(str),
                    user_features_df.index,
                )
            )
        # Also allow passing the integer id directly as a string.
        for uid in user_features_df.index:
            self._str_to_int_user[str(uid)] = uid

        logger.info(
            "RecommendationPipeline ready — %d users, %d items, FAISS vectors: %d",
            len(user_features_df),
            len(item_features_df),
            faiss_index.ntotal,
        )

    # ── public API ────────────────────────────────────────────────────────

    async def recommend(
        self,
        user_id: str,
        num_candidates: int = 200,
        num_results: int = 10,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Run the full retrieval → ranking pipeline.

        Returns
        -------
        items : list[dict]
            Each dict has keys ``item_id``, ``asin``, ``title``, ``score``,
            ``price``, ``category``.
        timing : dict[str, float]
            ``retrieval_time_ms``, ``ranking_time_ms``, ``total_time_ms``.
        """
        t_start = time.perf_counter()

        # ── 1. Resolve user ──────────────────────────────────────────────
        int_user_id = self._str_to_int_user.get(user_id)
        if int_user_id is None:
            logger.warning("Unknown user_id=%s — returning popular fallback.", user_id)
            items = self._popular_fallback(num_results)
            t_total = (time.perf_counter() - t_start) * 1000
            return items, {
                "retrieval_time_ms": 0.0,
                "ranking_time_ms": 0.0,
                "total_time_ms": t_total,
            }

        # ── 2. Fetch user profile (Redis → in-memory fallback) ───────────
        user_feats = await self.redis.get_user_profile(user_id)
        if user_feats is None and int_user_id in self.user_features_df.index:
            user_feats = self.user_features_df.loc[int_user_id].to_dict()

        if user_feats is None:
            logger.warning("No features for user_id=%s — popular fallback.", user_id)
            items = self._popular_fallback(num_results)
            t_total = (time.perf_counter() - t_start) * 1000
            return items, {
                "retrieval_time_ms": 0.0,
                "ranking_time_ms": 0.0,
                "total_time_ms": t_total,
            }

        # ── 3. Retrieval (User Tower → FAISS) ────────────────────────────
        t_ret = time.perf_counter()
        candidate_ids, candidate_scores = await run_in_threadpool(
            self._retrieve, int_user_id, user_feats, num_candidates
        )
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        # ── 4. Ranking (cross-features → LightGBM) ──────────────────────
        t_rank = time.perf_counter()
        ranked_items = await run_in_threadpool(
            self._rank,
            int_user_id,
            user_feats,
            candidate_ids,
            candidate_scores,
            num_results,
        )
        ranking_ms = (time.perf_counter() - t_rank) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000
        timing = {
            "retrieval_time_ms": round(retrieval_ms, 2),
            "ranking_time_ms": round(ranking_ms, 2),
            "total_time_ms": round(total_ms, 2),
        }
        return ranked_items, timing

    # ── internal steps ────────────────────────────────────────────────────

    def _retrieve(
        self,
        int_user_id: int,
        user_feats: dict[str, float],
        num_candidates: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode user → FAISS ANN search.  Runs in a thread."""
        # Build input tensor from user features (exclude non-numeric / id cols).
        feature_cols = [
            c
            for c in self.user_features_df.columns
            if c not in ("reviewer_id",)
        ]
        feature_values = np.array(
            [user_feats.get(c, 0.0) for c in feature_cols], dtype=np.float32
        )
        # Add user-id int as first element (expected by the tower).
        user_id_tensor = torch.tensor([int_user_id], dtype=torch.long)
        feat_tensor = torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            user_embedding = self.user_tower.encode_user(
                user_id_tensor, feat_tensor
            )  # (1, emb_dim)

        query_vec = user_embedding.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(query_vec)  # cosine search via inner product

        distances, indices = self.faiss_index.search(query_vec, num_candidates)
        # Map FAISS row indices → real item IDs.
        candidate_ids = self.faiss_id_map[indices[0]]
        scores = distances[0]
        return candidate_ids, scores

    def _rank(
        self,
        int_user_id: int,
        user_feats: dict[str, float],
        candidate_ids: np.ndarray,
        retrieval_scores: np.ndarray,
        num_results: int,
    ) -> list[dict[str, Any]]:
        """Build cross-features and score with LightGBM.  Runs in a thread."""
        rows: list[dict[str, Any]] = []
        for cid, ret_score in zip(candidate_ids, retrieval_scores):
            cid = int(cid)
            item_feats = (
                self.item_features_df.loc[cid].to_dict()
                if cid in self.item_features_df.index
                else {}
            )
            row: dict[str, Any] = {}
            # User features (prefixed).
            for k, v in user_feats.items():
                row[f"user_{k}"] = v
            # Item features (prefixed).
            for k, v in item_feats.items():
                row[f"item_{k}"] = v
            # Cross / retrieval feature.
            row["retrieval_score"] = float(ret_score)
            rows.append(row)

        if not rows:
            return self._popular_fallback(num_results)

        ranking_df = pd.DataFrame(rows)
        # Ensure column order matches what the ranker was trained on.
        expected_cols = self.ranker.feature_name()
        for col in expected_cols:
            if col not in ranking_df.columns:
                ranking_df[col] = 0.0
        ranking_df = ranking_df[expected_cols]

        scores = self.ranker.predict(ranking_df)

        top_indices = np.argsort(scores)[::-1][:num_results]
        items: list[dict[str, Any]] = []
        for idx in top_indices:
            cid = int(candidate_ids[idx])
            meta = self._item_meta(cid)
            items.append(
                {
                    "item_id": cid,
                    "asin": meta.get("asin", ""),
                    "title": meta.get("title", ""),
                    "score": round(float(scores[idx]), 6),
                    "price": meta.get("price"),
                    "category": meta.get("category"),
                }
            )
        return items

    # ── helpers ───────────────────────────────────────────────────────────

    def _item_meta(self, item_id: int) -> dict[str, Any]:
        """Look up display metadata for a single item."""
        if item_id in self.item_metadata_df.index:
            return self.item_metadata_df.loc[item_id].to_dict()
        return {}

    def _popular_fallback(self, n: int) -> list[dict[str, Any]]:
        """Return the top-*n* popular items as a cold-start fallback."""
        items: list[dict[str, Any]] = []
        for cid in self._popular_item_ids[:n]:
            meta = self._item_meta(cid)
            items.append(
                {
                    "item_id": cid,
                    "asin": meta.get("asin", ""),
                    "title": meta.get("title", ""),
                    "score": 0.0,
                    "price": meta.get("price"),
                    "category": meta.get("category"),
                }
            )
        return items
