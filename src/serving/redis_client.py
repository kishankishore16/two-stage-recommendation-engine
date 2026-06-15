"""
Async Redis client for user-profile and item-feature caching.

Uses ``redis.asyncio`` (the official asyncio driver shipped with redis-py ≥ 4.2).
All feature values are stored as Redis hashes with string-encoded float values.
"""

from __future__ import annotations

import logging
from typing import Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class RedisClient:
    """Thin async wrapper around a Redis connection pool.

    Parameters
    ----------
    host : str
        Redis server hostname.
    port : int
        Redis server port.
    db : int
        Redis database index.
    max_connections : int
        Maximum connections in the async pool.
    """

    # ── lifecycle ─────────────────────────────────────────────────────────

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        max_connections: int = 50,
    ) -> None:
        self._pool = aioredis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            max_connections=max_connections,
            decode_responses=True,
        )
        self._redis: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        """Initialize the Redis connection from the pool."""
        self._redis = aioredis.Redis(connection_pool=self._pool)
        logger.info("Redis client connected (%s:%s/%s)", self._pool.connection_kwargs.get("host"), self._pool.connection_kwargs.get("port"), self._pool.connection_kwargs.get("db"))

    async def close(self) -> None:
        """Gracefully close the connection pool."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        await self._pool.disconnect()
        logger.info("Redis client closed.")

    # ── helpers ───────────────────────────────────────────────────────────

    def _ensure_connected(self) -> aioredis.Redis:
        if self._redis is None:
            raise RuntimeError("RedisClient is not connected. Call connect() first.")
        return self._redis

    # ── user profiles ─────────────────────────────────────────────────────

    async def get_user_profile(self, user_id: str) -> Optional[dict]:
        """Retrieve a user feature hash from ``user:{user_id}``.

        Returns
        -------
        dict[str, float] | None
            Mapping of feature-name → float value, or *None* if the key
            does not exist.
        """
        r = self._ensure_connected()
        key = f"user:{user_id}"
        data = await r.hgetall(key)
        if not data:
            return None
        # All values were stored as strings; cast back to float.
        return {k: float(v) for k, v in data.items()}

    async def set_user_profile(
        self, user_id: str, features: dict, ttl: int = 3600
    ) -> None:
        """Store a user feature dict as a Redis hash with expiry.

        Parameters
        ----------
        user_id : str
            User identifier.
        features : dict
            Feature-name → numeric-value mapping.
        ttl : int
            Time-to-live in seconds (default 1 h).
        """
        r = self._ensure_connected()
        key = f"user:{user_id}"
        # Convert all values to strings for Redis hash storage.
        str_features = {k: str(v) for k, v in features.items()}
        await r.hset(key, mapping=str_features)
        if ttl > 0:
            await r.expire(key, ttl)

    # ── item features ─────────────────────────────────────────────────────

    async def get_item_features(self, item_id: int) -> Optional[dict]:
        """Retrieve an item feature hash from ``item:{item_id}``."""
        r = self._ensure_connected()
        key = f"item:{item_id}"
        data = await r.hgetall(key)
        if not data:
            return None
        return {k: float(v) for k, v in data.items()}

    async def get_item_features_batch(
        self, item_ids: list[int]
    ) -> list[Optional[dict]]:
        """Pipeline-batch fetch item features for *item_ids*.

        Returns a list aligned with *item_ids*; missing items are ``None``.
        """
        r = self._ensure_connected()

        async with r.pipeline(transaction=False) as pipe:
            for iid in item_ids:
                pipe.hgetall(f"item:{iid}")
            results = await pipe.execute()

        out: list[Optional[dict]] = []
        for raw in results:
            if raw:
                out.append({k: float(v) for k, v in raw.items()})
            else:
                out.append(None)
        return out

    async def set_item_features(
        self, item_id: int, features: dict, ttl: int = 86400
    ) -> None:
        """Store item features as a Redis hash with expiry."""
        r = self._ensure_connected()
        key = f"item:{item_id}"
        str_features = {k: str(v) for k, v in features.items()}
        await r.hset(key, mapping=str_features)
        if ttl > 0:
            await r.expire(key, ttl)

    # ── health ────────────────────────────────────────────────────────────

    async def ping(self) -> bool:
        """Return ``True`` if the Redis server responds to PING."""
        try:
            r = self._ensure_connected()
            return await r.ping()
        except Exception:
            logger.warning("Redis ping failed", exc_info=True)
            return False
