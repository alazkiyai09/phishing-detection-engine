"""
Redis cache service for URL reputation and prediction caching.
"""
import json
import hashlib
from typing import Optional, Any
from datetime import datetime, timedelta

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from src.api.legacy_app.config import settings
from src.api.legacy_app.utils.logger import get_logger

logger = get_logger(__name__)


class CacheService:
    """
    Async Redis cache service.

    Handles URL reputation caching and model prediction caching.
    """

    def __init__(self):
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
        self._connected = False

    async def connect(self):
        """
        Establish Redis connection.
        """
        if self._connected:
            return

        try:
            # Create connection pool
            self.pool = ConnectionPool(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )

            # Create client
            self.client = redis.Redis(connection_pool=self.pool)

            # Test connection
            await self.client.ping()

            self._connected = True
            logger.info("Connected to Redis", extra={
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT,
                "db": settings.REDIS_DB
            })

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}", extra={
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT
            }, exc_info=True)
            self._connected = False
            raise

    async def disconnect(self):
        """
        Close Redis connection.
        """
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()
        self._connected = False
        logger.info("Disconnected from Redis")

    async def ping(self) -> bool:
        """
        Check if Redis connection is alive.

        Returns:
            True if connection is alive, False otherwise
        """
        if not self.client:
            return False

        try:
            await self.client.ping()
            return True
        except Exception:
            return False

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self._connected:
            return None

        try:
            value = await self.client.get(key)
            if value:
                return json.loads(value)
            return None

        except Exception as e:
            logger.warning(f"Cache get failed: {e}", extra={"key": key})
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self._connected:
            return False

        try:
            serialized = json.dumps(value)
            await self.client.set(key, serialized, ex=ttl)
            return True

        except Exception as e:
            logger.warning(f"Cache set failed: {e}", extra={"key": key})
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        if not self._connected:
            return False

        try:
            return await self.client.exists(key) > 0

        except Exception as e:
            logger.warning(f"Cache exists check failed: {e}", extra={"key": key})
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if successful, False otherwise
        """
        if not self._connected:
            return False

        try:
            await self.client.delete(key)
            return True

        except Exception as e:
            logger.warning(f"Cache delete failed: {e}", extra={"key": key})
            return False

    def generate_url_key(self, url: str) -> str:
        """
        Generate cache key for URL reputation.

        Args:
            url: URL to analyze

        Returns:
            Cache key
        """
        # Hash URL to use as key (URLs can be long)
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        return f"url_reputation:{url_hash}"

    def generate_prediction_key(
        self,
        email_content: str,
        model_type: str
    ) -> str:
        """
        Generate cache key for prediction.

        Args:
            email_content: Email content to hash
            model_type: Model type used

        Returns:
            Cache key
        """
        # Hash email content (can be very large)
        content_hash = hashlib.sha256(email_content.encode()).hexdigest()
        return f"prediction:{model_type}:{content_hash}"


# Global cache service instance
cache_service = CacheService()
