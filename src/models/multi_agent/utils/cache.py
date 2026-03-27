"""
Response caching for LLM calls to avoid redundant API calls.
"""
import hashlib
import json
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional
from pathlib import Path


logger = logging.getLogger(__name__)


class ResponseCache:
    """
    LRU cache for LLM responses with TTL support.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of cached responses
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: Dict[str, float] = {}

    def _generate_key(self, prompt: str, **kwargs) -> str:
        """Generate a cache key from prompt and parameters."""
        # Include relevant kwargs in key
        key_data = {"prompt": prompt}
        for k in ["temperature", "max_tokens", "model"]:
            if k in kwargs:
                key_data[k] = kwargs[k]

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Get cached response if available and not expired.

        Args:
            prompt: The prompt that was used
            **kwargs: Additional parameters that affect the response

        Returns:
            Cached response or None
        """
        key = self._generate_key(prompt, **kwargs)

        if key not in self._cache:
            return None

        entry = self._cache[key]

        # Check if expired
        if time.time() - entry["timestamp"] > entry.get("ttl", self.default_ttl):
            del self._cache[key]
            del self._access_order[key]
            logger.debug("Cache entry expired")
            return None

        # Update access time for LRU
        self._access_order[key] = time.time()
        logger.debug(f"Cache hit for key: {key[:16]}...")
        return entry["response"]

    def set(self, prompt: str, response: str, ttl: Optional[int] = None, **kwargs) -> None:
        """
        Cache a response.

        Args:
            prompt: The prompt that generated the response
            response: The LLM response to cache
            ttl: Time-to-live in seconds (uses default if None)
            **kwargs: Additional parameters
        """
        key = self._generate_key(prompt, **kwargs)

        # Evict if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru()

        self._cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "ttl": ttl or self.default_ttl,
        }
        self._access_order[key] = time.time()
        logger.debug(f"Cached response for key: {key[:16]}...")

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._access_order:
            return

        lru_key = min(self._access_order, key=self._access_order.get)
        del self._cache[lru_key]
        del self._access_order[lru_key]
        logger.debug(f"Evicted LRU entry: {lru_key[:16]}...")

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
        }


# Global cache instance
_global_cache: Optional[ResponseCache] = None


def get_global_cache() -> ResponseCache:
    """Get or create the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ResponseCache()
    return _global_cache


def cached_call(ttl: Optional[int] = None, cache: Optional[ResponseCache] = None):
    """
    Decorator for caching LLM calls.

    Args:
        ttl: Time-to-live for cached responses
        cache: Custom cache instance (uses global if None)

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract prompt from args/kwargs
            prompt = kwargs.get("prompt")
            if not prompt and args:
                prompt = args[0]

            if not prompt:
                return await func(*args, **kwargs)

            # Use provided cache or global
            cache_instance = cache or get_global_cache()

            # Check cache
            cached_response = cache_instance.get(prompt, **kwargs)
            if cached_response is not None:
                return cached_response

            # Call function and cache result
            response = await func(*args, **kwargs)
            cache_instance.set(prompt, response, ttl=ttl, **kwargs)

            return response

        return wrapper

    return decorator


class PersistentCache:
    """
    Persistent cache backed by file storage.
    """

    def __init__(self, cache_dir: str = ".cache/llm"):
        """
        Initialize persistent cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = ResponseCache()

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        return self.cache_dir / f"{key}.json"

    def get(self, prompt: str, **kwargs) -> Optional[str]:
        """Get cached response from memory or disk."""
        # Check memory cache first
        response = self.memory_cache.get(prompt, **kwargs)
        if response:
            return response

        # Check disk cache
        key = self.memory_cache._generate_key(prompt, **kwargs)
        cache_path = self._get_cache_path(key)

        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    entry = json.load(f)

                # Check if expired
                if time.time() - entry["timestamp"] > entry.get("ttl", self.memory_cache.default_ttl):
                    cache_path.unlink()
                    return None

                # Load into memory cache
                self.memory_cache._cache[key] = entry
                return entry["response"]
            except (json.JSONDecodeError, KeyError, IOError) as e:
                logger.warning(f"Failed to load cached response: {e}")

        return None

    def set(self, prompt: str, response: str, ttl: Optional[int] = None, **kwargs) -> None:
        """Cache response to memory and disk."""
        # Set in memory
        self.memory_cache.set(prompt, response, ttl=ttl, **kwargs)

        # Write to disk
        key = self.memory_cache._generate_key(prompt, **kwargs)
        cache_path = self._get_cache_path(key)

        entry = {
            "response": response,
            "timestamp": time.time(),
            "ttl": ttl or self.memory_cache.default_ttl,
        }

        try:
            with open(cache_path, "w") as f:
                json.dump(entry, f)
        except IOError as e:
            logger.warning(f"Failed to write cached response: {e}")

    def clear(self) -> None:
        """Clear all cached entries."""
        self.memory_cache.clear()

        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except IOError as e:
                logger.warning(f"Failed to delete cache file: {e}")
