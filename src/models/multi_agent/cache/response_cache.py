"""
LLM response caching to avoid redundant API calls.
"""
import hashlib
import json
import logging
import time
from typing import Optional, Any
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class ResponseCache:
    """
    Cache for LLM responses to avoid redundant API calls.

    Supports:
    - In-memory caching with LRU eviction
    - TTL-based expiration
    - Deterministic cache keys
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached responses
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._timestamps = {}
        self._access_count = {}

    def _generate_key(self, prompt: str, **kwargs) -> str:
        """
        Generate a deterministic cache key.

        Args:
            prompt: The prompt text
            **kwargs: Additional parameters that affect the response

        Returns:
            Cache key (hash)
        """
        # Create a deterministic string from prompt and kwargs
        cache_input = {
            "prompt": prompt,
            "params": sorted(kwargs.items())
        }
        cache_str = json.dumps(cache_input, sort_keys=True)

        # Generate hash
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def get(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Get cached response if available and not expired.

        Args:
            prompt: The prompt text
            **kwargs: Additional parameters

        Returns:
            Cached response or None
        """
        key = self._generate_key(prompt, **kwargs)

        # Check if key exists
        if key not in self._cache:
            return None

        # Check if expired
        timestamp = self._timestamps.get(key, 0)
        if time.time() - timestamp > self.ttl_seconds:
            # Expired, remove from cache
            del self._cache[key]
            del self._timestamps[key]
            if key in self._access_count:
                del self._access_count[key]
            return None

        # Update access statistics
        self._access_count[key] = self._access_count.get(key, 0) + 1

        logger.debug(f"Cache hit for key {key[:8]}...")
        return self._cache[key]

    def set(self, prompt: str, response: str, **kwargs) -> None:
        """
        Cache a response.

        Args:
            prompt: The prompt text
            response: The response to cache
            **kwargs: Additional parameters
        """
        key = self._generate_key(prompt, **kwargs)

        # Check if we need to evict
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru()

        # Store response
        self._cache[key] = response
        self._timestamps[key] = time.time()
        self._access_count[key] = 1

        logger.debug(f"Cached response for key {key[:8]}...")

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._cache:
            return

        # Find entry with oldest access timestamp
        lru_key = min(self._timestamps.items(), key=lambda x: x[1])[0]

        del self._cache[lru_key]
        del self._timestamps[lru_key]
        if lru_key in self._access_count:
            del self._access_count[lru_key]

        logger.debug(f"Evicted LRU entry {lru_key[:8]}...")

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._timestamps.clear()
        self._access_count.clear()
        logger.info("Cleared cache")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "total_accesses": sum(self._access_count.values())
        }


class CachedLLM:
    """
    Wrapper for LLM backends that adds caching.
    """

    def __init__(self, llm_backend, cache: ResponseCache):
        """
        Initialize cached LLM.

        Args:
            llm_backend: The actual LLM backend to wrap
            cache: Response cache instance
        """
        self.llm = llm_backend
        self.cache = cache

    async def generate(self, prompt: str, **kwargs) -> Any:
        """
        Generate response with caching.

        Args:
            prompt: The prompt
            **kwargs: Additional parameters

        Returns:
            LLM response (cached if available)
        """
        # Check cache first
        cached_response = self.cache.get(prompt, **kwargs)
        if cached_response is not None:
            # Return cached response in the expected format
            # We need to create a response object that looks like it came from the LLM
            from .base_llm import LLMResponse
            return LLMResponse(
                content=cached_response,
                model=self.llm.model_name,
                tokens_used=self.llm.count_tokens(cached_response),
                metadata={"cached": True}
            )

        # Cache miss, call actual LLM
        response = await self.llm.generate(prompt, **kwargs)

        # Cache the response
        self.cache.set(prompt, response.content, **kwargs)

        return response

    def count_tokens(self, text: str) -> int:
        """Delegate to underlying LLM."""
        return self.llm.count_tokens(text)

    def get_model_info(self) -> dict:
        """Delegate to underlying LLM."""
        return self.llm.get_model_info()
