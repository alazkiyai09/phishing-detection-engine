"""
Rate limiting for API calls to prevent hitting rate limits.
"""
import asyncio
import time
import logging
from typing import Optional
from collections import deque


logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.

    Allows bursts up to bucket capacity, then refills at a steady rate.
    """

    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.

        Args:
            rate: Tokens added per second
            capacity: Maximum bucket capacity (burst size)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False if rate limited
        """
        async with self._lock:
            now = time.time()

            # Refill bucket based on time elapsed
            elapsed = now - self.last_update
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                # Not enough tokens, calculate wait time
                wait_time = (tokens - self.tokens) / self.rate
                logger.info(f"Rate limited, wait {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

                # Refill and acquire
                self.tokens = self.capacity
                self.last_update = time.time()
                self.tokens -= tokens
                return True


class FixedWindowRateLimiter:
    """
    Fixed window rate limiter.

    Allows up to N requests per time window.
    """

    def __init__(self, max_requests: int, window_seconds: int = 60):
        """
        Initialize fixed window limiter.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Window duration in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """
        Acquire permission to make a request.

        Returns:
            True if allowed, waits if rate limited
        """
        async with self._lock:
            now = time.time()

            # Remove old requests outside the window
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()

            # Check if we're at the limit
            if len(self.requests) >= self.max_requests:
                # Calculate wait time until window resets
                oldest_request = self.requests[0]
                wait_time = self.window_seconds - (now - oldest_request)
                logger.info(f"Rate limited, wait {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

                # Clear old requests
                now = time.time()
                while self.requests and self.requests[0] < now - self.window_seconds:
                    self.requests.popleft()

            # Record this request
            self.requests.append(now)
            return True


class LeakyBucketRateLimiter:
    """
    Leaky bucket rate limiter.

    Smooths out request rate, allows bursts but enforces average rate.
    """

    def __init__(self, rate: float, capacity: int):
        """
        Initialize leaky bucket.

        Args:
            rate: Leaky rate (requests per second)
            capacity: Bucket capacity
        """
        self.rate = rate
        self.capacity = capacity
        self.queue = deque()
        self.last_leak = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """
        Acquire permission to make a request.

        Returns:
            True if allowed, waits if bucket is full
        """
        async with self._lock:
            now = time.time()

            # Leak from bucket
            elapsed = now - self.last_leak
            leak_amount = int(elapsed * self.rate)
            if leak_amount > 0:
                for _ in range(min(leak_amount, len(self.queue))):
                    self.queue.popleft()
                self.last_leak = now

            # Check if bucket has space
            if len(self.queue) < self.capacity:
                self.queue.append(now)
                return True
            else:
                # Bucket full, calculate wait time
                wait_time = 1.0 / self.rate
                logger.info(f"Rate limited, wait {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

                # Force leak and acquire
                self.queue.popleft()
                self.queue.append(now)
                return True


def create_rate_limiter(
    limiter_type: str = "token_bucket",
    **kwargs
) -> TokenBucketRateLimiter | FixedWindowRateLimiter | LeakyBucketRateLimiter:
    """
    Factory function to create rate limiter.

    Args:
        limiter_type: Type of rate limiter
        **kwargs: Parameters for the rate limiter

    Returns:
        Rate limiter instance
    """
    if limiter_type == "token_bucket":
        return TokenBucketRateLimiter(
            rate=kwargs.get("rate", 1.0),  # 1 req/sec by default
            capacity=kwargs.get("capacity", 10)  # Burst of 10
        )
    elif limiter_type == "fixed_window":
        return FixedWindowRateLimiter(
            max_requests=kwargs.get("max_requests", 60),
            window_seconds=kwargs.get("window_seconds", 60)
        )
    elif limiter_type == "leaky_bucket":
        return LeakyBucketRateLimiter(
            rate=kwargs.get("rate", 1.0),
            capacity=kwargs.get("capacity", 10)
        )
    else:
        raise ValueError(f"Unknown rate limiter type: {limiter_type}")
