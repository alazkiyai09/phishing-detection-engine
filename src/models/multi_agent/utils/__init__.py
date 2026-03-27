"""Utility functions for caching, rate limiting, and cost tracking."""

from .cache import cached_call
from .cost_tracker import CostTracker
from .rate_limiter import (
    TokenBucketRateLimiter,
    FixedWindowRateLimiter,
    LeakyBucketRateLimiter,
    create_rate_limiter,
)

# Backward-compatible alias expected by legacy imports.
RateLimiter = TokenBucketRateLimiter

__all__ = [
    "cached_call",
    "CostTracker",
    "RateLimiter",
    "TokenBucketRateLimiter",
    "FixedWindowRateLimiter",
    "LeakyBucketRateLimiter",
    "create_rate_limiter",
]
