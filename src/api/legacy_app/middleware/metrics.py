"""
Prometheus metrics middleware for FastAPI.
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.registry import CollectorRegistry
import time
import logging

from src.api.legacy_app.config import settings

logger = logging.getLogger(__name__)

# Create metrics registry
registry = CollectorRegistry()

# Define metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
    registry=registry
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
    registry=registry
)

http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "HTTP requests currently in progress",
    ["method", "endpoint"],
    registry=registry
)

model_predictions_total = Counter(
    "model_predictions_total",
    "Total model predictions",
    ["model_type", "verdict"],
    registry=registry
)

model_prediction_duration_seconds = Histogram(
    "model_prediction_duration_seconds",
    "Model prediction latency",
    ["model_type"],
    buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
    registry=registry
)

cache_hits_total = Counter(
    "cache_hits_total",
    "Total cache hits",
    ["cache_type"],  # url_reputation, prediction
    registry=registry
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total cache misses",
    ["cache_type"],
    registry=registry
)

model_errors_total = Counter(
    "model_errors_total",
    "Total model errors",
    ["model_type", "error_type"],
    registry=registry
)

feedback_submitted_total = Counter(
    "feedback_submitted_total",
    "Total feedback submitted",
    ["feedback_type"],
    registry=registry
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting Prometheus metrics.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request and collect metrics.
        """
        start_time = time.time()
        method = request.method
        path = request.url.path

        # Ignore metrics endpoint itself
        if path == "/metrics":
            return await call_next(request)

        # Increment in-progress gauge
        http_requests_in_progress.labels(
            method=method,
            endpoint=path
        ).inc()

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Record metrics
            http_requests_total.labels(
                method=method,
                endpoint=path,
                status_code=response.status_code
            ).inc()

            http_request_duration_seconds.labels(
                method=method,
                endpoint=path
            ).observe(duration)

            return response

        except Exception as e:
            # Calculate duration for failed requests
            duration = time.time() - start_time

            # Record error metrics
            http_requests_total.labels(
                method=method,
                endpoint=path,
                status_code=500
            ).inc()

            http_request_duration_seconds.labels(
                method=method,
                endpoint=path
            ).observe(duration)

            raise

        finally:
            # Decrement in-progress gauge
            http_requests_in_progress.labels(
                method=method,
                endpoint=path
            ).dec()


def record_model_prediction(model_type: str, verdict: str, duration_ms: float):
    """
    Record model prediction metrics.

    Args:
        model_type: Type of model (xgboost, transformer, multi_agent)
        verdict: Prediction verdict (PHISHING, LEGITIMATE, SUSPICIOUS)
        duration_ms: Prediction duration in milliseconds
    """
    model_predictions_total.labels(
        model_type=model_type,
        verdict=verdict
    ).inc()

    model_prediction_duration_seconds.labels(
        model_type=model_type
    ).observe(duration_ms / 1000.0)


def record_cache_hit(cache_type: str):
    """
    Record cache hit.

    Args:
        cache_type: Type of cache (url_reputation, prediction)
    """
    cache_hits_total.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str):
    """
    Record cache miss.

    Args:
        cache_type: Type of cache (url_reputation, prediction)
    """
    cache_misses_total.labels(cache_type=cache_type).inc()


def record_model_error(model_type: str, error_type: str):
    """
    Record model error.

    Args:
        model_type: Type of model
        error_type: Type of error
    """
    model_errors_total.labels(
        model_type=model_type,
        error_type=error_type
    ).inc()


def record_feedback(feedback_type: str):
    """
    Record feedback submission.

    Args:
        feedback_type: Type of feedback (false_positive, false_negative, etc.)
    """
    feedback_submitted_total.labels(feedback_type=feedback_type).inc()


def get_metrics() -> bytes:
    """
    Get Prometheus metrics in text format.

    Returns:
        Metrics in Prometheus text format
    """
    return generate_latest(registry)


def get_content_type() -> str:
    """
    Get content type for metrics endpoint.

    Returns:
        Content type string
    """
    return CONTENT_TYPE_LATEST
