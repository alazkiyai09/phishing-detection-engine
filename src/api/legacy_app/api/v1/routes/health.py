"""
Health check and metrics endpoints.
"""
from fastapi import APIRouter, Response
from datetime import datetime
import time
import asyncio

from src.api.legacy_app.config import settings
from src.api.legacy_app.schemas.responses import HealthResponse
from src.api.legacy_app.middleware.metrics import get_metrics, get_content_type
from src.api.legacy_app.services.cache import cache_service

router = APIRouter()

# Track startup time
startup_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint with model and dependency status.

    Returns overall health status and availability of all components.
    """
    uptime = time.time() - startup_time

    # Check cache connection
    cache_status = "connected"
    try:
        await cache_service.ping()
    except Exception as e:
        cache_status = f"disconnected: {str(e)}"

    # Determine overall health
    model_status = settings.get_model_status()
    operating_mode = model_status["operating_mode"]

    if operating_mode == "UNAVAILABLE":
        health_status = "unhealthy"
    elif operating_mode == "MINIMAL":
        health_status = "degraded"
    else:
        health_status = "healthy"

    # Extract just the boolean model statuses
    models_dict = {
        "xgboost": model_status["xgboost"],
        "transformer": model_status["transformer"],
        "multi_agent": model_status["multi_agent"]
    }

    return HealthResponse(
        status=health_status,
        version=settings.APP_VERSION,
        uptime_seconds=round(uptime, 2),
        models=models_dict,
        cache_status=cache_status,
        dependencies={
            "redis": cache_status,
            "xgboost": "available" if settings.XGBOOST_AVAILABLE else "unavailable",
            "transformer": "available" if settings.TRANSFORMER_AVAILABLE else "unavailable",
            "multi_agent": "available" if settings.MULTI_AGENT_AVAILABLE else "unavailable"
        }
    )


@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.
    """
    metrics_data = get_metrics()
    return Response(
        content=metrics_data,
        media_type=get_content_type(),
        headers={
            "Content-Type": "text/plain; version=0.0.4; charset=utf-8"
        }
    )


@router.get("/readiness")
async def readiness():
    """
    Readiness probe for Kubernetes.

    Returns 200 if the service is ready to accept traffic.
    """
    # Check if at least XGBoost is available
    if not settings.XGBOOST_AVAILABLE:
        return Response(
            content='{"ready": false, "reason": "No models available"}',
            status_code=503,
            media_type="application/json"
        )

    return Response(
        content='{"ready": true}',
        status_code=200,
        media_type="application/json"
    )


@router.get("/liveness")
async def liveness():
    """
    Liveness probe for Kubernetes.

    Returns 200 if the service is alive.
    """
    return Response(
        content='{"alive": true}',
        status_code=200,
        media_type="application/json"
    )
