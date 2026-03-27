"""
FastAPI application factory for Unified Phishing Detection API.
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
import time
from datetime import datetime

from src.api.legacy_app.config import settings
from src.api.legacy_app.api.v1.routes import analyze, models, feedback, health
from src.api.legacy_app.middleware.logging import LoggingMiddleware
from src.api.legacy_app.middleware.metrics import MetricsMiddleware
from src.api.legacy_app.middleware.request_size import RequestSizeMiddleware
from src.api.legacy_app.utils.logger import setup_logger

# Setup structured logging
logger = setup_logger(__name__, settings.LOG_LEVEL, settings.LOG_FORMAT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting Unified Phishing Detection API...", extra={
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    })

    # Initialize models
    from src.api.legacy_app.services.model_service import initialize_models
    initialize_models()

    model_status = {
        "xgboost": settings.XGBOOST_AVAILABLE,
        "transformer": settings.TRANSFORMER_AVAILABLE,
        "multi_agent": settings.MULTI_AGENT_AVAILABLE
    }

    logger.info("Model loading complete", extra=model_status)

    # Log operating mode
    operating_mode = settings._get_operating_mode()
    logger.info(f"API running in {operating_mode} mode", extra={
        "operating_mode": operating_mode,
        "xgboost_available": settings.XGBOOST_AVAILABLE,
        "transformer_available": settings.TRANSFORMER_AVAILABLE,
        "multi_agent_available": settings.MULTI_AGENT_AVAILABLE
    })

    # Give control to the application
    yield

    # Shutdown
    logger.info("Shutting down Unified Phishing Detection API...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="""
        Unified Phishing Detection API serving multiple ML models with ensemble capability.

        ## Models
        - **XGBoost**: Classical ML with 60+ features (<200ms latency)
        - **Transformer**: DistilBERT with special token injection (<1s latency)
        - **Multi-Agent**: GLM-powered agent analysis (3-5s latency)
        - **Ensemble**: Weighted combination of available models

        ## Features
        - URL reputation caching
        - Batch processing (max 100 emails)
        - Graceful degradation
        - Comprehensive monitoring
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        lifespan=lifespan
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware
    app.add_middleware(RequestSizeMiddleware, max_size_mb=10)  # 10MB max request size
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(MetricsMiddleware)

    # Register exception handlers
    register_exception_handlers(app)

    # Register routes
    register_routes(app)

    return app


def register_exception_handlers(app: FastAPI):
    """
    Register global exception handlers.
    """

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
    ):
        """Handle Pydantic validation errors."""
        logger.warning("Validation error", extra={
            "path": request.url.path,
            "errors": exc.errors()
        })

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "ValidationError",
                "message": "Invalid request data",
                "detail": exc.errors(),
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url.path),
                "status_code": 422
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unhandled exceptions."""
        logger.error(f"Unhandled exception: {exc}", extra={
            "path": request.url.path,
            "error_type": type(exc).__name__,
            "error_message": str(exc)
        }, exc_info=True)

        # Don't expose internal errors in production
        message = "Internal server error" if not settings.DEBUG else str(exc)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url.path),
                "status_code": 500
            }
        )


def register_routes(app: FastAPI):
    """
    Register all API routes.
    """
    # Health and metrics (no version prefix)
    app.include_router(health.router, tags=["Health"])

    # API v1 routes
    app.include_router(
        analyze.router,
        prefix=settings.API_V1_PREFIX,
        tags=["Analysis"]
    )
    app.include_router(
        models.router,
        prefix=settings.API_V1_PREFIX,
        tags=["Models"]
    )
    app.include_router(
        feedback.router,
        prefix=settings.API_V1_PREFIX,
        tags=["Feedback"]
    )

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with API information."""
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "operational",
            "documentation": "/docs",
            "health": "/health",
            "metrics": "/metrics",
            "api_v1": settings.API_V1_PREFIX
        }


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_config=None  # Use our custom logging
    )
