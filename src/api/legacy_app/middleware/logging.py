"""
Structured logging middleware for FastAPI.
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import logging
from typing import Callable

from src.api.legacy_app.utils.logger import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured HTTP request/response logging.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and log with structured context.
        """
        start_time = time.time()

        # Extract request details
        method = request.method
        path = request.url.path
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")

        # Generate request ID
        request_id = request.headers.get("X-Request-ID", f"req_{int(start_time * 1000)}")

        # Log request
        logger.info("Incoming request", extra={
            "request_id": request_id,
            "method": method,
            "path": path,
            "client_ip": client_ip,
            "user_agent": user_agent
        })

        # Add request ID to request state for use in endpoints
        request.state.request_id = request_id

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            logger.info("Request completed", extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2)
            })

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            # Calculate duration for failed requests
            duration_ms = (time.time() - start_time) * 1000

            # Log error
            logger.error("Request failed", extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "duration_ms": round(duration_ms, 2)
            }, exc_info=True)

            raise

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP from request, handling proxy headers.
        """
        # Check for forwarded headers (reverse proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        if request.client:
            return request.client.host

        return "unknown"
