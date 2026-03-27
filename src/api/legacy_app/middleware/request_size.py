"""
Request size validation middleware.

Prevents excessively large requests that could cause memory issues or DoS attacks.
"""
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate request size before processing.

    Checks Content-Length header and rejects requests exceeding maximum size.
    """

    def __init__(self, app, max_size_mb: int = 10):
        """
        Initialize request size middleware.

        Args:
            app: ASGI application
            max_size_mb: Maximum request size in megabytes
        """
        super().__init__(app)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_size_mb = max_size_mb
        logger.info(f"RequestSizeMiddleware initialized: max_size={max_size_mb}MB")

    async def dispatch(self, request: Request, call_next):
        """
        Process request and validate size.

        Args:
            request: Incoming request
            call_next: Next middleware or route handler

        Returns:
            Response or error if request too large
        """
        # Check Content-Length header if present
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                content_length_int = int(content_length)
                if content_length_int > self.max_size_bytes:
                    logger.warning(
                        f"Request too large: {content_length_int} bytes "
                        f"(max: {self.max_size_bytes} bytes)",
                        extra={"path": request.url.path, "client": request.client}
                    )
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "error": "RequestTooLarge",
                            "message": f"Request exceeds maximum size of {self.max_size_mb}MB. "
                                     f"Got {content_length_int / (1024*1024):.1f}MB",
                            "timestamp": datetime.utcnow().isoformat(),
                            "path": str(request.url.path),
                            "status_code": 413,
                            "max_size_mb": self.max_size_mb,
                            "actual_size_mb": round(content_length_int / (1024*1024), 2)
                        }
                    )
            except ValueError:
                # Invalid Content-Length header, let request proceed
                logger.warning("Invalid Content-Length header", extra={"path": request.url.path})

        # Process request
        response = await call_next(request)

        # Log request size for monitoring
        if content_length:
            logger.debug(
                f"Request size: {content_length} bytes",
                extra={"path": request.url.path, "size_bytes": int(content_length)}
            )

        return response
