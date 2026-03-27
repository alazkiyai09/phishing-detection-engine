"""Input validation utilities for API endpoints."""

from typing import List, Any, Dict
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


async def validate_batch_size(emails: List[Any], max_size: int = 100) -> None:
    """Validate batch size limit.

    Args:
        emails: List of email data
        max_size: Maximum allowed batch size

    Raises:
        HTTPException: If batch size exceeds limit
    """
    if len(emails) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum of {max_size}. Got {len(emails)} emails"
        )
    logger.info(f"Batch size validation passed: {len(emails)} emails")


def validate_email_dict(email_data: Dict[str, Any]) -> None:
    """Validate email dictionary has required fields.

    Args:
        email_data: Email dictionary to validate

    Raises:
        HTTPException: If required fields missing
    """
    if not isinstance(email_data, dict):
        raise HTTPException(
            status_code=400,
            detail="Email data must be a dictionary"
        )

    # Check for at least one valid field
    has_content = (
        'raw_email' in email_data or
        'parsed_email' in email_data or
        'url' in email_data
    )

    if not has_content:
        raise HTTPException(
            status_code=400,
            detail="Email data must contain one of: raw_email, parsed_email, or url"
        )


def validate_request_size(
    content_length: int,
    max_size_mb: int = 10
) -> None:
    """Validate request size limit.

    Args:
        content_length: Content-Length header value
        max_size_mb: Maximum size in megabytes

    Raises:
        HTTPException: If request exceeds limit
    """
    max_bytes = max_size_mb * 1024 * 1024

    if content_length > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Request too large. Maximum {max_size_mb}MB, got {content_length / (1024*1024):.1f}MB"
        )
