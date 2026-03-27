"""
Utility functions for text processing and formatting.
"""

from src.explainability.legacy.utils.text_processing import (
    tokenize_email,
    extract_urls,
    extract_email_addresses,
    normalize_text
)
from src.explainability.legacy.utils.formatters import (
    format_explanation_for_user,
    format_explanation_for_analyst,
    format_confidence_score
)

__all__ = [
    "tokenize_email",
    "extract_urls",
    "extract_email_addresses",
    "normalize_text",
    "format_explanation_for_user",
    "format_explanation_for_analyst",
    "format_confidence_score",
]
