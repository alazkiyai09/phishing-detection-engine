"""
Request schemas for the Phishing Detection API.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from .enums import ModelType


class EmailAnalysisRequest(BaseModel):
    """Request for full email analysis."""

    # Email can be provided as raw text (EML format) or pre-parsed
    raw_email: Optional[str] = Field(
        None,
        description="Raw RFC 822 email content (EML format)"
    )
    parsed_email: Optional[Dict[str, Any]] = Field(
        None,
        description="Pre-parsed email with headers, body, urls, etc."
    )

    # Analysis options
    model_type: ModelType = Field(
        default=ModelType.ENSEMBLE,
        description="Which model(s) to use for prediction"
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use cached predictions"
    )
    include_explanation: bool = Field(
        default=True,
        description="Include detailed explanation in response"
    )
    timeout_ms: Optional[int] = Field(
        None,
        description="Custom timeout in milliseconds (overrides default)"
    )

    @validator('parsed_email')
    def validate_email_input(cls, v, values):
        """Ensure either raw_email or parsed_email is provided."""
        if v is None and values.get('raw_email') is None:
            raise ValueError("Either raw_email or parsed_email must be provided")
        return v

    @validator('timeout_ms')
    def validate_timeout(cls, v):
        """Validate timeout is reasonable."""
        if v is not None and v < 100:
            raise ValueError("Timeout must be at least 100ms")
        if v is not None and v > 30000:
            raise ValueError("Timeout cannot exceed 30 seconds")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "raw_email": "Received: from mail.example.com...\nSubject: URGENT...",
                "model_type": "ensemble",
                "use_cache": True,
                "include_explanation": True
            }
        }


class URLAnalysisRequest(BaseModel):
    """Request for URL-only analysis (quick check)."""

    url: str = Field(
        ...,
        description="URL to analyze",
        min_length=1
    )
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context (e.g., email subject, sender)"
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use URL reputation cache"
    )

    @validator('url')
    def validate_url(cls, v):
        """Basic URL validation."""
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "url": "http://chase-secure-portal.xyz/login",
                "context": {
                    "sender": "security@chase-secure-portal.xyz",
                    "subject": "Account Verification Required"
                }
            }
        }


class BatchAnalysisRequest(BaseModel):
    """Request for batch email analysis."""

    emails: List[Dict[str, Any]] = Field(
        ...,
        description="List of emails/URLs to analyze. For URL analysis: {'url': 'http://...', 'context': {...}, 'use_cache': true}",
        min_length=1,
        max_length=100
    )
    model_type: ModelType = Field(
        default=ModelType.ENSEMBLE,
        description="Which model(s) to use for all predictions"
    )
    parallel: bool = Field(
        default=True,
        description="Process emails in parallel"
    )

    @validator('emails')
    def validate_batch_size(cls, v):
        """Validate batch size limits."""
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 emails")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "emails": [
                    {"url": "http://example.com", "context": {}},
                    {"raw_email": "..."}
                ],
                "model_type": "ensemble",
                "parallel": True
            }
        }


class FeedbackRequest(BaseModel):
    """Request for submitting feedback on predictions."""

    email_id: str = Field(
        ...,
        description="Unique identifier for the email"
    )
    predicted_verdict: str = Field(
        ...,
        description="Model's prediction"
    )
    actual_verdict: str = Field(
        ...,
        description="Ground truth verdict"
    )
    model_used: str = Field(
        ...,
        description="Which model made the prediction"
    )
    feedback_type: str = Field(
        ...,
        description="Type of feedback (false_positive, false_negative, true_positive, true_negative)"
    )
    user_comments: Optional[str] = Field(
        None,
        description="Additional comments from the user"
    )

    @validator('feedback_type')
    def validate_feedback_type(cls, v):
        """Validate feedback type."""
        valid_types = [
            "false_positive",
            "false_negative",
            "true_positive",
            "true_negative"
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid feedback_type. Must be one of: {valid_types}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "email_id": "abc123",
                "predicted_verdict": "PHISHING",
                "actual_verdict": "LEGITIMATE",
                "model_used": "xgboost",
                "feedback_type": "false_positive",
                "user_comments": "Legitimate bank email"
            }
        }
