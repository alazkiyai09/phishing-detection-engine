"""
Response schemas for the Phishing Detection API.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from .enums import Verdict, ModelType, RiskLevel


class ModelPrediction(BaseModel):
    """Individual model prediction result."""
    model_name: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    risk_score: int = Field(ge=0, le=100, description="Risk score 0-100")
    processing_time_ms: float


class FeatureBreakdown(BaseModel):
    """Detailed feature analysis breakdown."""
    url_risk: Optional[Dict[str, Any]] = None
    content_risk: Optional[Dict[str, Any]] = None
    header_risk: Optional[Dict[str, Any]] = None
    financial_indicators: Optional[Dict[str, Any]] = None


class AnalysisResponse(BaseModel):
    """Standard response for email/URL analysis."""

    # Basic prediction info
    email_id: str = Field(description="Unique identifier for this request")
    verdict: Verdict = Field(description="Final classification verdict")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    risk_score: int = Field(ge=0, le=100, description="Risk score (0-100)")
    risk_level: RiskLevel = Field(description="Risk category")

    # Model info
    model_used: str = Field(description="Which model(s) were used")
    individual_predictions: Optional[List[ModelPrediction]] = Field(
        None,
        description="Individual model predictions (for ensemble)"
    )

    # Detailed analysis
    analysis: Optional[FeatureBreakdown] = Field(
        None,
        description="Detailed feature breakdown"
    )
    explanation: Optional[str] = Field(
        None,
        description="Human-readable explanation"
    )

    # Performance metadata
    processing_time_ms: float = Field(description="Total processing time")
    cache_hit: bool = Field(default=False, description="Whether result came from cache")
    timestamp: str = Field(description="ISO timestamp of analysis")

    # Error handling
    error: Optional[str] = Field(None, description="Error message if analysis failed")
    warnings: Optional[List[str]] = Field(None, description="Warning messages")

    class Config:
        json_schema_extra = {
            "example": {
                "email_id": "req_abc123",
                "verdict": "PHISHING",
                "confidence": 0.95,
                "risk_score": 92,
                "risk_level": "critical",
                "model_used": "ensemble",
                "individual_predictions": [
                    {
                        "model_name": "xgboost",
                        "verdict": "PHISHING",
                        "confidence": 0.93,
                        "risk_score": 88,
                        "processing_time_ms": 45.2
                    },
                    {
                        "model_name": "transformer",
                        "verdict": "PHISHING",
                        "confidence": 0.97,
                        "risk_score": 95,
                        "processing_time_ms": 234.5
                    }
                ],
                "analysis": {
                    "url_risk": {
                        "has_suspicious_tld": True,
                        "has_ip_url": False,
                        "suspicious_urls": ["chase-secure-portal.xyz"]
                    },
                    "content_risk": {
                        "urgency_score": 0.8,
                        "credential_harvesting": True
                    },
                    "financial_indicators": {
                        "bank_impersonation": True,
                        "bank_name": "Chase",
                        "similarity_score": 0.92
                    }
                },
                "explanation": "This email contains multiple indicators of phishing: suspicious domain impersonating Chase bank, urgency language, and credential harvesting requests.",
                "processing_time_ms": 295.3,
                "cache_hit": False,
                "timestamp": "2026-01-29T12:34:56Z"
            }
        }


class BatchAnalysisResponse(BaseModel):
    """Response for batch analysis."""

    batch_id: str = Field(description="Unique identifier for this batch")
    results: List[AnalysisResponse] = Field(description="Individual email analyses")
    summary: Dict[str, Any] = Field(
        description="Batch summary statistics"
    )
    total_processing_time_ms: float = Field(description="Total time for batch")
    successful_count: int = Field(description="Number of successful analyses")
    failed_count: int = Field(description="Number of failed analyses")

    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "batch_xyz789",
                "results": [],
                "summary": {
                    "total_emails": 100,
                    "phishing_count": 23,
                    "legitimate_count": 65,
                    "suspicious_count": 12,
                    "avg_risk_score": 45.6,
                    "avg_confidence": 0.87
                },
                "total_processing_time_ms": 15234.5,
                "successful_count": 98,
                "failed_count": 2
            }
        }


class ModelInfo(BaseModel):
    """Information about an available model."""
    name: str = Field(description="Model identifier")
    type: ModelType = Field(description="Model type")
    available: bool = Field(description="Whether model is currently available")
    description: str = Field(description="Model description")
    version: str = Field(description="Model version")
    performance: Optional[Dict[str, float]] = Field(
        None,
        description="Performance metrics (AUPRC, accuracy, etc.)"
    )
    avg_latency_ms: Optional[float] = Field(
        None,
        description="Average inference latency"
    )
    last_trained: Optional[str] = Field(
        None,
        description="Last training date"
    )


class ModelsListResponse(BaseModel):
    """Response listing all available models."""
    models: List[ModelInfo] = Field(description="List of models")
    operating_mode: str = Field(
        description="Current operating mode (FULL, DEGRADED, MINIMAL, UNAVAILABLE)"
    )
    ensemble_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Current ensemble weights"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Health status: healthy, degraded, unhealthy")
    version: str = Field(description="API version")
    uptime_seconds: float = Field(description="Server uptime")
    models: Dict[str, bool] = Field(description="Model availability status")
    cache_status: str = Field(description="Cache connection status")
    dependencies: Dict[str, str] = Field(description="Dependency status")


class FeedbackResponse(BaseModel):
    """Response for feedback submission."""
    success: bool
    message: str
    feedback_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(description="Error timestamp")
    path: str = Field(description="Request path")
    status_code: int = Field(description="HTTP status code")
