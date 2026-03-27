"""
Configuration management for Unified Phishing Detection API.
Uses Pydantic Settings for environment-based configuration.
"""
import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Get base directory (portable across systems)
BASE_DIR = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    APP_NAME: str = "Unified Phishing Detection API"
    APP_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )

    # Model Paths (portable with environment variable overrides)
    MODELS_BASE_PATH: str = Field(
        default=str(os.getenv("MODELS_BASE_PATH", str(BASE_DIR / "models"))),
        description="Base directory for all models"
    )
    DAY2_MODEL_PATH: str = Field(
        default="{MODELS_BASE_PATH}/day2_xgboost/xgboost_phishing_classifier.json",
        description="Path to XGBoost model"
    )
    DAY3_MODEL_PATH: str = Field(
        default="{MODELS_BASE_PATH}/day3_distilbert",
        description="Path to DistilBERT model"
    )
    DAY1_PIPELINE_PATH: str = Field(
        default=str(os.getenv("DAY1_PIPELINE_PATH", str(BASE_DIR.parent / "phishing_email_analysis"))),
        description="Path to Day 1 feature pipeline"
    )

    # Model Availability Flags (set based on successful loading)
    XGBOOST_AVAILABLE: bool = False
    TRANSFORMER_AVAILABLE: bool = False
    MULTI_AGENT_AVAILABLE: bool = False

    # Performance Requirements
    MAX_RESPONSE_TIME_MS_CLASSICAL: int = 200  # p95 latency requirement
    MAX_RESPONSE_TIME_MS_TRANSFORMER: int = 1000
    MAX_BATCH_SIZE: int = 100
    MAX_REQUEST_SIZE_MB: int = 10

    # Redis Cache Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_URL_REPUTATION_TTL: int = 3600  # 1 hour
    REDIS_PREDICTION_CACHE_TTL: int = 300  # 5 minutes

    # Multi-Agent Configuration
    MULTI_AGENT_ENABLED: bool = True
    MULTI_AGENT_TIMEOUT_S: int = 60
    MULTI_AGENT_PARALLEL: bool = True

    # GLM Configuration (Zhipu AI)
    GLM_API_KEY: Optional[str] = Field(default=None, description="GLM API key from environment")
    GLM_MODEL: str = "glm-4-flash"
    GLM_API_BASE: str = "https://open.bigmodel.cn/api/paas/v4"
    GLM_TEMPERATURE: float = 0.0
    GLM_MAX_TOKENS: int = 1000

    # Ensemble Weights (must sum to 1.0)
    ENSEMBLE_XGBOOST_WEIGHT: float = 0.4
    ENSEMBLE_TRANSFORMER_WEIGHT: float = 0.4
    ENSEMBLE_MULTI_AGENT_WEIGHT: float = 0.2

    @validator("ENSEMBLE_XGBOOST_WEIGHT", "ENSEMBLE_TRANSFORMER_WEIGHT", "ENSEMBLE_MULTI_AGENT_WEIGHT")
    def validate_weights(cls, v, values):
        """Ensure ensemble weights are valid."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Ensemble weights must be between 0.0 and 1.0")
        return v

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or text
    LOG_FILE: Optional[str] = None

    # Monitoring Configuration
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_PORT: int = 9090
    METRICS_PATH: str = "/metrics"

    # Feature Extraction Configuration
    FEATURE_EXTRACTION_TIMEOUT_MS: int = 100
    ENABLE_FINANCIAL_FEATURES: bool = True

    # Feedback Configuration
    FEEDBACK_STORAGE_PATH: str = "./data/feedback"

    # Security Configuration
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    def get_model_status(self) -> dict:
        """Return the current model availability status."""
        return {
            "xgboost": self.XGBOOST_AVAILABLE,
            "transformer": self.TRANSFORMER_AVAILABLE,
            "multi_agent": self.MULTI_AGENT_AVAILABLE,
            "operating_mode": self._get_operating_mode()
        }

    def _get_operating_mode(self) -> str:
        """Determine operating mode based on available models."""
        available = [
            self.XGBOOST_AVAILABLE,
            self.TRANSFORMER_AVAILABLE,
            self.MULTI_AGENT_AVAILABLE
        ]

        if all(available):
            return "FULL"
        elif self.XGBOOST_AVAILABLE and self.TRANSFORMER_AVAILABLE:
            return "DEGRADED"
        elif self.XGBOOST_AVAILABLE:
            return "MINIMAL"
        else:
            return "UNAVAILABLE"

    def get_ensemble_weights(self) -> dict:
        """Get dynamically adjusted ensemble weights based on available models."""
        weights = {}
        total = 0.0

        if self.XGBOOST_AVAILABLE:
            weights["xgboost"] = self.ENSEMBLE_XGBOOST_WEIGHT
            total += self.ENSEMBLE_XGBOOST_WEIGHT

        if self.TRANSFORMER_AVAILABLE:
            weights["transformer"] = self.ENSEMBLE_TRANSFORMER_WEIGHT
            total += self.ENSEMBLE_TRANSFORMER_WEIGHT

        if self.MULTI_AGENT_AVAILABLE:
            weights["multi_agent"] = self.ENSEMBLE_MULTI_AGENT_WEIGHT
            total += self.ENSEMBLE_MULTI_AGENT_WEIGHT

        # Normalize weights if some models are unavailable
        if total > 0 and total < 1.0:
            weights = {k: v / total for k, v in weights.items()}

        return weights


# Global settings instance
settings = Settings()
