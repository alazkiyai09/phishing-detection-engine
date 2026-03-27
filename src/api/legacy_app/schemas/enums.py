"""
Enumerations for the Phishing Detection API.
"""
from enum import Enum


class Verdict(str, Enum):
    """Phishing detection verdict."""
    PHISHING = "PHISHING"
    LEGITIMATE = "LEGITIMATE"
    SUSPICIOUS = "SUSPICIOUS"


class ModelType(str, Enum):
    """Available model types."""
    XGBOOST = "xgboost"
    TRANSFORMER = "transformer"
    MULTI_AGENT = "multi_agent"
    ENSEMBLE = "ensemble"


class RiskLevel(str, Enum):
    """Risk level categories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
