"""
Services module for unified phishing detection API.
"""
from .cache import cache_service
from .feature_extractor import feature_extraction_service
from .url_analyzer import url_analyzer
from .risk_calculator import RiskCalculator

__all__ = [
    "cache_service",
    "feature_extraction_service",
    "url_analyzer",
    "RiskCalculator"
]
