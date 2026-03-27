"""Feature extraction modules for phishing email analysis."""

from .base import BaseExtractor
from .content_features import ContentFeatureExtractor
from .financial_features import FinancialFeatureExtractor
from .header_features import HeaderFeatureExtractor
from .linguistic_features import LinguisticFeatureExtractor
from .sender_features import SenderFeatureExtractor
from .structural_features import StructuralFeatureExtractor
from .url_features import URLFeatureExtractor

__all__ = [
    "BaseExtractor",
    "URLFeatureExtractor",
    "HeaderFeatureExtractor",
    "SenderFeatureExtractor",
    "ContentFeatureExtractor",
    "StructuralFeatureExtractor",
    "LinguisticFeatureExtractor",
    "FinancialFeatureExtractor",
]
