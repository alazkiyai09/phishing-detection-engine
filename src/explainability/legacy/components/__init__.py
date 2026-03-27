"""
Email component analyzers.

Each analyzer focuses on a specific part of the email following human
cognitive processing order: sender → subject → body → URLs → attachments.
"""

from src.explainability.legacy.components.sender_analyzer import SenderAnalyzer
from src.explainability.legacy.components.subject_analyzer import SubjectAnalyzer
from src.explainability.legacy.components.body_analyzer import BodyAnalyzer
from src.explainability.legacy.components.url_analyzer import URLAnalyzer
from src.explainability.legacy.components.attachment_analyzer import AttachmentAnalyzer

__all__ = [
    "SenderAnalyzer",
    "SubjectAnalyzer",
    "BodyAnalyzer",
    "URLAnalyzer",
    "AttachmentAnalyzer",
]
