"""
Human-Aligned Phishing Explanation System

A system for generating explanations that align with human cognitive processing
patterns for phishing email detection.

Reference: "Eyes on the Phish(er): Towards Understanding Users' Email Processing
Pattern" (CHI 2025, Russello et al.)
"""

__version__ = "1.0.0"
__author__ = "PhD Portfolio Project"

from src.explainability.legacy.generators.base_generator import BaseExplanationGenerator
from src.explainability.legacy.generators.human_aligned import HumanAlignedGenerator

__all__ = [
    "BaseExplanationGenerator",
    "HumanAlignedGenerator",
]
