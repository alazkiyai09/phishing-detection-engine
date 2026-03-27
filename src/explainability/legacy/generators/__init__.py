"""
Explanation generators.

These orchestrate multiple explainers and component analyzers to produce
comprehensive explanations.
"""

from src.explainability.legacy.generators.base_generator import BaseExplanationGenerator
from src.explainability.legacy.generators.human_aligned import HumanAlignedGenerator
from src.explainability.legacy.generators.federated_generator import FederatedExplanationGenerator

__all__ = [
    "BaseExplanationGenerator",
    "HumanAlignedGenerator",
    "FederatedExplanationGenerator",
]
