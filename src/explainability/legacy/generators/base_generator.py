"""
Base class for explanation generators.

Defines the interface that all explanation generators must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import time

from src.explainability.legacy.utils.data_structures import (
    EmailData,
    ModelOutput,
    Explanation,
    ExplanationType
)


class BaseExplanationGenerator(ABC):
    """
    Abstract base class for explanation generators.

    All generators must inherit from this class and implement
    the generate_explanation method.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.generation_count = 0
        self.total_generation_time_ms = 0.0

    @abstractmethod
    def generate_explanation(
        self,
        email: EmailData,
        model_prediction: ModelOutput,
        **kwargs
    ) -> Explanation:
        """
        Generate explanation for email prediction.

        Args:
            email: Email data to explain
            model_prediction: Model's prediction output
            **kwargs: Additional parameters

        Returns:
            Explanation object
        """
        pass

    def generate_with_timing(
        self,
        email: EmailData,
        model_prediction: ModelOutput,
        **kwargs
    ) -> Explanation:
        """
        Generate explanation and track generation time.

        Args:
            email: Email data to explain
            model_prediction: Model's prediction output
            **kwargs: Additional parameters

        Returns:
            Explanation object with timing information
        """
        start_time = time.time()

        # Generate explanation
        explanation = self.generate_explanation(email, model_prediction, **kwargs)

        # Track timing
        generation_time = time.time() - start_time
        explanation.generation_time_ms = generation_time * 1000

        # Update statistics
        self.generation_count += 1
        self.total_generation_time_ms += explanation.generation_time_ms

        return explanation

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get generation statistics.

        Returns:
            Dictionary with statistics
        """
        avg_time = (
            self.total_generation_time_ms / self.generation_count
            if self.generation_count > 0
            else 0.0
        )

        return {
            'generation_count': self.generation_count,
            'total_time_ms': self.total_generation_time_ms,
            'average_time_ms': avg_time
        }

    def reset_statistics(self) -> None:
        """Reset generation statistics."""
        self.generation_count = 0
        self.total_generation_time_ms = 0.0

    def validate_input(
        self,
        email: EmailData,
        model_prediction: ModelOutput
    ) -> bool:
        """
        Validate input data.

        Args:
            email: Email data to validate
            model_prediction: Model prediction to validate

        Returns:
            True if valid, raises exception otherwise
        """
        if not email or not email.sender:
            raise ValueError("Email must have a valid sender")

        if not model_prediction:
            raise ValueError("Model prediction cannot be None")

        if not 0.0 <= model_prediction.confidence <= 1.0:
            raise ValueError("Model confidence must be between 0.0 and 1.0")

        return True

    def get_supported_explanation_types(self) -> list:
        """
        Get list of supported explanation types.

        Returns:
            List of ExplanationType enum values
        """
        return [ExplanationType.FEATURE_BASED]

    def check_generation_time_requirement(self, generation_time_ms: float) -> bool:
        """
        Check if generation time meets requirement (< 500ms).

        Args:
            generation_time_ms: Generation time in milliseconds

        Returns:
            True if requirement met
        """
        return generation_time_ms < 500.0
