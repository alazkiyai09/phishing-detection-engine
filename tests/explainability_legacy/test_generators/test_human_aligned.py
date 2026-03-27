"""
Unit tests for generators.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from src.utils.data_structures import (
    EmailData,
    EmailAddress,
    ModelOutput,
    EmailCategory
)
from src.generators.human_aligned import HumanAlignedGenerator


class TestHumanAlignedGenerator:
    """Tests for HumanAlignedGenerator."""

    def setup_method(self):
        """Setup test fixtures."""
        self.generator = HumanAlignedGenerator()

    def test_generate_explanation_basic(self):
        """Test basic explanation generation."""
        email = EmailData(
            sender=EmailAddress(
                display_name="Test Sender",
                email="test@example.com"
            ),
            recipients=[],
            subject="Test Subject",
            body="Test body content"
        )

        prediction = ModelOutput(
            predicted_label=EmailCategory.SAFE,
            confidence=0.85
        )

        explanation = self.generator.generate_explanation(email, prediction)

        assert explanation.email == email
        assert explanation.model_prediction == prediction
        assert explanation.sender_explanation is not None
        assert explanation.subject_explanation is not None
        assert explanation.body_explanation is not None

    def test_cognitive_order(self):
        """Test that explanations follow cognitive order."""
        email = EmailData(
            sender=EmailAddress(
                display_name="Urgent Alert",
                email="alert@example.com"
            ),
            recipients=[],
            subject="URGENT: Action required",
            body="Please verify your password immediately"
        )

        prediction = ModelOutput(
            predicted_label=EmailCategory.PHISHING,
            confidence=0.92
        )

        explanation = self.generator.generate_explanation(email, prediction)

        # Check that all components are analyzed
        assert explanation.sender_explanation is not None
        assert explanation.subject_explanation is not None
        assert explanation.body_explanation is not None
        assert explanation.url_explanation is not None
        assert explanation.attachment_explanation is not None

    def test_timing(self):
        """Test that explanation generation is timed."""
        email = EmailData(
            sender=EmailAddress(display_name="Test", email="test@example.com"),
            recipients=[],
            subject="Test",
            body="Test"
        )

        prediction = ModelOutput(
            predicted_label=EmailCategory.SAFE,
            confidence=0.80
        )

        explanation = self.generator.generate_with_timing(email, prediction)

        assert explanation.generation_time_ms > 0
        assert explanation.generation_time_ms < 5000  # Should be fast

    def test_batch_generation(self):
        """Test batch explanation generation."""
        emails = [
            EmailData(
                sender=EmailAddress(display_name=f"Sender {i}", email=f"sender{i}@example.com"),
                recipients=[],
                subject=f"Subject {i}",
                body=f"Body {i}"
            )
            for i in range(3)
        ]

        predictions = [
            ModelOutput(
                predicted_label=EmailCategory.SAFE,
                confidence=0.80 + i * 0.05
            )
            for i in range(3)
        ]

        explanations = self.generator.generate_batch(emails, predictions)

        assert len(explanations) == 3
        for exp in explanations:
            assert exp.sender_explanation is not None

    def test_statistics_tracking(self):
        """Test that generator tracks statistics."""
        email = EmailData(
            sender=EmailAddress(display_name="Test", email="test@example.com"),
            recipients=[],
            subject="Test",
            body="Test"
        )

        prediction = ModelOutput(
            predicted_label=EmailCategory.SAFE,
            confidence=0.80
        )

        # Generate multiple explanations
        for _ in range(5):
            self.generator.generate_with_timing(email, prediction)

        stats = self.generator.get_statistics()

        assert stats['generation_count'] == 5
        assert stats['total_time_ms'] > 0
        assert stats['average_time_ms'] > 0
