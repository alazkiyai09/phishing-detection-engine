"""Unit tests for the phishing detection agents."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.models.schemas import AgentOutput, EmailInput
from src.llm.mock_backend import MockLLM
from src.llm.base_llm import LLMResponse
from src.agents.url_analyst import URLAnalyst
from src.agents.content_analyst import ContentAnalyst
from src.agents.header_analyst import HeaderAnalyst
from src.agents.visual_analyst import VisualAnalyst


# Sample email data for testing
SAMPLE_EMAIL = EmailInput(
    subject="URGENT: Verify Your Account Now",
    sender="support@secure-login-verify.com",
    body="""
Dear Customer,

Your account will be suspended within 24 hours unless you take immediate action.
Please click here to verify your account: http://secure-login-verify.com/login

Enter your password and account number to confirm your identity.

Thank you,
Customer Support Team
""",
    urls=["http://secure-login-verify.com/login"],
    headers={
        "Received-SPF": "fail",
        "Reply-To": "phisher@evil.com",
    },
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = MockLLM(model_name="mock-model")

    # Override the generate method to return predictable responses
    async def mock_generate(prompt, **kwargs):
        return LLMResponse(
            content='{"is_phishing": true, "confidence": 0.85, "reasoning": "Test reasoning", "evidence": ["Test evidence"]}',
            model="mock-model",
            tokens_used=100,
        )

    llm.generate = mock_generate
    return llm


@pytest.mark.asyncio
async def test_url_analyst(mock_llm):
    """Test URL Analyst agent."""
    agent = URLAnalyst(llm=mock_llm)
    result = await agent.analyze(SAMPLE_EMAIL)

    assert isinstance(result, AgentOutput)
    assert result.agent_name == "url_analyst"
    assert result.verdict in ["phishing", "legitimate", "suspicious"]
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.reasoning) > 0
    assert isinstance(result.evidence, list)


@pytest.mark.asyncio
async def test_url_analyst_no_urls(mock_llm):
    """Test URL Analyst with email containing no URLs."""
    email_no_urls = EmailInput(
        subject="Test",
        sender="test@example.com",
        body="No URLs here",
        urls=[],
    )
    agent = URLAnalyst(llm=mock_llm)
    result = await agent.analyze(email_no_urls)

    assert result.verdict == "suspicious"
    assert "No URLs" in result.reasoning


@pytest.mark.asyncio
async def test_content_analyst(mock_llm):
    """Test Content Analyst agent."""
    agent = ContentAnalyst(llm=mock_llm)
    result = await agent.analyze(SAMPLE_EMAIL)

    assert isinstance(result, AgentOutput)
    assert result.agent_name == "content_analyst"
    assert result.verdict in ["phishing", "legitimate", "suspicious"]


@pytest.mark.asyncio
async def test_header_analyst(mock_llm):
    """Test Header Analyst agent."""
    agent = HeaderAnalyst(llm=mock_llm)
    result = await agent.analyze(SAMPLE_EMAIL)

    assert isinstance(result, AgentOutput)
    assert result.agent_name == "header_analyst"
    assert result.verdict in ["phishing", "legitimate", "suspicious"]


@pytest.mark.asyncio
async def test_visual_analyst(mock_llm):
    """Test Visual Analyst agent."""
    agent = VisualAnalyst(llm=mock_llm)
    result = await agent.analyze(SAMPLE_EMAIL)

    assert isinstance(result, AgentOutput)
    assert result.agent_name == "visual_analyst"
    assert result.verdict in ["phishing", "legitimate", "suspicious"]


@pytest.mark.asyncio
async def test_url_heuristic_analysis():
    """Test URL heuristic analysis."""
    agent = URLAnalyst(llm=MockLLM())

    # Test IP address detection
    urls = ["http://192.168.1.1/login"]
    result = agent._heuristic_analysis(urls)
    assert result["suspicious_count"] > 0
    assert any("IP address" in e for e in result["evidence"])

    # Test suspicious TLD detection
    urls = ["http://example.tk"]
    result = agent._heuristic_analysis(urls)
    assert result["suspicious_count"] > 0


@pytest.mark.asyncio
async def test_content_heuristic_analysis():
    """Test content heuristic analysis."""
    agent = ContentAnalyst(llm=MockLLM())

    result = agent._heuristic_analysis(SAMPLE_EMAIL)

    assert result["urgency_count"] > 0 or result["credential_count"] > 0
    assert len(result["evidence"]) > 0


def test_levenshtein_distance():
    """Test Levenshtein distance calculation."""
    agent = URLAnalyst(llm=MockLLM())

    # Exact match
    assert agent._levenshtein_distance("test", "test") == 0

    # One character different
    assert agent._levenshtein_distance("test", "tast") == 1

    # Completely different
    assert agent._levenshtein_distance("abc", "xyz") == 3


def test_typosquat_detection():
    """Test typosquatting detection."""
    agent = URLAnalyst(llm=MockLLM())

    # Detect typosquat
    assert agent._is_typosquat("g00gle.com", "google.com")
    assert agent._is_typosquat("faceb00k.com", "facebook.com")

    # Not typosquat
    assert not agent._is_typosquat("google.com", "google.com")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
