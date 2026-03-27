"""
Unit tests for Content Analyst agent.
"""
import pytest
from datetime import datetime

from src.agents.content_analyst import ContentAnalyst
from src.models.email import EmailData, EmailHeaders
from src.models.agent_output import AgentOutput
from src.llm.mock_backend import MockLLM


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    return MockLLM(model_name="mock-model")


@pytest.fixture
def content_analyst(mock_llm):
    """Create Content Analyst instance for testing."""
    return ContentAnalyst(llm=mock_llm, temperature=0.0)


@pytest.fixture
def sample_email_phishing():
    """Create a sample phishing email with urgency and credential requests."""
    return EmailData(
        headers=EmailHeaders(
            subject="URGENT: Immediate Action Required",
            from_address="support@secure-verify.com",
            to_addresses=["user@example.com"],
            date=datetime.now()
        ),
        body="""
        Dear Customer,

        Your account will be suspended within 24 hours unless you verify your password.
        Click here immediately to confirm your account details.

        Urgent notice: Please provide your account information to avoid service interruption.

        ACT NOW!!!
        """,
        email_id="test-001"
    )


@pytest.fixture
def sample_email_legitimate():
    """Create a sample legitimate email."""
    return EmailData(
        headers=EmailHeaders(
            subject="Your Monthly Statement",
            from_address="notifications@chase.com",
            to_addresses=["john.doe@example.com"],
            date=datetime.now()
        ),
        body="""
        Dear John Doe,

        Your monthly statement is now available. Please log in to your account to view it.

        If you have questions, please contact our customer service team.

        Best regards,
        Chase Bank
        """,
        email_id="test-002"
    )


@pytest.mark.asyncio
async def test_content_analyst_phishing_detection(content_analyst, sample_email_phishing):
    """Test content analyst detects social engineering tactics."""
    result = await content_analyst.analyze(sample_email_phishing)

    assert isinstance(result, AgentOutput)
    assert result.agent_name == "content_analyst"
    assert result.is_phishing == True
    assert result.confidence > 0.5
    assert len(result.evidence) > 0
    assert result.error is None


@pytest.mark.asyncio
async def test_content_analyst_legitimate(content_analyst, sample_email_legitimate):
    """Test content analyst correctly identifies legitimate content."""
    result = await content_analyst.analyze(sample_email_legitimate)

    assert isinstance(result, AgentOutput)
    assert result.agent_name == "content_analyst"
    assert result.confidence > 0.0


@pytest.mark.asyncio
async def test_urgency_detection(content_analyst):
    """Test urgency language detection."""
    urgent_text = "Immediate action required! Your account will be suspended."
    urgency_score = content_analyst._detect_urgency(urgent_text)
    assert urgency_score >= 0.3

    normal_text = "Here is your monthly statement. No action required."
    urgency_score = content_analyst._detect_urgency(normal_text)
    assert urgency_score < 0.3


@pytest.mark.asyncio
async def test_credential_harvesting_detection(content_analyst):
    """Test credential harvesting detection."""
    harvesting_text = "Please enter your password and verify your account immediately."
    harvesting_score = content_analyst._detect_credential_harvesting(harvesting_text)
    assert harvesting_score > 0.3

    safe_text = "Your statement is ready for viewing."
    harvesting_score = content_analyst._detect_credential_harvesting(safe_text)
    assert harvesting_score < 0.3


def test_heuristic_analysis_urgency(content_analyst, sample_email_phishing):
    """Test heuristic analysis detects urgency patterns."""
    result = content_analyst._heuristic_analysis(sample_email_phishing)

    assert result["urgency_count"] > 0
    assert len(result["evidence"]) > 0
    assert any("urgency" in ev.lower() for ev in result["evidence"])


def test_heuristic_analysis_credentials(content_analyst, sample_email_phishing):
    """Test heuristic analysis detects credential requests."""
    result = content_analyst._heuristic_analysis(sample_email_phishing)

    assert result["credential_count"] > 0


def test_heuristic_analysis_generic_greeting(content_analyst, sample_email_phishing):
    """Test heuristic analysis detects generic greetings."""
    result = content_analyst._heuristic_analysis(sample_email_phishing)

    assert result["has_generic_greeting"] == True


def test_heuristic_analysis_legitimate(content_analyst, sample_email_legitimate):
    """Test heuristic analysis on legitimate email."""
    result = content_analyst._heuristic_analysis(sample_email_legitimate)

    assert result["urgency_count"] == 0
    assert result["credential_count"] == 0
    assert result["has_generic_greeting"] == False


@pytest.mark.asyncio
async def test_error_handling(content_analyst, sample_email_phishing, monkeypatch):
    """Test content analyst handles LLM failures gracefully."""
    async def mock_generate_fail(*args, **kwargs):
        raise Exception("LLM API error")

    monkeypatch.setattr(content_analyst.llm, 'generate', mock_generate_fail)

    result = await content_analyst.analyze(sample_email_phishing)

    assert isinstance(result, AgentOutput)
    assert result.fallback_used == True
    assert result.error is not None
