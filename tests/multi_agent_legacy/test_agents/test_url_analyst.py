"""
Unit tests for URL Analyst agent.
"""
import pytest
from datetime import datetime

from src.agents.url_analyst import URLAnalyst
from src.agents.base_agent import BaseAgent
from src.models.email import EmailData, EmailHeaders, URL
from src.models.agent_output import AgentOutput
from src.llm.mock_backend import MockLLM


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    return MockLLM(model_name="mock-model")


@pytest.fixture
def url_analyst(mock_llm):
    """Create URL Analyst instance for testing."""
    return URLAnalyst(llm=mock_llm, temperature=0.0)


@pytest.fixture
def sample_email_phishing():
    """Create a sample phishing email with suspicious URLs."""
    return EmailData(
        headers=EmailHeaders(
            subject="Urgent: Verify Your Account",
            from_address="support@secure-verify.com",
            to_addresses=["user@example.com"],
            date=datetime.now()
        ),
        body="Click here to verify your account: http://192.168.1.1/login",
        urls=[
            URL(
                original="http://192.168.1.1/login",
                domain="192.168.1.1",
                is_suspicious=True,
                suspicion_reasons=["Uses IP address instead of domain"]
            ),
            URL(
                original="http://g00gle.com/verify",
                domain="g00gle.com",
                is_suspicious=True,
                suspicion_reasons=["Possible typosquatting of google.com"]
            )
        ],
        email_id="test-001"
    )


@pytest.fixture
def sample_email_legitimate():
    """Create a sample legitimate email."""
    return EmailData(
        headers=EmailHeaders(
            subject="Your Statement is Ready",
            from_address="notifications@chase.com",
            to_addresses=["customer@example.com"],
            date=datetime.now()
        ),
        body="View your statement at https://www.chase.com/statements",
        urls=[
            URL(
                original="https://www.chase.com/statements",
                domain="chase.com",
                is_suspicious=False,
                suspicion_reasons=[]
            )
        ],
        email_id="test-002"
    )


@pytest.mark.asyncio
async def test_url_analyst_phishing_detection(url_analyst, sample_email_phishing):
    """Test URL analyst detects phishing URLs."""
    result = await url_analyst.analyze(sample_email_phishing)

    assert isinstance(result, AgentOutput)
    assert result.agent_name == "url_analyst"
    assert result.is_phishing == True
    assert result.confidence > 0.5
    assert len(result.evidence) > 0
    assert any("IP address" in ev or "typosquat" in ev.lower() for ev in result.evidence)
    assert result.error is None


@pytest.mark.asyncio
async def test_url_analyst_legitimate(url_analyst, sample_email_legitimate):
    """Test URL analyst correctly identifies legitimate URLs."""
    result = await url_analyst.analyze(sample_email_legitimate)

    assert isinstance(result, AgentOutput)
    assert result.agent_name == "url_analyst"
    assert result.confidence > 0.0
    assert len(result.reasoning) > 0


@pytest.mark.asyncio
async def test_url_analyst_no_urls(url_analyst, mock_llm):
    """Test URL analyst handles emails with no URLs."""
    email = EmailData(
        headers=EmailHeaders(
            subject="Test Email",
            from_address="test@example.com",
            to_addresses=["user@example.com"],
            date=datetime.now()
        ),
        body="This email has no URLs.",
        urls=[],
        email_id="test-003"
    )

    result = await url_analyst.analyze(email)

    assert isinstance(result, AgentOutput)
    assert result.confidence == 0.5  # Neutral confidence
    assert "no urls" in result.reasoning.lower()


@pytest.mark.asyncio
async def test_url_analyst_error_handling(url_analyst, sample_email_phishing, monkeypatch):
    """Test URL analyst handles LLM failures gracefully."""
    async def mock_generate_fail(*args, **kwargs):
        raise Exception("LLM API error")

    monkeypatch.setattr(url_analyst.llm, 'generate', mock_generate_fail)

    result = await url_analyst.analyze(sample_email_phishing)

    assert isinstance(result, AgentOutput)
    assert result.fallback_used == True
    assert result.error is not None
    assert "LLM API error" in result.error


def test_typosquat_detection(url_analyst):
    """Test typosquatting detection logic."""
    assert url_analyst._is_typosquat("g00gle.com", "google.com") == True
    assert url_analyst._is_typosquat("faceb00k.com", "facebook.com") == True
    assert url_analyst._is_typosquat("amazon.com", "amazon.com") == False
    assert url_analyst._is_typosquat("completely-different.com", "google.com") == False


def test_levenshtein_distance(url_analyst):
    """Test Levenshtein distance calculation."""
    assert url_analyst._levenshtein_distance("google.com", "g00gle.com") <= 2
    assert url_analyst._levenshtein_distance("amazon.com", "amazon.com") == 0
    assert url_analyst._levenshtein_distance("test", "test") == 0
    assert url_analyst._levenshtein_distance("kitten", "sitting") == 3


def test_url_parsing(url_analyst):
    """Test URL parsing and analysis."""
    url_str = "https://www.example.com/path?param=value"
    url_obj = url_analyst._parse_url(url_str)

    assert url_obj.original == url_str
    assert url_obj.domain == "example.com"
    assert url_obj.path == "/path"
    assert url_obj.query_params == {"param": "value"}


def test_suspicious_pattern_detection(url_analyst):
    """Test suspicious pattern detection in URLs."""
    ip_url = URL(original="http://192.168.1.1/login", domain="192.168.1.1")
    url_analyst._check_suspicious_patterns(ip_url)
    assert ip_url.is_suspicious == True
    assert "IP address" in " ".join(ip_url.suspicion_reasons)

    long_domain = "a" * 60 + ".com"
    long_url = URL(original=f"http://{long_domain}/", domain=long_domain)
    url_analyst._check_suspicious_patterns(long_url)
    assert long_url.is_suspicious == True
