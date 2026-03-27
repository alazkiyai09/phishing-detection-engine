"""
Test URL analyzer service.
"""
import pytest
from app.services.url_analyzer import url_analyzer


@pytest.mark.asyncio
async def test_analyze_phishing_url():
    """Test analysis of phishing URL."""
    result = await url_analyzer.analyze_url(
        "http://chase-secure-portal.xyz/login",
        context={"sender": "security@chase-secure-portal.xyz"}
    )

    assert "verdict" in result
    assert "risk_score" in result
    assert result["risk_score"] >= 50  # Should be high risk
    assert result["checks"]["suspicious_tld"] == True
    assert result["checks"]["bank_impersonation"] is not None


@pytest.mark.asyncio
async def test_analyze_legitimate_url():
    """Test analysis of legitimate URL."""
    result = await url_analyzer.analyze_url("https://chase.com")

    assert "verdict" in result
    assert result["risk_score"] < 30  # Should be low risk
    assert result["checks"]["suspicious_tld"] == False


@pytest.mark.asyncio
async def test_analyze_ip_address_url():
    """Test analysis of URL with IP address."""
    result = await url_analyzer.analyze_url("http://192.168.1.1/login")

    assert result["checks"]["has_ip_address"] == True
    assert result["risk_score"] >= 30


@pytest.mark.asyncio
async def test_analyze_url_with_port():
    """Test analysis of URL with port number."""
    result = await url_analyzer.analyze_url("http://example.com:8080/login")

    assert result["checks"]["has_port"] == True


@pytest.mark.asyncio
async def test_analyze_url_shortener():
    """Test analysis of URL shortener."""
    result = await url_analyzer.analyze_url("https://bit.ly/3abc123")

    assert result["checks"]["url_shortener"] == True
