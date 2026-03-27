"""
Test analyze endpoints.
"""
import pytest
from fastapi.testclient import TestClient


def test_analyze_url_phishing(client):
    """Test URL analysis with phishing URL."""
    response = client.post(
        "/api/v1/analyze/url",
        json={
            "url": "http://chase-secure-portal.xyz/login",
            "context": {
                "sender": "security@chase-secure-portal.xyz",
                "subject": "Account Verification Required"
            }
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "verdict" in data
    assert "risk_score" in data
    assert "email_id" in data
    assert data["model_used"] == "url_heuristic"


def test_analyze_url_legitimate(client):
    """Test URL analysis with legitimate URL."""
    response = client.post(
        "/api/v1/analyze/url",
        json={"url": "https://chase.com"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["risk_score"] < 50  # Should be low risk


def test_analyze_url_missing(client):
    """Test URL analysis with missing URL field."""
    response = client.post(
        "/api/v1/analyze/url",
        json={}
    )

    assert response.status_code == 422  # Validation error


def test_analyze_url_empty(client):
    """Test URL analysis with empty URL."""
    response = client.post(
        "/api/v1/analyze/url",
        json={"url": ""}
    )

    assert response.status_code == 422  # Validation error


def test_analyze_email_raw_not_implemented(client, sample_phishing_email):
    """Test email analysis with raw email (should return feature preview)."""
    response = client.post(
        "/api/v1/analyze/email",
        json={
            "raw_email": sample_phishing_email["raw_email"],
            "model_type": "xgboost"
        }
    )

    # Should return 503 (feature extraction not available in tests)
    # or 200 with warning if available
    assert response.status_code in [200, 503]


def test_analyze_email_no_data(client):
    """Test email analysis without email data."""
    response = client.post(
        "/api/v1/analyze/email",
        json={}
    )

    assert response.status_code == 422  # Validation error


def test_analyze_batch_urls(client):
    """Test batch analysis with URLs."""
    response = client.post(
        "/api/v1/analyze/batch",
        json={
            "emails": [
                {"url": "http://chase-secure-portal.xyz/login"},
                {"url": "https://chase.com"}
            ],
            "parallel": True
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "batch_id" in data
    assert "results" in data
    assert "summary" in data
    assert len(data["results"]) == 2


def test_analyze_batch_too_many_emails(client):
    """Test batch analysis exceeds maximum."""
    emails = [{"url": "https://example.com"} for _ in range(101)]

    response = client.post(
        "/api/v1/analyze/batch",
        json={"emails": emails}
    )

    assert response.status_code == 422  # Validation error
