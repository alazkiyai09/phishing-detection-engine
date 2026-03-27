"""
Pytest configuration and fixtures.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings


@pytest.fixture
def client():
    """
    Create a test client for the API.
    """
    return TestClient(app)


@pytest.fixture
def sample_phishing_email():
    """
    Sample phishing email for testing.
    """
    return {
        "raw_email": """From: security@chase-secure-portal.xyz
Subject: URGENT: Verify your account now

Your account will be suspended within 24 hours unless you verify your information.
Click here: http://chase-secure-portal.xyz/login
"""
    }


@pytest.fixture
def sample_legitimate_email():
    """
    Sample legitimate email for testing.
    """
    return {
        "raw_email": """From: notifications@chase.com
Subject: Your statement is available

Your monthly statement is now available. Log in to view it.
"""
    }


@pytest.fixture
def sample_phishing_url():
    """
    Sample phishing URL for testing.
    """
    return {
        "url": "http://chase-secure-portal.xyz/login",
        "context": {
            "sender": "security@chase-secure-portal.xyz",
            "subject": "Account Verification Required"
        }
    }
