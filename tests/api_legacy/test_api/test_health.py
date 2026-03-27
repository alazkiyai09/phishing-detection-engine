"""
Test health and metrics endpoints.
"""
import pytest


def test_health_check(client):
    """
    Test health check endpoint.
    """
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "models" in data
    assert "uptime_seconds" in data


def test_metrics_endpoint(client):
    """
    Test Prometheus metrics endpoint.
    """
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")


def test_readiness_probe(client):
    """
    Test readiness probe endpoint.
    """
    response = client.get("/readiness")
    # Should return 200 if XGBoost is available, 503 otherwise
    assert response.status_code in [200, 503]


def test_liveness_probe(client):
    """
    Test liveness probe endpoint.
    """
    response = client.get("/liveness")
    assert response.status_code == 200
