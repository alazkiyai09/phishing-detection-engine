from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_and_explain_roundtrip() -> None:
    analyze = client.post(
        "/api/v1/analyze/email",
        json={
            "subject": "Urgent verify your account",
            "sender": "alerts@example.com",
            "body": "Verify your password immediately at https://bad.example/login",
            "headers": {"Reply-To": "different@example.net"},
        },
    )
    assert analyze.status_code == 200
    payload = analyze.json()
    assert "prediction_id" in payload
    explain = client.post(f"/api/v1/explain/{payload['prediction_id']}")
    assert explain.status_code == 200
    assert "Prediction:" in explain.json()["explanation"]
