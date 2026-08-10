from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_tracks_endpoint() -> None:
    response = client.get("/tracks")
    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["title"] == "Sample Signal"


def test_track_analysis_endpoint() -> None:
    response = client.get("/tracks/demo-track-001/analysis")
    assert response.status_code == 200
    payload = response.json()
    assert payload["telemetry_json"]["bpm"] == 92.4
