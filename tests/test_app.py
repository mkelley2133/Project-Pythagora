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
    assert payload["telemetry_json"]["duration"] > 0
    assert len(payload["telemetry_json"]["segments"]) == 5
    assert len(payload["telemetry_json"]["timestamped_lyrics"]) == 14


def test_track_analysis_404_for_unknown_track() -> None:
    response = client.get("/tracks/does-not-exist/analysis")
    assert response.status_code == 404


def test_create_track_returns_201_and_is_fetchable() -> None:
    response = client.post(
        "/tracks",
        json={
            "title": "Test Signal",
            "artist": "Pythagoras",
            "original_file_path": "/storage/test.mp3",
        },
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["title"] == "Test Signal"
    assert payload["id"].startswith("track-")
    assert payload["created_at"] is not None

    fetched = client.get(f"/tracks/{payload['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == payload["id"]


def test_create_track_rejects_blank_title() -> None:
    response = client.post(
        "/tracks",
        json={"title": "", "artist": "A", "original_file_path": "/x.mp3"},
    )
    assert response.status_code == 422
