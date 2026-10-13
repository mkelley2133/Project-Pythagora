"""Live-mode reliability pins (Codespaces fix).

The dashboard must never silently fall back to demo data when the backend
served the page. The mechanism: app.main injects a
<meta name="pythagoras-served-by" content="backend"> marker into the root
document at serve time. The raw file on disk must NOT contain the marker,
so a static host (GitHub Pages) can never fake live mode.
"""
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_MARKER = 'name="pythagoras-served-by" content="backend"'
_FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


def test_root_document_carries_live_marker() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert _MARKER in response.text


def test_index_html_route_carries_live_marker() -> None:
    response = client.get("/index.html")
    assert response.status_code == 200
    assert _MARKER in response.text


def test_raw_index_file_has_no_marker() -> None:
    """The file on disk must stay marker-free — injection is serve-time only."""
    assert _MARKER not in _FRONTEND_INDEX.read_text(encoding="utf-8")


def test_root_document_is_no_cache() -> None:
    response = client.get("/")
    assert response.headers.get("cache-control") == "no-cache"


def test_config_endpoint_shape() -> None:
    response = client.get("/config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["served_by"] == "backend"
    assert payload["app"]
    assert payload["version"]
    assert isinstance(payload["api_key_required"], bool)


def test_health_answers_so_demo_must_not_engage() -> None:
    """Frontend rule (main.js checkApi): a same-origin /health 200 on a
    backend-served page means LIVE. Pin the backend half of that contract."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dashboard_assets_still_served() -> None:
    """The marker route must not shadow the static assets."""
    response = client.get("/src/main.js")
    assert response.status_code == 200
    assert "pythagoras-served-by" in response.text  # the mode logic itself
    assert "EXPLICIT_DEMO" in response.text
    css = client.get("/styles.css")
    assert css.status_code == 200
