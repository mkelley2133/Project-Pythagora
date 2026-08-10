from typing import Any


class DatabaseSessionManager:
    """Small placeholder session wrapper for future SQLAlchemy wiring."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    def get_session(self) -> dict[str, Any]:
        return self._state


session_manager = DatabaseSessionManager()
