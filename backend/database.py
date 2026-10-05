"""Database wiring for Project Pythagoras.

Uses SQLAlchemy 2.x when it is installed and falls back to a lightweight
in-memory placeholder otherwise, so the API still boots in minimal
environments (e.g. CI without the full dependency set).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from backend.config import settings

try:  # SQLAlchemy is an optional-at-runtime dependency.
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    _HAVE_SQLALCHEMY = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_SQLALCHEMY = False


class DatabaseSessionManager:
    """Owns the engine and hands out sessions via a context manager."""

    def __init__(self, url: str | None = None) -> None:
        self.url = url or settings.postgres_url
        if _HAVE_SQLALCHEMY:
            self._engine = create_engine(self.url, pool_pre_ping=True)
            self._factory = sessionmaker(bind=self._engine, autoflush=False)
        else:
            self._engine = None
            self._factory = None
        self._fallback_state: dict[str, Any] = {}

    @contextmanager
    def session(self) -> Iterator[Any]:
        """Yield a transactional session (or the in-memory fallback)."""
        if self._factory is None:
            yield self._fallback_state
            return
        db_session = self._factory()
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    def get_session(self) -> dict[str, Any]:
        """Legacy placeholder API kept for backwards compatibility."""
        return self._fallback_state


session_manager = DatabaseSessionManager()
