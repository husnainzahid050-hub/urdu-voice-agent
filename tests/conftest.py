"""Shared test setup."""

from __future__ import annotations

import pytest

from src import server
from src.limits import RateLimiter


@pytest.fixture(autouse=True)
def _fresh_rate_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with unlimited, empty rate-limit counters."""
    monkeypatch.setattr(server, "limiter", RateLimiter(0, 0, 0))
