"""Injectable clock so hold expiry and wall-clock limits are testable."""
from __future__ import annotations

import time
from datetime import datetime, timedelta


class Clock:
    def now(self) -> datetime:  # pragma: no cover - trivial
        return datetime.now()

    def monotonic(self) -> float:  # pragma: no cover - trivial
        return time.monotonic()


class FakeClock(Clock):
    """Deterministic clock. Tests advance it explicitly."""

    def __init__(self, start: datetime):
        self._now = start
        self._mono = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, **kwargs) -> None:
        delta = timedelta(**kwargs)
        self._now += delta
        self._mono += delta.total_seconds()
