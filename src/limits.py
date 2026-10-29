"""Abuse protection for a public demo: per-client and global turn limits.

A "turn" is one request that can cost Gemini quota (a spoken or typed message). Limits:
- per client: a token bucket (`per_minute` turns, refilled continuously) and a daily cap;
- all clients together: a daily cap, so strangers cannot use up the owner's free quota.
Days are UTC days. Everything is in memory; a restart resets the counters.
"""

from __future__ import annotations

import ipaddress
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

MSG_PER_MINUTE = "Too many requests: please wait a minute and try again."
MSG_PER_DAY = "You have reached today's limit for this demo. Please come back tomorrow."
MSG_GLOBAL = "This demo has reached its daily limit. Please try again tomorrow."


@dataclass
class _Bucket:
    tokens: float
    updated: float
    day: int
    today: int = 0


class RateLimiter:
    """Thread-safe turn limiter. 0 disables a limit."""

    def __init__(
        self,
        per_minute: int,
        per_day: int,
        global_per_day: int,
        clock: Callable[[], float] = time.time,
        max_clients: int = 10_000,
    ) -> None:
        self.per_minute = per_minute
        self.per_day = per_day
        self.global_per_day = global_per_day
        self._clock = clock
        self._max_clients = max_clients
        self._clients: dict[str, _Bucket] = {}
        self._global_day = -1
        self._global_today = 0
        self._lock = threading.Lock()

    def _day(self, now: float) -> int:
        return int(now // 86_400)

    def check(self, client: str) -> str | None:
        """Count one turn for `client`. Returns None if allowed, else a message to show."""
        with self._lock:
            now = self._clock()
            day = self._day(now)
            if day != self._global_day:
                self._global_day, self._global_today = day, 0
            if self.global_per_day and self._global_today >= self.global_per_day:
                return MSG_GLOBAL
            bucket = self._clients.get(client)
            if bucket is None:
                self._prune(now)
                bucket = _Bucket(tokens=float(self.per_minute), updated=now, day=day)
                self._clients[client] = bucket
            if bucket.day != day:
                bucket.day, bucket.today = day, 0
            if self.per_day and bucket.today >= self.per_day:
                return MSG_PER_DAY
            if self.per_minute:
                rate = self.per_minute / 60.0
                bucket.tokens = min(
                    float(self.per_minute), bucket.tokens + (now - bucket.updated) * rate
                )
                bucket.updated = now
                if bucket.tokens < 1.0:
                    return MSG_PER_MINUTE
                bucket.tokens -= 1.0
            bucket.today += 1
            self._global_today += 1
            return None

    def _prune(self, now: float) -> None:
        """Forget idle clients once the table is large (bounded memory)."""
        if len(self._clients) < self._max_clients:
            return
        idle = [k for k, b in self._clients.items() if now - b.updated > 3_600]
        for key in idle or list(self._clients)[: len(self._clients) // 2]:
            del self._clients[key]


def client_ip(peer: str | None, forwarded_for: str | None, trust_proxy: bool) -> str:
    """The client's IP address.

    Behind a reverse proxy (Render, Fly, nginx) the TCP peer is the proxy, and the real
    client is the last address the proxy appended to X-Forwarded-For. Earlier entries
    can be forged by the client, so they are ignored. Without TRUST_PROXY the header is
    never read (anyone could send it).
    """
    if trust_proxy and forwarded_for:
        candidate = forwarded_for.split(",")[-1].strip()
        try:
            return str(ipaddress.ip_address(candidate))
        except ValueError:
            pass
    return peer or "unknown"
