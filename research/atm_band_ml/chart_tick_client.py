"""Fetch session ticks (incl. cumulative day volume) from the local Angel chart server."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from config.urls import CHART_SERVER_BASE_URL

DEFAULT_CHART_TICK_TTL_SEC = 30.0


@dataclass
class ChartTickClient:
    """HTTP client for ``GET /replay/{token}/ticks`` (today when ``date`` omitted)."""

    base_url: str = CHART_SERVER_BASE_URL
    timeout_sec: float = 4.0
    cache_ttl_sec: float = DEFAULT_CHART_TICK_TTL_SEC
    _cache: dict[str, tuple[float, dict[str, Any]]] = field(default_factory=dict, repr=False)
    _last_error: str | None = field(default=None, repr=False)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def is_available(self) -> bool:
        url = f"{self.base_url.rstrip('/')}/storage/stats"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=min(2.0, self.timeout_sec)) as resp:
                return int(getattr(resp, "status", 200) or 200) < 400
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            return False

    def fetch_ticks(self, token: str, *, date: str | None = None, bust_cache: bool = False) -> dict[str, Any] | None:
        tok = str(token or "").strip()
        if not tok:
            return None
        cache_key = f"{tok}|{date or ''}"
        if not bust_cache:
            cached = self._cache.get(cache_key)
            if cached is not None and (time.time() - cached[0]) < self.cache_ttl_sec:
                return cached[1]

        params = ""
        if date:
            params = "?" + urllib.parse.urlencode({"date": str(date).strip()})
        url = f"{self.base_url.rstrip('/')}/replay/{urllib.parse.quote(tok, safe='')}/ticks{params}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read()
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                self._last_error = "invalid JSON payload"
                return None
            self._last_error = None
            self._cache[cache_key] = (time.time(), payload)
            return payload
        except urllib.error.HTTPError as exc:
            self._last_error = f"HTTP {exc.code}"
            return None
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            self._last_error = str(exc)
            return None

    def clear_cache(self) -> None:
        self._cache.clear()
