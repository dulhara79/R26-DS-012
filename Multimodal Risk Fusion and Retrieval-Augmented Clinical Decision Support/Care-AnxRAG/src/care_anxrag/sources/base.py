from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import httpx

from ..models import RawDocument, SourceConfig, SourceState


@dataclass(slots=True)
class FetchResult:
    documents: list[RawDocument] = field(default_factory=list)
    etag: str | None = None
    last_modified: str | None = None
    cursor: str | None = None
    changed: bool = False
    warnings: list[str] = field(default_factory=list)


class SourceConnector(Protocol):
    def fetch(
        self,
        source: SourceConfig,
        state: SourceState,
        since: datetime | None,
        until: datetime,
    ) -> FetchResult: ...


class ResilientHttpClient:
    def __init__(
        self,
        user_agent: str,
        timeout_seconds: float = 60.0,
        minimum_interval_seconds: float = 0.0,
        max_attempts: int = 3,
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if minimum_interval_seconds < 0:
            raise ValueError("minimum_interval_seconds cannot be negative")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.client = httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
        )
        self.minimum_interval_seconds = minimum_interval_seconds
        self.max_attempts = max_attempts
        self._last_request_monotonic = 0.0

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "ResilientHttpClient":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def request(
        self,
        method: str,
        url: str,
        *,
        acceptable_statuses: set[int] | None = None,
        **kwargs: object,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            elapsed = time.monotonic() - self._last_request_monotonic
            if elapsed < self.minimum_interval_seconds:
                time.sleep(self.minimum_interval_seconds - elapsed)
            try:
                response = self.client.request(method, url, **kwargs)
                self._last_request_monotonic = time.monotonic()
                if acceptable_statuses and response.status_code in acceptable_statuses:
                    return response
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_attempts:
                    retry_after = response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt - 1)
                    time.sleep(min(delay, 10.0))
                    continue
                response.raise_for_status()
                return response
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                time.sleep(min(2 ** (attempt - 1), 10.0))
        if last_error is None:
            raise RuntimeError("HTTP request exhausted retries without a captured error")
        raise last_error
