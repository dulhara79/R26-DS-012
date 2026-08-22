from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from datetime import UTC, date, datetime
from collections.abc import Mapping, Sequence
from typing import Any, Iterable

import numpy as np


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

# A compact retrieval stop-word list. It is intentionally conservative: domain words are
# never removed, while function words that create spurious lexical/heuristic matches are.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "ours", "she", "should", "that",
    "the", "their", "theirs", "them", "then", "there", "these", "they", "this",
    "those", "to", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your", "yours",
}


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_datetime(value: str | datetime | date | None) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m",
        "%Y",
        "%d %B %Y",
        "%B %d, %Y",
        "%Y %b %d",
    ]
    try:
        return ensure_utc(datetime.fromisoformat(text))
    except ValueError:
        pass
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


_SENSITIVE_SETTING_PARTS = {
    "authorization",
    "credential",
    "credentials",
    "key",
    "password",
    "secret",
    "token",
}


def redact_sensitive_settings(value: Any) -> Any:
    """Return a JSON-safe shape with credential-like settings replaced recursively."""
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            parts = {part for part in re.split(r"[^a-z0-9]+", key.lower()) if part}
            if parts & _SENSITIVE_SETTING_PARTS:
                output[key] = "[REDACTED]"
            else:
                output[key] = redact_sensitive_settings(child)
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_sensitive_settings(child) for child in value]
    return value


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_id(*parts: str, length: int = 40) -> str:
    return sha256_text("\x1f".join(parts))[:length]


def normalize_whitespace(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def content_tokens(text: str) -> list[str]:
    """Return normalized information-bearing tokens for retrieval heuristics."""
    return [token for token in tokens(text) if len(token) >= 2 and token not in _STOPWORDS]


def fts_query(text: str, max_terms: int = 24) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for token in content_tokens(text):
        if token in seen:
            continue
        seen.add(token)
        terms.append(token.replace('"', '""'))
        if len(terms) >= max_terms:
            break
    return " OR ".join(f'"{term}"' for term in terms)


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    va = np.asarray(list(a), dtype=np.float32)
    vb = np.asarray(list(b), dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def normalize_vector(vector: Iterable[float]) -> list[float]:
    array = np.asarray(list(vector), dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array.tolist()
    return (array / norm).astype(np.float32).tolist()


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(maximum, max(minimum, value))


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("Batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def age_days(value: datetime | None, now: datetime | None = None) -> float:
    if value is None:
        return float("inf")
    now = ensure_utc(now or utc_now())
    value = ensure_utc(value)
    if now is None or value is None:
        raise RuntimeError("UTC normalization unexpectedly returned None")
    return max(0.0, (now - value).total_seconds() / 86400.0)


def freshness_score(value: datetime | None, half_life_days: int, floor: float = 0.15) -> float:
    if value is None:
        return floor
    days = age_days(value)
    return clamp(max(floor, math.exp(-math.log(2.0) * days / float(half_life_days))))


def xml_text(element: Any) -> str:
    if element is None:
        return ""
    return normalize_whitespace("".join(element.itertext()))
