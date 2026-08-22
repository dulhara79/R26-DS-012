from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from bs4 import BeautifulSoup

from ..models import RawDocument, Section, SourceConfig, SourceState
from ..util import normalize_whitespace, parse_datetime, utc_now
from .base import FetchResult


class LocalFilesConnector:
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def fetch(
        self,
        source: SourceConfig,
        state: SourceState,
        since: datetime | None,
        until: datetime,
    ) -> FetchResult:
        del state, until
        configured = Path(str(source.settings.get("path", "data/local")))
        base_path = configured if configured.is_absolute() else self.project_root / configured
        patterns = _string_list(
            source.settings.get(
                "patterns", ["**/*.md", "**/*.txt", "**/*.html", "**/*.json"]
            ),
            "patterns",
        )
        files: list[Path] = []
        for pattern in patterns:
            files.extend(path for path in base_path.glob(str(pattern)) if path.is_file())
        documents: list[RawDocument] = []
        for path in sorted(set(files)):
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            if since and modified_at < since:
                continue
            document = _read_local_document(source.id, base_path, path, modified_at)
            if document is not None:
                documents.append(document)
        return FetchResult(documents=documents, changed=bool(documents))


def _read_local_document(
    source_id: str,
    base_path: Path,
    path: Path,
    modified_at: datetime,
) -> RawDocument | None:
    raw = path.read_text(encoding="utf-8")
    metadata: dict[str, Any] = {}
    body = raw
    if path.suffix.lower() == ".md" and raw.startswith("---\n"):
        closing = raw.find("\n---\n", 4)
        if closing != -1:
            metadata = _mapping(yaml.safe_load(raw[4:closing]), f"front matter in {path}")
            body = raw[closing + 5 :]
    elif path.suffix.lower() == ".json":
        payload = json.loads(raw)
        if isinstance(payload, dict):
            metadata = _mapping(payload.get("metadata"), f"metadata in {path}")
            body = str(payload.get("text", ""))
            metadata.setdefault("title", payload.get("title"))
        else:
            return None
    elif path.suffix.lower() == ".html":
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup.select("script,style,noscript,nav,footer"):
            tag.decompose()
        body = soup.get_text("\n", strip=True)
        if soup.title:
            metadata.setdefault("title", soup.title.get_text(" ", strip=True))

    body = normalize_whitespace(body)
    if not body:
        return None
    relative = path.relative_to(base_path).as_posix()
    title = normalize_whitespace(str(metadata.get("title") or path.stem.replace("_", " ")))
    published_at = parse_datetime(metadata.get("published_at"))
    updated_at = parse_datetime(metadata.get("updated_at")) or modified_at
    topics = _string_list(metadata.get("topics"), "topics")
    publication_types = _string_list(metadata.get("publication_types"), "publication_types")
    language = str(metadata.get("language", "en"))
    url = metadata.get("url")
    section = Section(path="root", heading=title, text=body, ordinal=0)
    return RawDocument(
        source_id=source_id,
        external_id=str(metadata.get("external_id") or relative),
        title=title,
        text=body,
        url=None if url is None else str(url),
        published_at=published_at,
        updated_at=updated_at,
        retrieved_at=utc_now(),
        authors=_string_list(metadata.get("authors"), "authors"),
        language=language,
        publication_types=publication_types,
        topics=topics,
        sections=[section],
        metadata={**metadata, "local_path": relative},
    )


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping/object")
    return dict(value)


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"{field} must be a string or a list of strings")
