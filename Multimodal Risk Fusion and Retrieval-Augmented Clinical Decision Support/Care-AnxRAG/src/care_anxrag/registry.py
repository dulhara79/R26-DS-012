from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from .models import SourceConfig


_SOURCE_LIST = TypeAdapter(list[SourceConfig])


def load_source_registry(path: Path | str) -> list[SourceConfig]:
    registry_path = Path(path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Source registry not found: {registry_path}")
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if payload is None:
        raw_sources = []
    elif isinstance(payload, dict):
        if "sources" not in payload:
            raise ValueError("Source registry mapping must contain a 'sources' key")
        raw_sources = payload["sources"]
    elif isinstance(payload, list):
        raw_sources = payload
    else:
        raise ValueError("Source registry root must be a mapping or a list")
    sources = _SOURCE_LIST.validate_python(raw_sources)
    for source in sources:
        _reject_inline_credentials(source)
    ids = [source.id for source in sources]
    duplicates = sorted({source_id for source_id in ids if ids.count(source_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate source IDs: {', '.join(duplicates)}")
    return sources


def source_by_id(sources: list[SourceConfig]) -> dict[str, SourceConfig]:
    return {source.id: source for source in sources}


def _reject_inline_credentials(source: SourceConfig) -> None:
    forbidden_parts = {"authorization", "credential", "credentials", "key", "password", "secret", "token"}

    def walk(value: object, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                normalized = key.lower()
                parts = {part for part in re.split(r"[^a-z0-9]+", normalized) if part}
                is_env_reference = normalized.endswith("_env")
                if (
                    not is_env_reference
                    and parts & forbidden_parts
                    and child not in (None, "", [], {})
                ):
                    location = ".".join((*path, key))
                    raise ValueError(
                        f"Source {source.id!r} contains inline credential setting {location!r}; "
                        "store credentials in an environment variable and reference it with an *_env setting"
                    )
                walk(child, (*path, key))
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                walk(child, (*path, str(index)))

    walk(source.settings, ("settings",))
