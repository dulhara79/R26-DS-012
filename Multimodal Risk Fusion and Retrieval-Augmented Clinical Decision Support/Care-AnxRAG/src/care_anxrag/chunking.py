from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from .models import (
    ChunkRecord,
    DocumentStatus,
    DocumentVersion,
    Section,
    SourceConfig,
)
from .util import normalize_whitespace, sha256_text, stable_id


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass(slots=True)
class ChunkingConfig:
    max_words: int = 180
    overlap_words: int = 35
    min_words: int = 35

    def validate(self) -> None:
        if self.max_words <= 0 or self.overlap_words < 0 or self.min_words <= 0:
            raise ValueError("Chunk sizes must be positive")
        if self.overlap_words >= self.max_words:
            raise ValueError("overlap_words must be less than max_words")


class SectionAwareChunker:
    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
        self.config.validate()

    def sections(self, title: str, text: str, supplied: Sequence[Section] | None = None) -> list[Section]:
        if supplied:
            result: list[Section] = []
            for index, section in enumerate(supplied):
                cleaned = normalize_whitespace(section.text)
                if not cleaned:
                    continue
                result.append(
                    Section(
                        path=section.path or f"section-{index + 1}",
                        heading=section.heading or title,
                        text=cleaned,
                        ordinal=index,
                        content_hash=sha256_text(cleaned),
                    )
                )
            if result:
                return result

        lines = text.splitlines()
        sections: list[Section] = []
        heading_stack: list[tuple[int, str]] = []
        current_heading = title
        current_path = "root"
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer
            cleaned = normalize_whitespace("\n".join(buffer))
            if cleaned:
                sections.append(
                    Section(
                        path=current_path,
                        heading=current_heading,
                        text=cleaned,
                        ordinal=len(sections),
                        content_hash=sha256_text(cleaned),
                    )
                )
            buffer = []

        for line in lines:
            match = _HEADING_RE.match(line.strip())
            if match:
                flush()
                level = len(match.group(1))
                heading = normalize_whitespace(match.group(2))
                heading_stack = [(lvl, value) for lvl, value in heading_stack if lvl < level]
                heading_stack.append((level, heading))
                current_heading = heading
                current_path = " / ".join(value for _, value in heading_stack)
            else:
                buffer.append(line)
        flush()

        if not sections:
            cleaned = normalize_whitespace(text)
            sections = [
                Section(
                    path="root",
                    heading=title,
                    text=cleaned,
                    ordinal=0,
                    content_hash=sha256_text(cleaned),
                )
            ]
        return sections

    def chunk_text(self, text: str) -> list[str]:
        text = normalize_whitespace(text)
        if not text:
            return []
        sentences = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
        if not sentences:
            sentences = [text]

        chunks: list[str] = []
        current: list[str] = []
        current_words = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            if len(sentence_words) > self.config.max_words:
                if current:
                    chunks.append(" ".join(current).strip())
                    current = []
                    current_words = 0
                step = self.config.max_words - self.config.overlap_words
                for start in range(0, len(sentence_words), step):
                    piece = sentence_words[start : start + self.config.max_words]
                    if piece:
                        chunks.append(" ".join(piece))
                    if start + self.config.max_words >= len(sentence_words):
                        break
                continue

            if current and current_words + len(sentence_words) > self.config.max_words:
                chunks.append(" ".join(current).strip())
                overlap: list[str] = []
                overlap_count = 0
                for prior in reversed(current):
                    prior_words = prior.split()
                    if overlap_count + len(prior_words) > self.config.overlap_words:
                        break
                    overlap.insert(0, prior)
                    overlap_count += len(prior_words)
                current = overlap
                current_words = overlap_count

            current.append(sentence)
            current_words += len(sentence_words)

        if current:
            chunks.append(" ".join(current).strip())

        if len(chunks) > 1 and len(chunks[-1].split()) < self.config.min_words:
            chunks[-2] = normalize_whitespace(f"{chunks[-2]} {chunks[-1]}")
            chunks.pop()
        return chunks

    def build_chunks(
        self,
        version: DocumentVersion,
        source: SourceConfig,
        sections: Sequence[Section],
        previous_sections: Sequence[Section] | None = None,
        previous_chunks: Sequence[ChunkRecord] | None = None,
    ) -> list[ChunkRecord]:
        prior_section_hashes = {
            section.path: section.content_hash for section in (previous_sections or [])
        }
        prior_chunks_by_section: dict[str, list[ChunkRecord]] = defaultdict(list)
        for chunk in previous_chunks or []:
            prior_chunks_by_section[chunk.section_path].append(chunk)
        for values in prior_chunks_by_section.values():
            values.sort(key=lambda chunk: chunk.ordinal)

        output: list[ChunkRecord] = []
        for section in sections:
            if (
                prior_section_hashes.get(section.path) == section.content_hash
                and prior_chunks_by_section.get(section.path)
            ):
                text_chunks = [chunk.text for chunk in prior_chunks_by_section[section.path]]
            else:
                text_chunks = self.chunk_text(section.text)

            for ordinal, text in enumerate(text_chunks):
                text_hash = sha256_text(text)
                chunk_id = stable_id(
                    version.version_id,
                    section.path,
                    str(ordinal),
                    text_hash,
                )
                output.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        document_id=version.document_id,
                        version_id=version.version_id,
                        source_id=source.id,
                        source_name=source.name,
                        title=version.title,
                        url=version.url,
                        layer=version.layer,
                        status=DocumentStatus.STAGING,
                        section_path=section.path,
                        section_heading=section.heading,
                        ordinal=ordinal,
                        text=text,
                        text_hash=text_hash,
                        published_at=version.published_at,
                        updated_at=version.updated_at,
                        retrieved_at=version.retrieved_at,
                        authority_score=version.authority_score,
                        evidence_level=version.evidence_level,
                        evidence_score=version.evidence_score,
                        topics=version.topics,
                        metadata={
                            "section_hash": section.content_hash,
                            "external_id": version.external_id,
                            "content_hash": version.content_hash,
                        },
                    )
                )
        return output
