from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .clinical_match import (
    extract_subtype_concepts,
    extract_treatment_concepts,
)
from .db import Database
from .models import DocumentStatus


@dataclass(frozen=True, slots=True)
class CoverageCombination:
    subtype: str
    treatment: str
    supporting_chunks: int
    supporting_documents: int
    supporting_sources: int
    evidence_levels: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class CorpusCoverageReport:
    active_chunks: int
    active_documents: int
    subtype_counts: dict[str, int]
    treatment_counts: dict[str, int]
    combinations: tuple[CoverageCombination, ...]
    requested_subtype: str | None = None
    requested_treatment: str | None = None
    requested_supporting_chunks: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "active_chunks": self.active_chunks,
            "active_documents": self.active_documents,
            "subtype_counts": dict(self.subtype_counts),
            "treatment_counts": dict(self.treatment_counts),
            "combinations": [asdict(item) for item in self.combinations],
            "requested_subtype": self.requested_subtype,
            "requested_treatment": self.requested_treatment,
            "requested_supporting_chunks": self.requested_supporting_chunks,
        }


def _clinical_text(chunk: Any) -> str:
    return "\n".join(
        value
        for value in (
            getattr(chunk, "title", ""),
            getattr(chunk, "section_heading", ""),
            getattr(chunk, "text", ""),
        )
        if value
    )


def _subtypes_for_chunk(chunk: Any) -> set[str]:
    evidence_text = _clinical_text(chunk)
    topic_text = " ".join(
        str(topic).replace("_", " ")
        for topic in getattr(chunk, "topics", [])
    )
    return (
        extract_subtype_concepts(evidence_text)
        | extract_subtype_concepts(topic_text)
    )


def audit_corpus_coverage(
    database: Database,
    *,
    subtype: str | None = None,
    treatment: str | None = None,
) -> CorpusCoverageReport:
    """Summarize direct clinical coverage in the active evidence corpus.

    A subtype/treatment combination is counted only when the same active chunk
    supports both concepts. Treatment support is derived from source text/title,
    never from topic metadata alone.
    """
    chunks = database.list_chunks(status=DocumentStatus.ACTIVE)

    subtype_counts: dict[str, int] = {}
    treatment_counts: dict[str, int] = {}
    combination_chunks: dict[tuple[str, str], set[str]] = {}
    combination_documents: dict[tuple[str, str], set[str]] = {}
    combination_sources: dict[tuple[str, str], set[str]] = {}
    combination_levels: dict[tuple[str, str], set[str]] = {}

    for chunk in chunks:
        subtypes = _subtypes_for_chunk(chunk)
        treatments = extract_treatment_concepts(_clinical_text(chunk))

        for item in subtypes:
            subtype_counts[item] = subtype_counts.get(item, 0) + 1
        for item in treatments:
            treatment_counts[item] = treatment_counts.get(item, 0) + 1

        for current_subtype in subtypes:
            for current_treatment in treatments:
                key = (current_subtype, current_treatment)
                combination_chunks.setdefault(key, set()).add(chunk.chunk_id)
                combination_documents.setdefault(key, set()).add(chunk.document_id)
                combination_sources.setdefault(key, set()).add(chunk.source_id)
                combination_levels.setdefault(key, set()).add(
                    chunk.evidence_level.value
                )

    combinations = tuple(
        CoverageCombination(
            subtype=current_subtype,
            treatment=current_treatment,
            supporting_chunks=len(combination_chunks[key]),
            supporting_documents=len(combination_documents[key]),
            supporting_sources=len(combination_sources[key]),
            evidence_levels=tuple(sorted(combination_levels[key])),
        )
        for key in sorted(combination_chunks)
        for current_subtype, current_treatment in [key]
        if (subtype is None or current_subtype == subtype)
        and (treatment is None or current_treatment == treatment)
    )

    requested_supporting_chunks: int | None = None
    if subtype is not None and treatment is not None:
        requested_supporting_chunks = len(
            combination_chunks.get((subtype, treatment), set())
        )

    return CorpusCoverageReport(
        active_chunks=len(chunks),
        active_documents=len({chunk.document_id for chunk in chunks}),
        subtype_counts=dict(sorted(subtype_counts.items())),
        treatment_counts=dict(sorted(treatment_counts.items())),
        combinations=combinations,
        requested_subtype=subtype,
        requested_treatment=treatment,
        requested_supporting_chunks=requested_supporting_chunks,
    )
