from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .models import GeneratedPayload, RelationLabel, SearchHit
from .util import content_tokens, normalize_whitespace


_CITATION_RE = re.compile(r"\[(S\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])(?:\s+|$)|\n+")


class TextNliClassifier(Protocol):
    def classify_text_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[tuple[RelationLabel, float]]: ...


@dataclass(frozen=True, slots=True)
class GroundedClaim:
    text: str
    citation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroundingFailure:
    claim: str
    citation_id: str | None
    reason: str
    label: RelationLabel | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class GroundingReport:
    supported: bool
    reason: str | None
    claim_count: int
    checked_pairs: int
    failures: tuple[GroundingFailure, ...] = field(default_factory=tuple)


def extract_claims(answer: str) -> list[GroundedClaim]:
    """Split generated prose into sentence-level claims and bind nearby citations.

    A citation immediately after terminal punctuation belongs to the preceding
    sentence, so citations work both before and after terminal punctuation.
    """
    normalized = normalize_whitespace(answer)
    if not normalized:
        return []

    canonical = re.sub(
        r"([.!?])\s*((?:\[S\d+\]\s*)+)",
        lambda match: f" {match.group(2).strip()}{match.group(1)} ",
        normalized,
    )

    claims: list[GroundedClaim] = []
    for raw_sentence in _SENTENCE_SPLIT_RE.split(canonical):
        sentence = normalize_whitespace(raw_sentence)
        if not sentence:
            continue
        citations = tuple(dict.fromkeys(_CITATION_RE.findall(sentence)))
        text = normalize_whitespace(_CITATION_RE.sub("", sentence))
        text = re.sub(r"^[-*]\s+", "", text)
        text = re.sub(r"\s+([.!?,;:])", r"\1", text)
        if not text:
            continue
        claims.append(
            GroundedClaim(
                text=text,
                citation_ids=citations,
            )
        )
    return claims


def _is_substantive(claim: GroundedClaim) -> bool:
    return len(content_tokens(claim.text)) >= 2


class ClaimGroundingVerifier:
    def __init__(
        self,
        nli: TextNliClassifier,
        threshold: float = 0.65,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Grounding entailment threshold must be between 0 and 1")
        self.nli = nli
        self.threshold = threshold

    def verify(
        self,
        payload: GeneratedPayload,
        hits: Sequence[SearchHit],
    ) -> GroundingReport:
        extracted_claims = extract_claims(payload.answer)
        inline_citations = {
            citation_id
            for claim in extracted_claims
            for citation_id in claim.citation_ids
        }
        if inline_citations != set(payload.cited_source_ids):
            return GroundingReport(
                supported=False,
                reason="citation_list_mismatch",
                claim_count=len(extracted_claims),
                checked_pairs=0,
            )

        claims = [
            claim
            for claim in extracted_claims
            if _is_substantive(claim)
        ]
        if not claims:
            return GroundingReport(
                supported=False,
                reason="no_groundable_claims",
                claim_count=0,
                checked_pairs=0,
            )

        source_map = {
            f"S{index}": hit
            for index, hit in enumerate(hits, start=1)
        }

        failures: list[GroundingFailure] = []
        checked_pairs = 0

        for claim in claims:
            if not claim.citation_ids:
                failures.append(
                    GroundingFailure(
                        claim=claim.text,
                        citation_id=None,
                        reason="uncited_claim",
                    )
                )
                continue

            normalized_claim = normalize_whitespace(claim.text)
            for citation_id in claim.citation_ids:
                hit = source_map.get(citation_id)
                if hit is None:
                    failures.append(
                        GroundingFailure(
                            claim=claim.text,
                            citation_id=citation_id,
                            reason="unknown_citation",
                        )
                    )
                    continue

                checked_pairs += 1
                source_text = normalize_whitespace(hit.chunk.text)
                if normalized_claim not in source_text:
                    failures.append(
                        GroundingFailure(
                            claim=claim.text,
                            citation_id=citation_id,
                            reason="unsupported_claim_citation",
                        )
                    )

        if failures:
            priority = (
                "uncited_claim",
                "unknown_citation",
                "unsupported_claim_citation",
            )
            reasons = {failure.reason for failure in failures}
            reason = next(item for item in priority if item in reasons)
            return GroundingReport(
                supported=False,
                reason=reason,
                claim_count=len(claims),
                checked_pairs=checked_pairs,
                failures=tuple(failures),
            )

        return GroundingReport(
            supported=True,
            reason=None,
            claim_count=len(claims),
            checked_pairs=checked_pairs,
        )
