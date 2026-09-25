from datetime import UTC, datetime

from care_anxrag.grounding import ClaimGroundingVerifier, extract_claims
from care_anxrag.models import (
    ChunkRecord,
    DocumentStatus,
    EvidenceLevel,
    GeneratedPayload,
    KnowledgeLayer,
    RelationLabel,
    SearchHit,
)


class StubNli:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def classify_text_pairs(self, pairs):
        self.calls.extend(pairs)
        return self.responses[: len(pairs)]


def _hit(chunk_id: str, text: str) -> SearchHit:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    chunk = ChunkRecord(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        version_id=f"version-{chunk_id}",
        source_id=f"source-{chunk_id}",
        source_name=f"Source {chunk_id}",
        title="Anxiety evidence",
        url="https://example.org/evidence",
        layer=KnowledgeLayer.CLINICAL_CORE,
        status=DocumentStatus.ACTIVE,
        section_path="evidence",
        section_heading="Evidence",
        ordinal=0,
        text=text,
        text_hash=f"hash-{chunk_id}",
        published_at=now,
        updated_at=now,
        retrieved_at=now,
        authority_score=0.9,
        evidence_level=EvidenceLevel.CLINICAL_GUIDELINE,
        evidence_score=1.0,
        topics=["anxiety"],
        metadata={},
    )
    return SearchHit(chunk=chunk, care_score=0.9)


def test_extract_claims_handles_citation_after_terminal_punctuation() -> None:
    claims = extract_claims(
        "CBT is supported for panic disorder. [S1] Exposure therapy is also discussed [S2]."
    )
    assert [claim.text for claim in claims] == [
        "CBT is supported for panic disorder.",
        "Exposure therapy is also discussed.",
    ]
    assert claims[0].citation_ids == ("S1",)
    assert claims[1].citation_ids == ("S2",)


def test_grounding_rejects_uncited_substantive_claim() -> None:
    payload = GeneratedPayload(
        answer="CBT is effective for panic disorder.",
        cited_source_ids=[],
    )
    report = ClaimGroundingVerifier(StubNli([]), threshold=0.65).verify(
        payload,
        [_hit("1", "CBT is effective for panic disorder.")],
    )
    assert not report.supported
    assert report.reason == "uncited_claim"


def test_grounding_accepts_entailed_claim() -> None:
    nli = StubNli([(RelationLabel.ENTAILMENT, 0.94)])
    payload = GeneratedPayload(
        answer="CBT is effective for panic disorder [S1].",
        cited_source_ids=["S1"],
    )
    report = ClaimGroundingVerifier(nli, threshold=0.65).verify(
        payload,
        [_hit("1", "CBT is effective for panic disorder.")],
    )
    assert report.supported
    assert report.reason is None


def test_grounding_rejects_neutral_or_low_confidence_citation() -> None:
    nli = StubNli([(RelationLabel.NEUTRAL, 0.91)])
    payload = GeneratedPayload(
        answer="CBT prevents all future panic attacks [S1].",
        cited_source_ids=["S1"],
    )
    report = ClaimGroundingVerifier(nli, threshold=0.65).verify(
        payload,
        [_hit("1", "CBT can reduce panic symptoms.")],
    )
    assert not report.supported
    assert report.reason == "unsupported_claim_citation"


def test_grounding_rejects_irrelevant_extra_citation() -> None:
    nli = StubNli([
        (RelationLabel.ENTAILMENT, 0.95),
        (RelationLabel.NEUTRAL, 0.92),
    ])
    payload = GeneratedPayload(
        answer="CBT is effective for panic disorder [S1] [S2].",
        cited_source_ids=["S1", "S2"],
    )
    report = ClaimGroundingVerifier(nli, threshold=0.65).verify(
        payload,
        [
            _hit("1", "CBT is effective for panic disorder."),
            _hit("2", "This genetics study describes anxiety-associated loci."),
        ],
    )
    assert not report.supported
    assert report.reason == "unsupported_claim_citation"


def test_grounding_rejects_unknown_inline_source() -> None:
    payload = GeneratedPayload(
        answer="CBT is effective for panic disorder [S9].",
        cited_source_ids=["S9"],
    )
    report = ClaimGroundingVerifier(StubNli([]), threshold=0.65).verify(
        payload,
        [_hit("1", "CBT is effective for panic disorder.")],
    )
    assert not report.supported
    assert report.reason == "unknown_citation"



def test_heuristic_nli_classifies_raw_text_pairs_for_grounding() -> None:
    from care_anxrag.nli import HeuristicNliClassifier

    result = HeuristicNliClassifier().classify_text_pairs(
        [
            (
                "Cognitive behavioural therapy is effective for panic disorder.",
                "Cognitive behavioural therapy is effective for panic disorder.",
            )
        ]
    )

    assert len(result) == 1
    assert result[0][0] == RelationLabel.ENTAILMENT
    assert result[0][1] >= 0.65



def test_grounding_rejects_structured_citation_list_mismatch() -> None:
    nli = StubNli([(RelationLabel.ENTAILMENT, 0.95)])
    payload = GeneratedPayload(
        answer="CBT is effective for panic disorder [S1].",
        cited_source_ids=["S1", "S2"],
    )
    report = ClaimGroundingVerifier(nli, threshold=0.65).verify(
        payload,
        [
            _hit("1", "CBT is effective for panic disorder."),
            _hit("2", "A separate anxiety genetics study."),
        ],
    )

    assert not report.supported
    assert report.reason == "citation_list_mismatch"
