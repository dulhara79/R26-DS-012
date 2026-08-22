from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class KnowledgeLayer(StrEnum):
    CLINICAL_CORE = "clinical_core"
    RESEARCH_FRONTIER = "research_frontier"


class DocumentStatus(StrEnum):
    STAGING = "staging"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class EvidenceLevel(StrEnum):
    CLINICAL_GUIDELINE = "clinical_guideline"
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    RANDOMIZED_CONTROLLED_TRIAL = "randomized_controlled_trial"
    COHORT_STUDY = "cohort_study"
    CASE_CONTROL_STUDY = "case_control_study"
    CROSS_SECTIONAL_STUDY = "cross_sectional_study"
    CASE_REPORT = "case_report"
    GOVERNMENT_HEALTH_INFORMATION = "government_health_information"
    RESEARCH_UPDATE = "research_update"
    GENERAL_INFORMATION = "general_information"
    UNKNOWN = "unknown"


class QueryIntent(StrEnum):
    TREATMENT = "treatment"
    SYMPTOMS = "symptoms"
    DIAGNOSIS = "diagnosis"
    CAUSES = "causes"
    MEDICATION = "medication"
    SELF_HELP = "self_help"
    RECENT_RESEARCH = "recent_research"
    GENERAL = "general"


class SafetyLevel(StrEnum):
    NORMAL = "normal"
    URGENT = "urgent"
    CRISIS = "crisis"


class RelationLabel(StrEnum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class SourceConfig(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]+$")
    name: str
    connector: Literal[
        "http_page",
        "pubmed",
        "pmc",
        "nice_syndication",
        "local_files",
    ]
    enabled: bool = True
    publish_to_rag: bool = True
    auto_promote: bool = False
    layer: KnowledgeLayer
    authority_score: float = Field(ge=0.0, le=1.0)
    evidence_level: EvidenceLevel = EvidenceLevel.UNKNOWN
    check_interval_minutes: int = Field(default=1440, gt=0)
    settings: dict[str, Any] = Field(default_factory=dict)


class SourceState(StrictModel):
    source_id: str
    last_attempt_at: datetime | None = None
    last_success_at: datetime | None = None
    last_changed_at: datetime | None = None
    etag: str | None = None
    last_modified: str | None = None
    cursor: str | None = None
    last_error: str | None = None


class Section(StrictModel):
    path: str
    heading: str
    text: str
    ordinal: int = Field(ge=0)
    content_hash: str = ""


class RawDocument(StrictModel):
    source_id: str
    external_id: str
    title: str
    text: str
    url: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    retrieved_at: datetime
    authors: list[str] = Field(default_factory=list)
    language: str = "en"
    publication_types: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("external_id", "title", "text")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value must not be empty")
        return value


class DocumentVersion(StrictModel):
    document_id: str
    version_id: str
    source_id: str
    external_id: str
    title: str
    text: str
    url: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    retrieved_at: datetime
    content_hash: str
    status: DocumentStatus
    layer: KnowledgeLayer
    evidence_level: EvidenceLevel
    authority_score: float = Field(ge=0.0, le=1.0)
    evidence_score: float = Field(ge=0.0, le=1.0)
    topics: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)
    supersedes_version_id: str | None = None
    rejection_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(StrictModel):
    chunk_id: str
    document_id: str
    version_id: str
    source_id: str
    source_name: str
    title: str
    url: str | None = None
    layer: KnowledgeLayer
    status: DocumentStatus
    section_path: str
    section_heading: str
    ordinal: int = Field(ge=0)
    text: str
    text_hash: str
    published_at: datetime | None = None
    updated_at: datetime | None = None
    retrieved_at: datetime
    authority_score: float = Field(ge=0.0, le=1.0)
    evidence_level: EvidenceLevel
    evidence_score: float = Field(ge=0.0, le=1.0)
    topics: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryAnalysis(StrictModel):
    original_query: str
    normalized_query: str
    intent: QueryIntent
    anxiety_subtypes: list[str] = Field(default_factory=list)
    population: str | None = None
    wants_recent: bool = False
    preferred_layers: list[KnowledgeLayer] = Field(default_factory=list)
    safety_level: SafetyLevel = SafetyLevel.NORMAL
    safety_reason: str | None = None


class SearchHit(StrictModel):
    chunk: ChunkRecord
    dense_rank: int | None = None
    lexical_rank: int | None = None
    dense_score: float = Field(default=0.0, ge=0.0, le=1.0)
    lexical_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rrf_score: float = Field(default=0.0, ge=0.0)
    rrf_normalized: float = Field(default=0.0, ge=0.0, le=1.0)
    rerank_score: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    applicability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    care_score: float = Field(default=0.0, ge=0.0, le=1.0)
    excluded_due_to_conflict: bool = False
    exclusion_reason: str | None = None


class EvidenceRelation(StrictModel):
    left_chunk_id: str
    right_chunk_id: str
    label: RelationLabel
    confidence: float = Field(ge=0.0, le=1.0)


class RetrievalResult(StrictModel):
    query_analysis: QueryAnalysis
    hits: list[SearchHit]
    relations: list[EvidenceRelation] = Field(default_factory=list)
    conflict_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    should_abstain: bool = False
    abstention_reason: str | None = None
    latest_evidence_at: datetime | None = None
    knowledge_base_last_sync_at: datetime | None = None


class Citation(StrictModel):
    citation_id: str
    chunk_id: str
    title: str
    source_name: str
    source_id: str
    url: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    evidence_level: EvidenceLevel
    excerpt: str


class GeneratedPayload(StrictModel):
    answer: str
    cited_source_ids: list[str]
    uncertainty: str | None = None


class AnswerResponse(StrictModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    conflict_score: float = Field(ge=0.0, le=1.0)
    abstained: bool
    abstention_reason: str | None = None
    safety_level: SafetyLevel
    safety_message: str | None = None
    latest_evidence_at: datetime | None = None
    knowledge_base_last_sync_at: datetime | None = None
    retrieval: RetrievalResult | None = None


class AskRequest(StrictModel):
    question: str = Field(min_length=2, max_length=4000)
    include_debug: bool = False


class SyncRequest(StrictModel):
    source_ids: list[str] = Field(default_factory=list)
    dry_run: bool = False
    force: bool = False


class SyncSummary(StrictModel):
    run_id: str
    started_at: datetime
    finished_at: datetime
    source_ids: list[str]
    discovered: int = 0
    unchanged: int = 0
    staged: int = 0
    promoted: int = 0
    rejected: int = 0
    failed: int = 0
    errors: list[str] = Field(default_factory=list)


class HealthStatus(StrictModel):
    status: Literal["ok", "degraded", "error"]
    database: bool
    vector_store: bool
    ollama: bool | None
    details: dict[str, Any] = Field(default_factory=dict)
