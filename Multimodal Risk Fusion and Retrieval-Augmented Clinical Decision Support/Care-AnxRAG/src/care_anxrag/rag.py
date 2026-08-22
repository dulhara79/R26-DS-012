from __future__ import annotations

from .config import Settings
from .generation import Generator
from .models import (
    AnswerResponse,
    Citation,
    QueryIntent,
    SafetyLevel,
    SearchHit,
)
from .retrieval import CareRetriever
from .safety import safety_message
from .util import normalize_whitespace


class CareAnxRag:
    def __init__(
        self,
        settings: Settings,
        retriever: CareRetriever,
        generator: Generator,
    ):
        self.settings = settings
        self.retriever = retriever
        self.generator = generator

    def answer(self, question: str, include_debug: bool = False) -> AnswerResponse:
        retrieval = self.retriever.retrieve(question)
        safety_level = retrieval.query_analysis.safety_level
        safety_text = safety_message(safety_level, self.settings.crisis_resource_text)
        if safety_level != SafetyLevel.NORMAL:
            return AnswerResponse(
                answer=safety_text or "This question requires urgent human support.",
                confidence=0.0,
                conflict_score=0.0,
                abstained=True,
                abstention_reason=retrieval.abstention_reason,
                safety_level=safety_level,
                safety_message=safety_text,
                latest_evidence_at=None,
                knowledge_base_last_sync_at=retrieval.knowledge_base_last_sync_at,
                retrieval=retrieval if include_debug else None,
            )

        context_hits = [
            hit
            for hit in retrieval.hits
            if not hit.excluded_due_to_conflict
        ][: self.settings.final_context_chunks]
        if retrieval.should_abstain:
            return AnswerResponse(
                answer=self._abstention_answer(retrieval.abstention_reason),
                confidence=retrieval.confidence,
                conflict_score=retrieval.conflict_score,
                abstained=True,
                abstention_reason=retrieval.abstention_reason,
                safety_level=SafetyLevel.NORMAL,
                latest_evidence_at=retrieval.latest_evidence_at,
                knowledge_base_last_sync_at=retrieval.knowledge_base_last_sync_at,
                retrieval=retrieval if include_debug else None,
            )

        try:
            generated = self.generator.generate(question, context_hits, retrieval)
        except Exception as exc:
            return AnswerResponse(
                answer=(
                    "The evidence retrieval completed, but the response could not be generated "
                    "and citation-validated. No unvalidated medical answer was returned."
                ),
                confidence=retrieval.confidence,
                conflict_score=retrieval.conflict_score,
                abstained=True,
                abstention_reason=f"generation_or_citation_validation_failed:{type(exc).__name__}",
                safety_level=SafetyLevel.NORMAL,
                latest_evidence_at=retrieval.latest_evidence_at,
                knowledge_base_last_sync_at=retrieval.knowledge_base_last_sync_at,
                retrieval=retrieval if include_debug else None,
            )

        hit_by_source_id = {f"S{index}": hit for index, hit in enumerate(context_hits, start=1)}
        citations = [
            self._citation(source_id, hit_by_source_id[source_id])
            for source_id in generated.cited_source_ids
            if source_id in hit_by_source_id
        ]
        answer = normalize_whitespace(generated.answer)
        if generated.uncertainty:
            answer = f"{answer}\n\nUncertainty: {normalize_whitespace(generated.uncertainty)}"
        if retrieval.query_analysis.intent in {QueryIntent.DIAGNOSIS, QueryIntent.MEDICATION}:
            answer += (
                "\n\nThis is general evidence-based information, not a diagnosis or an individualized "
                "medication recommendation."
            )
        return AnswerResponse(
            answer=answer,
            citations=citations,
            confidence=retrieval.confidence,
            conflict_score=retrieval.conflict_score,
            abstained=False,
            safety_level=SafetyLevel.NORMAL,
            latest_evidence_at=retrieval.latest_evidence_at,
            knowledge_base_last_sync_at=retrieval.knowledge_base_last_sync_at,
            retrieval=retrieval if include_debug else None,
        )

    @staticmethod
    def _citation(source_id: str, hit: SearchHit) -> Citation:
        excerpt = normalize_whitespace(hit.chunk.text)
        if len(excerpt) > 320:
            excerpt = excerpt[:319].rstrip() + "…"
        return Citation(
            citation_id=source_id,
            chunk_id=hit.chunk.chunk_id,
            title=hit.chunk.title,
            source_name=hit.chunk.source_name,
            source_id=hit.chunk.source_id,
            url=hit.chunk.url,
            published_at=hit.chunk.published_at,
            updated_at=hit.chunk.updated_at,
            evidence_level=hit.chunk.evidence_level,
            excerpt=excerpt,
        )

    @staticmethod
    def _abstention_answer(reason: str | None) -> str:
        reason_map = {
            "no_relevant_evidence_retrieved": "No relevant evidence was retrieved from the active knowledge base.",
            "no_active_evidence_after_conflict_resolution": "No active evidence remained after conflict resolution.",
            "top_evidence_below_relevance_threshold": "The strongest retrieved evidence was not relevant enough.",
            "top_evidence_below_care_threshold": "The relevant evidence did not meet the configured quality threshold.",
            "retrieval_confidence_below_threshold": "The system could not reach its minimum evidence-confidence threshold.",
            "insufficient_source_diversity": "The answer was supported by too few independent sources.",
            "unresolved_high_confidence_evidence_conflict": "High-quality retrieved sources disagreed and the conflict could not be resolved safely.",
        }
        detail = reason_map.get(reason or "", "The available evidence was insufficient or uncertain.")
        return (
            f"I cannot provide a confident evidence-grounded answer. {detail} "
            "A qualified health professional can interpret this question in the context of an individual's situation."
        )
