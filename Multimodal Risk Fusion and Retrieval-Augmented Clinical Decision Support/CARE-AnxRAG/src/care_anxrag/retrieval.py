from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from .config import Settings
from .db import Database
from .embeddings import Embedder
from .evidence import applicability_score, calculate_freshness
from .models import (
    EvidenceRelation,
    KnowledgeLayer,
    RelationLabel,
    RetrievalResult,
    SafetyLevel,
    SearchHit,
)
from .nli import NliClassifier
from .query import QueryAnalyzer
from .rerank import Reranker
from .safety import SafetyRouter
from .util import clamp, fts_query
from .vector_store import VectorStore, collection_for_layer


@dataclass(slots=True)
class _RankedId:
    chunk_id: str
    rank: int
    score: float


class CareRetriever:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
        nli: NliClassifier,
        query_analyzer: QueryAnalyzer | None = None,
        safety_router: SafetyRouter | None = None,
    ):
        self.settings = settings
        self.database = database
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.nli = nli
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        self.safety_router = safety_router or SafetyRouter()

    def retrieve(self, query: str) -> RetrievalResult:
        safety = self.safety_router.assess(query)
        analysis = self.query_analyzer.analyze(query, safety.level, safety.reason)
        if safety.level != SafetyLevel.NORMAL:
            return RetrievalResult(
                query_analysis=analysis,
                hits=[],
                confidence=0.0,
                should_abstain=True,
                abstention_reason=safety.reason,
                knowledge_base_last_sync_at=self.database.last_successful_sync_at(),
            )

        if self.database.count_chunks(status=None) > 0:
            self.database.assert_embedding_identity(self.embedder.model_id)
        query_embedding = self.embedder.embed([analysis.normalized_query])[0]
        dense_ranked = self._dense_search(query_embedding, analysis.preferred_layers)
        lexical_ranked = self._lexical_search(analysis.normalized_query)
        hits = self._fuse(dense_ranked, lexical_ranked)
        if not hits:
            return RetrievalResult(
                query_analysis=analysis,
                hits=[],
                confidence=0.0,
                should_abstain=True,
                abstention_reason="no_relevant_evidence_retrieved",
                knowledge_base_last_sync_at=self.database.last_successful_sync_at(),
            )

        rerank_subset = hits[: self.settings.rerank_candidates]
        rerank_scores = self.reranker.score(analysis.original_query, rerank_subset)
        for hit, score in zip(rerank_subset, rerank_scores, strict=True):
            hit.rerank_score = clamp(score)

        max_rrf = max((hit.rrf_score for hit in hits), default=1.0)
        for hit in hits:
            hit.rrf_normalized = clamp(hit.rrf_score / max_rrf if max_rrf else 0.0)
            hit.freshness_score = calculate_freshness(
                hit.chunk.layer,
                hit.chunk.updated_at,
                hit.chunk.published_at,
                self.settings.clinical_half_life_days,
                self.settings.research_half_life_days,
            )
            hit.applicability_score = applicability_score(
                hit.chunk.topics,
                analysis,
                hit.chunk.layer,
            )
            hit.relevance_score = self._relevance_score(hit)
            hit.care_score = self._care_score(hit)
        hits.sort(key=lambda value: value.care_score, reverse=True)

        # Source authority and evidence quality must never make an unrelated chunk
        # answerable. Conflict analysis and final context operate only on candidates
        # that pass an independent query-evidence relevance gate.
        relevant_hits = [
            hit for hit in hits
            if hit.relevance_score >= self.settings.minimum_relevance_score
        ]
        relation_candidates = self._diversify(
            relevant_hits, limit=min(6, len(relevant_hits)), max_per_document=1
        )
        pairs = [
            (left, right)
            for left, right in itertools.combinations(relation_candidates, 2)
            if left.chunk.document_id != right.chunk.document_id
        ]
        relations = self.nli.classify(pairs)
        conflict_score, unresolved_conflict = self._resolve_conflicts(hits, relations)

        selected = self._diversify(
            [hit for hit in relevant_hits if not hit.excluded_due_to_conflict],
            limit=self.settings.final_context_chunks,
            max_per_document=2,
        )
        selected_ids = {hit.chunk.chunk_id for hit in selected}
        ordered_hits = selected + [hit for hit in hits if hit.chunk.chunk_id not in selected_ids]
        ordered_hits = ordered_hits[: max(self.settings.final_context_chunks, 12)]

        confidence = self._confidence(selected, conflict_score)
        should_abstain, reason = self._abstention(
            selected,
            confidence,
            unresolved_conflict,
        )
        evidence_dates = [
            hit.chunk.updated_at or hit.chunk.published_at
            for hit in selected
            if hit.chunk.updated_at or hit.chunk.published_at
        ]
        latest_evidence = max(evidence_dates) if evidence_dates else None
        return RetrievalResult(
            query_analysis=analysis,
            hits=ordered_hits,
            relations=relations,
            conflict_score=conflict_score,
            confidence=confidence,
            should_abstain=should_abstain,
            abstention_reason=reason,
            latest_evidence_at=latest_evidence,
            knowledge_base_last_sync_at=self.database.last_successful_sync_at(),
        )

    def _dense_search(
        self,
        query_embedding: Sequence[float],
        preferred_layers: Sequence[KnowledgeLayer],
    ) -> list[_RankedId]:
        results: list[tuple[str, float, KnowledgeLayer]] = []
        layers = list(dict.fromkeys(preferred_layers)) or [
            KnowledgeLayer.CLINICAL_CORE,
            KnowledgeLayer.RESEARCH_FRONTIER,
        ]
        for preference_index, layer in enumerate(layers):
            collection = collection_for_layer(
                layer,
                self.settings.clinical_collection,
                self.settings.research_collection,
            )
            vector_hits = self.vector_store.query(
                collection,
                query_embedding,
                limit=self.settings.dense_candidates * 3,
            )
            preference_multiplier = 1.0 if preference_index == 0 else 0.93
            for hit in vector_hits:
                similarity = clamp(1.0 - float(hit.distance))
                results.append((hit.chunk_id, similarity * preference_multiplier, layer))

        active = self.database.get_chunks_by_ids([item[0] for item in results], active_only=True)
        best_by_id: dict[str, float] = {}
        for chunk_id, score, _ in results:
            if chunk_id not in active:
                continue
            best_by_id[chunk_id] = max(best_by_id.get(chunk_id, 0.0), score)
        ranked = sorted(best_by_id.items(), key=lambda item: item[1], reverse=True)
        return [
            _RankedId(chunk_id=chunk_id, rank=index + 1, score=score)
            for index, (chunk_id, score) in enumerate(ranked[: self.settings.dense_candidates])
        ]

    def _lexical_search(self, query: str) -> list[_RankedId]:
        expression = fts_query(query)
        if not expression:
            return []
        rows = self.database.search_fts(expression, limit=self.settings.lexical_candidates * 4)
        active = self.database.get_chunks_by_ids([chunk_id for chunk_id, _ in rows], active_only=True)
        ranked: list[_RankedId] = []
        for chunk_id, _raw_score in rows:
            if chunk_id not in active:
                continue
            rank = len(ranked) + 1
            ranked.append(_RankedId(chunk_id=chunk_id, rank=rank, score=1.0 / rank))
            if len(ranked) >= self.settings.lexical_candidates:
                break
        return ranked

    def _fuse(
        self,
        dense_ranked: Sequence[_RankedId],
        lexical_ranked: Sequence[_RankedId],
    ) -> list[SearchHit]:
        ids = list(dict.fromkeys(
            [item.chunk_id for item in dense_ranked] + [item.chunk_id for item in lexical_ranked]
        ))
        chunks = self.database.get_chunks_by_ids(ids, active_only=True)
        hit_by_id = {chunk_id: SearchHit(chunk=chunk) for chunk_id, chunk in chunks.items()}
        for item in dense_ranked:
            if item.chunk_id not in hit_by_id:
                continue
            hit = hit_by_id[item.chunk_id]
            hit.dense_rank = item.rank
            hit.dense_score = item.score
            hit.rrf_score += 1.0 / (self.settings.rrf_k + item.rank)
        for item in lexical_ranked:
            if item.chunk_id not in hit_by_id:
                continue
            hit = hit_by_id[item.chunk_id]
            hit.lexical_rank = item.rank
            hit.lexical_score = item.score
            hit.rrf_score += 1.0 / (self.settings.rrf_k + item.rank)
        hits = sorted(hit_by_id.values(), key=lambda item: item.rrf_score, reverse=True)
        return hits[: self.settings.fused_candidates]

    @staticmethod
    def _relevance_score(hit: SearchHit) -> float:
        # Deliberately excludes authority/evidence/freshness: this is the guardrail
        # against highly authoritative but irrelevant evidence. Lexical presence is
        # useful after stop-word removal; the reranker remains the strongest signal.
        lexical_presence = 1.0 if hit.lexical_rank is not None else 0.0
        return clamp(
            0.35 * hit.dense_score
            + 0.45 * hit.rerank_score
            + 0.15 * lexical_presence
            + 0.05 * hit.applicability_score
        )

    def _care_score(self, hit: SearchHit) -> float:
        weights = self.settings.weights
        return clamp(
            weights.semantic * hit.dense_score
            + weights.lexical * hit.lexical_score
            + weights.rrf * hit.rrf_normalized
            + weights.rerank * hit.rerank_score
            + weights.authority * hit.chunk.authority_score
            + weights.evidence * hit.chunk.evidence_score
            + weights.freshness * hit.freshness_score
            + weights.applicability * hit.applicability_score
        )

    def _resolve_conflicts(
        self,
        hits: list[SearchHit],
        relations: Sequence[EvidenceRelation],
    ) -> tuple[float, float]:
        hit_by_id = {hit.chunk.chunk_id: hit for hit in hits}
        weighted_conflict = 0.0
        total_weight = 0.0
        unresolved = 0.0
        for relation in relations:
            left = hit_by_id.get(relation.left_chunk_id)
            right = hit_by_id.get(relation.right_chunk_id)
            if left is None or right is None:
                continue
            pair_weight = min(left.care_score, right.care_score)
            total_weight += pair_weight
            if (
                relation.label != RelationLabel.CONTRADICTION
                or relation.confidence < self.settings.contradiction_threshold
            ):
                continue
            weighted_conflict += pair_weight * relation.confidence
            left_strength = self._evidence_strength(left)
            right_strength = self._evidence_strength(right)
            difference = abs(left_strength - right_strength)
            if difference >= self.settings.dominance_margin:
                weaker = right if left_strength > right_strength else left
                stronger = left if left_strength > right_strength else right
                weaker.excluded_due_to_conflict = True
                weaker.exclusion_reason = (
                    f"Contradicted by stronger evidence from {stronger.chunk.source_name} "
                    f"(NLI confidence {relation.confidence:.2f})"
                )
            else:
                unresolved += pair_weight * relation.confidence
        conflict_score = clamp(weighted_conflict / total_weight) if total_weight else 0.0
        unresolved_score = clamp(unresolved / total_weight) if total_weight else 0.0
        return conflict_score, unresolved_score

    @staticmethod
    def _evidence_strength(hit: SearchHit) -> float:
        return clamp(
            0.42 * hit.chunk.authority_score
            + 0.38 * hit.chunk.evidence_score
            + 0.12 * hit.freshness_score
            + 0.08 * hit.care_score
        )

    @staticmethod
    def _diversify(
        hits: Sequence[SearchHit],
        limit: int,
        max_per_document: int,
    ) -> list[SearchHit]:
        output: list[SearchHit] = []
        per_document: dict[str, int] = {}
        for hit in hits:
            count = per_document.get(hit.chunk.document_id, 0)
            if count >= max_per_document:
                continue
            output.append(hit)
            per_document[hit.chunk.document_id] = count + 1
            if len(output) >= limit:
                break
        return output

    @staticmethod
    def _confidence(hits: Sequence[SearchHit], conflict_score: float) -> float:
        if not hits:
            return 0.0
        top_score = hits[0].care_score
        mean_top = statistics.fmean(hit.care_score for hit in hits[:3])
        distinct_sources = len({hit.chunk.source_id for hit in hits})
        diversity = min(1.0, distinct_sources / 3.0)
        authority_coverage = max(hit.chunk.authority_score for hit in hits)
        consensus = 1.0 - conflict_score
        return clamp(
            0.32 * top_score
            + 0.24 * mean_top
            + 0.14 * diversity
            + 0.15 * authority_coverage
            + 0.15 * consensus
        )

    def _abstention(
        self,
        hits: Sequence[SearchHit],
        confidence: float,
        unresolved_conflict: float,
    ) -> tuple[bool, str | None]:
        if not hits:
            return True, "no_active_evidence_after_conflict_resolution"
        if hits[0].relevance_score < self.settings.minimum_relevance_score:
            return True, "top_evidence_below_relevance_threshold"
        if hits[0].care_score < self.settings.minimum_care_score:
            return True, "top_evidence_below_care_threshold"
        if confidence < self.settings.minimum_confidence:
            return True, "retrieval_confidence_below_threshold"
        if len({hit.chunk.source_id for hit in hits}) < self.settings.min_distinct_sources:
            return True, "insufficient_source_diversity"
        if unresolved_conflict > self.settings.unresolved_conflict_threshold:
            return True, "unresolved_high_confidence_evidence_conflict"
        return False, None
