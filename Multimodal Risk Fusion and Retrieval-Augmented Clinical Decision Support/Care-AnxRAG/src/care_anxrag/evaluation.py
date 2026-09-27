from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from .models import DocumentStatus
from .rag import CareAnxRag
from .retrieval import CareRetriever


class BenchmarkItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    relevant_external_ids: list[str] = Field(default_factory=list)
    relevant_source_ids: list[str] = Field(default_factory=list)
    prohibited_external_ids: list[str] = Field(default_factory=list)
    prohibited_source_ids: list[str] = Field(default_factory=list)
    must_abstain: bool = False
    expects_conflict: bool = False

    stratum: str = "unspecified"
    split: str = "unassigned"
    intent: str | None = None
    anxiety_subtypes: list[str] = Field(default_factory=list)
    treatments: list[str] = Field(default_factory=list)
    population: str | None = None
    gold_evidence_excerpts: list[str] = Field(default_factory=list)
    prohibited_claims: list[str] = Field(default_factory=list)
    annotator_ids: list[str] = Field(default_factory=list)
    adjudicated: bool = False


@dataclass(slots=True)
class EvaluationReport:
    count: int
    retrieval_evaluable_count: int
    recall_at_5: float
    precision_at_5: float
    mrr: float
    ndcg_at_5: float
    abstention_accuracy: float
    conflict_accuracy: float
    citation_validity: float
    extractive_evaluable_count: int
    extractive_faithfulness: float
    gold_evidence_evaluable_count: int
    gold_evidence_coverage: float
    prohibited_evidence_evaluable_count: int
    prohibited_evidence_intrusion_rate: float
    active_version_evaluable_count: int
    active_version_accuracy: float
    stale_evidence_evaluable_count: int
    stale_evidence_intrusion_rate: float
    per_stratum: dict[str, dict[str, Any]]
    per_item: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "retrieval_evaluable_count": self.retrieval_evaluable_count,
            "recall_at_5": self.recall_at_5,
            "precision_at_5": self.precision_at_5,
            "mrr": self.mrr,
            "ndcg_at_5": self.ndcg_at_5,
            "abstention_accuracy": self.abstention_accuracy,
            "conflict_accuracy": self.conflict_accuracy,
            "citation_validity": self.citation_validity,
            "extractive_evaluable_count": self.extractive_evaluable_count,
            "extractive_faithfulness": self.extractive_faithfulness,
            "gold_evidence_evaluable_count": self.gold_evidence_evaluable_count,
            "gold_evidence_coverage": self.gold_evidence_coverage,
            "prohibited_evidence_evaluable_count": self.prohibited_evidence_evaluable_count,
            "prohibited_evidence_intrusion_rate": self.prohibited_evidence_intrusion_rate,
            "active_version_evaluable_count": self.active_version_evaluable_count,
            "active_version_accuracy": self.active_version_accuracy,
            "stale_evidence_evaluable_count": self.stale_evidence_evaluable_count,
            "stale_evidence_intrusion_rate": self.stale_evidence_intrusion_rate,
            "per_stratum": self.per_stratum,
            "per_item": self.per_item,
        }


def load_benchmark(path: Path | str) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = BenchmarkItem.model_validate_json(line)
        except Exception as exc:
            raise ValueError(f"Invalid benchmark JSONL at line {line_number}: {exc}") from exc
        if item.id in seen_ids:
            raise ValueError(
                f"Duplicate benchmark item id {item.id!r} at line {line_number}"
            )
        seen_ids.add(item.id)
        items.append(item)
    return items


def evaluate(
    retriever: CareRetriever,
    rag: CareAnxRag,
    items: Iterable[BenchmarkItem],
) -> EvaluationReport:
    benchmark_items = list(items)
    _validate_locked_test_items(benchmark_items)

    rows: list[dict[str, Any]] = []
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    abstention_matches: list[float] = []
    conflict_matches: list[float] = []
    citation_checks: list[float] = []
    extractive_checks: list[float] = []
    gold_evidence_coverages: list[float] = []
    prohibited_intrusions: list[float] = []
    active_version_accuracies: list[float] = []
    stale_evidence_intrusions: list[float] = []
    stratum_rows: dict[str, list[dict[str, Any]]] = {}

    for item in benchmark_items:
        retrieval = retriever.retrieve(item.question)
        top_hits = [hit for hit in retrieval.hits if not hit.excluded_due_to_conflict][:5]
        relevant_flags = [_is_relevant(hit.chunk.source_id, hit.chunk.metadata, item) for hit in top_hits]
        has_retrieval_labels = bool(item.relevant_external_ids or item.relevant_source_ids)
        recall: float | None = None
        precision: float | None = None
        rr: float | None = None
        ndcg: float | None = None
        if has_retrieval_labels:
            relevant_total = len(item.relevant_external_ids) + len(item.relevant_source_ids)
            retrieved_relevant = sum(relevant_flags)
            recall = min(1.0, retrieved_relevant / relevant_total)
            precision = retrieved_relevant / max(1, len(top_hits))
            first_rank = next(
                (index + 1 for index, flag in enumerate(relevant_flags) if flag),
                None,
            )
            rr = 0.0 if first_rank is None else 1.0 / first_rank
            dcg = sum(flag / math.log2(index + 2) for index, flag in enumerate(relevant_flags))
            ideal_hits = min(5, relevant_total)
            idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
            ndcg = 0.0 if idcg == 0 else dcg / idcg
        predicted_conflict = retrieval.conflict_score > 0.0
        answer = rag.answer(item.question)
        citations_valid = all(
            citation.citation_id in answer.answer for citation in answer.citations
        ) if answer.citations else answer.abstained

        extractive_faithful: bool | None = None
        if not answer.abstained:
            extractive_faithful = _is_extractive_answer(answer, top_hits)
            extractive_checks.append(float(extractive_faithful))

        gold_evidence_coverage: float | None = None
        if item.gold_evidence_excerpts:
            gold_evidence_coverage = _gold_evidence_coverage(
                answer.answer if not answer.abstained else "",
                item.gold_evidence_excerpts,
            )
            gold_evidence_coverages.append(gold_evidence_coverage)

        has_prohibited_labels = bool(
            item.prohibited_external_ids or item.prohibited_source_ids
        )
        prohibited_intrusion: bool | None = None
        if has_prohibited_labels:
            prohibited_intrusion = any(
                _is_prohibited(
                    hit.chunk.source_id,
                    hit.chunk.metadata,
                    item,
                )
                for hit in top_hits
            )
            prohibited_intrusions.append(float(prohibited_intrusion))

        active_version_accuracy: float | None = None
        stale_evidence_intrusion_rate: float | None = None
        if top_hits:
            active_version_accuracy = (
                sum(
                    hit.chunk.status == DocumentStatus.ACTIVE
                    for hit in top_hits
                )
                / len(top_hits)
            )
            stale_evidence_intrusion_rate = (
                sum(
                    hit.chunk.status
                    in {
                        DocumentStatus.SUPERSEDED,
                        DocumentStatus.WITHDRAWN,
                    }
                    for hit in top_hits
                )
                / len(top_hits)
            )
            active_version_accuracies.append(active_version_accuracy)
            stale_evidence_intrusions.append(stale_evidence_intrusion_rate)

        if has_retrieval_labels:
            if recall is None or precision is None or rr is None or ndcg is None:
                raise RuntimeError("Retrieval metrics were not calculated for a labeled item")
            recalls.append(recall)
            precisions.append(precision)
            reciprocal_ranks.append(rr)
            ndcgs.append(ndcg)
        abstention_matches.append(float(answer.abstained == item.must_abstain))
        conflict_matches.append(float(predicted_conflict == item.expects_conflict))
        citation_checks.append(float(citations_valid))
        rows.append(
            {
                "id": item.id,
                "recall_at_5": recall,
                "precision_at_5": precision,
                "reciprocal_rank": rr,
                "ndcg_at_5": ndcg,
                "predicted_abstain": answer.abstained,
                "expected_abstain": item.must_abstain,
                "conflict_score": retrieval.conflict_score,
                "expected_conflict": item.expects_conflict,
                "citation_valid": citations_valid,
                "extractive_faithful": extractive_faithful,
                "gold_evidence_coverage": gold_evidence_coverage,
                "prohibited_evidence_intrusion": prohibited_intrusion,
                "active_version_accuracy": active_version_accuracy,
                "stale_evidence_intrusion_rate": stale_evidence_intrusion_rate,
                "stratum": item.stratum,
                "split": item.split,
                "retrieved_chunk_ids": [hit.chunk.chunk_id for hit in top_hits],
            }
        )
        stratum_rows.setdefault(item.stratum, []).append(rows[-1])

    count = len(rows)
    mean = lambda values: sum(values) / len(values) if values else 0.0
    per_stratum = {
        stratum: {
            "count": len(group),
            "abstention_accuracy": mean(
                [
                    float(row["predicted_abstain"] == row["expected_abstain"])
                    for row in group
                ]
            ),
            "extractive_faithfulness": mean(
                [
                    float(row["extractive_faithful"])
                    for row in group
                    if row["extractive_faithful"] is not None
                ]
            ),
        }
        for stratum, group in sorted(stratum_rows.items())
    }
    return EvaluationReport(
        count=count,
        retrieval_evaluable_count=len(recalls),
        recall_at_5=mean(recalls),
        precision_at_5=mean(precisions),
        mrr=mean(reciprocal_ranks),
        ndcg_at_5=mean(ndcgs),
        abstention_accuracy=mean(abstention_matches),
        conflict_accuracy=mean(conflict_matches),
        citation_validity=mean(citation_checks),
        extractive_evaluable_count=len(extractive_checks),
        extractive_faithfulness=mean(extractive_checks),
        gold_evidence_evaluable_count=len(gold_evidence_coverages),
        gold_evidence_coverage=mean(gold_evidence_coverages),
        prohibited_evidence_evaluable_count=len(prohibited_intrusions),
        prohibited_evidence_intrusion_rate=mean(prohibited_intrusions),
        active_version_evaluable_count=len(active_version_accuracies),
        active_version_accuracy=mean(active_version_accuracies),
        stale_evidence_evaluable_count=len(stale_evidence_intrusions),
        stale_evidence_intrusion_rate=mean(stale_evidence_intrusions),
        per_stratum=per_stratum,
        per_item=rows,
    )


ABLATION_MODES: tuple[tuple[str, str], ...] = (
    ("B0_dense_only", "dense_only"),
    ("B1_lexical_only", "lexical_only"),
    ("B2_hybrid_rrf", "hybrid_rrf"),
    ("B3_hybrid_rerank", "hybrid_rerank"),
    ("B4_care", "care"),
    ("B5_care_conflict", "care_conflict"),
    ("CARE_full", "full"),
)


def evaluate_ablation(
    retriever: CareRetriever,
    rag: CareAnxRag,
    items: Iterable[BenchmarkItem],
) -> dict[str, EvaluationReport]:
    benchmark_items = list(items)
    original_mode = retriever.settings.retrieval_mode
    reports: dict[str, EvaluationReport] = {}
    try:
        for label, mode in ABLATION_MODES:
            retriever.settings.retrieval_mode = mode
            reports[label] = evaluate(
                retriever,
                rag,
                benchmark_items,
            )
    finally:
        retriever.settings.retrieval_mode = original_mode
    return reports


def _validate_locked_test_items(items: Iterable[BenchmarkItem]) -> None:
    for item in items:
        if item.split.lower() != "test":
            continue
        reviewers = {
            annotator.strip()
            for annotator in item.annotator_ids
            if annotator.strip()
        }
        if not item.adjudicated or len(reviewers) < 2:
            raise ValueError(
                f"Benchmark locked test item {item.id!r} must be adjudicated "
                "and have at least two distinct annotator IDs"
            )

def _is_relevant(source_id: str, metadata: dict[str, Any], item: BenchmarkItem) -> bool:
    external_id = str(metadata.get("external_id", ""))
    return source_id in item.relevant_source_ids or external_id in item.relevant_external_ids



def _is_prohibited(
    source_id: str,
    metadata: dict[str, Any],
    item: BenchmarkItem,
) -> bool:
    external_id = str(metadata.get("external_id", ""))
    return (
        source_id in item.prohibited_source_ids
        or external_id in item.prohibited_external_ids
    )


def _is_extractive_answer(
    answer: Any,
    top_hits: Iterable[Any],
) -> bool:
    if answer.abstained:
        return True
    if not answer.citations:
        return False

    source_text_by_id = {
        f"S{index}": " ".join(str(hit.chunk.text).split())
        for index, hit in enumerate(top_hits, start=1)
    }
    if not source_text_by_id:
        return False

    for raw_line in str(answer.answer).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = line.removeprefix("-").strip()
        citation_ids = re.findall(r"\[(S\d+)\]", line)
        medical_text = re.sub(r"\s*\[S\d+\]\s*$", "", line).strip()
        if not medical_text or not citation_ids:
            return False
        normalized = " ".join(medical_text.split())
        if not all(
            citation_id in source_text_by_id
            and normalized in source_text_by_id[citation_id]
            for citation_id in citation_ids
        ):
            return False
    return True


def _gold_evidence_coverage(
    answer_text: str,
    gold_excerpts: Iterable[str],
) -> float:
    gold = [
        " ".join(str(excerpt).split())
        for excerpt in gold_excerpts
        if str(excerpt).strip()
    ]
    if not gold:
        return 0.0

    answer_without_citations = re.sub(r"\[S\d+\]", "", str(answer_text))
    normalized_answer = " ".join(answer_without_citations.split())
    matched = sum(excerpt in normalized_answer for excerpt in gold)
    return matched / len(gold)
