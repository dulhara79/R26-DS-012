from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from .rag import CareAnxRag
from .retrieval import CareRetriever


class BenchmarkItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    relevant_external_ids: list[str] = Field(default_factory=list)
    relevant_source_ids: list[str] = Field(default_factory=list)
    must_abstain: bool = False
    expects_conflict: bool = False


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
            "per_item": self.per_item,
        }


def load_benchmark(path: Path | str) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            items.append(BenchmarkItem.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"Invalid benchmark JSONL at line {line_number}: {exc}") from exc
    return items


def evaluate(
    retriever: CareRetriever,
    rag: CareAnxRag,
    items: Iterable[BenchmarkItem],
) -> EvaluationReport:
    rows: list[dict[str, Any]] = []
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    abstention_matches: list[float] = []
    conflict_matches: list[float] = []
    citation_checks: list[float] = []

    for item in items:
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
                "retrieved_chunk_ids": [hit.chunk.chunk_id for hit in top_hits],
            }
        )

    count = len(rows)
    mean = lambda values: sum(values) / len(values) if values else 0.0
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
        per_item=rows,
    )


def _is_relevant(source_id: str, metadata: dict[str, Any], item: BenchmarkItem) -> bool:
    external_id = str(metadata.get("external_id", ""))
    return source_id in item.relevant_source_ids or external_id in item.relevant_external_ids
