from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .models import SafetyLevel
from .safety import SafetyAssessment


class SafetyBenchmarkItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    expected_level: SafetyLevel
    stratum: str = "unspecified"
    split: str = "unassigned"
    annotator_ids: list[str] = Field(default_factory=list)
    adjudicated: bool = False


class SafetyAssessor(Protocol):
    def assess(self, text: str) -> SafetyAssessment: ...


@dataclass(slots=True)
class SafetyEvaluationReport:
    count: int
    accuracy: float
    crisis_recall: float
    urgent_recall: float
    normal_false_positive_rate: float
    confusion_matrix: dict[str, dict[str, int]]
    per_stratum: dict[str, dict[str, Any]]
    per_item: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "accuracy": self.accuracy,
            "crisis_recall": self.crisis_recall,
            "urgent_recall": self.urgent_recall,
            "normal_false_positive_rate": self.normal_false_positive_rate,
            "confusion_matrix": self.confusion_matrix,
            "per_stratum": self.per_stratum,
            "per_item": self.per_item,
        }


def load_safety_benchmark(
    path: Path | str,
) -> list[SafetyBenchmarkItem]:
    items: list[SafetyBenchmarkItem] = []
    seen_ids: set[str] = set()

    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = SafetyBenchmarkItem.model_validate_json(line)
        except Exception as exc:
            raise ValueError(
                f"Invalid safety benchmark JSONL at line {line_number}: {exc}"
            ) from exc

        if item.id in seen_ids:
            raise ValueError(
                f"Duplicate safety benchmark item id {item.id!r} "
                f"at line {line_number}"
            )
        seen_ids.add(item.id)
        items.append(item)

    return items


def evaluate_safety(
    router: SafetyAssessor,
    items: Iterable[SafetyBenchmarkItem],
) -> SafetyEvaluationReport:
    levels = [
        SafetyLevel.NORMAL,
        SafetyLevel.URGENT,
        SafetyLevel.CRISIS,
    ]
    confusion = {
        expected.value: {
            predicted.value: 0
            for predicted in levels
        }
        for expected in levels
    }

    rows: list[dict[str, Any]] = []
    strata: dict[str, list[dict[str, Any]]] = {}

    for item in items:
        assessment = router.assess(item.text)
        predicted = assessment.level
        confusion[item.expected_level.value][predicted.value] += 1

        row = {
            "id": item.id,
            "expected_level": item.expected_level.value,
            "predicted_level": predicted.value,
            "correct": predicted == item.expected_level,
            "reason": assessment.reason,
            "stratum": item.stratum,
            "split": item.split,
        }
        rows.append(row)
        strata.setdefault(item.stratum, []).append(row)

    count = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)

    def recall(level: SafetyLevel) -> float:
        expected_total = sum(confusion[level.value].values())
        if expected_total == 0:
            return 0.0
        return (
            confusion[level.value][level.value]
            / expected_total
        )

    normal_total = sum(confusion[SafetyLevel.NORMAL.value].values())
    normal_false_positives = (
        normal_total
        - confusion[SafetyLevel.NORMAL.value][SafetyLevel.NORMAL.value]
    )
    normal_false_positive_rate = (
        normal_false_positives / normal_total
        if normal_total
        else 0.0
    )

    per_stratum = {
        stratum: {
            "count": len(group),
            "accuracy": (
                sum(bool(row["correct"]) for row in group)
                / len(group)
            ),
        }
        for stratum, group in sorted(strata.items())
    }

    return SafetyEvaluationReport(
        count=count,
        accuracy=correct / count if count else 0.0,
        crisis_recall=recall(SafetyLevel.CRISIS),
        urgent_recall=recall(SafetyLevel.URGENT),
        normal_false_positive_rate=normal_false_positive_rate,
        confusion_matrix=confusion,
        per_stratum=per_stratum,
        per_item=rows,
    )
