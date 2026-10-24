"""The offline diagnostic must never invent answerability or review labels."""
from pathlib import Path
import runpy

from care_anxrag.models import RetrievalResult
from care_anxrag.query import QueryAnalyzer

TOOLS = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "diagnose_clinician_recall.py")
)


def test_empty_retrieval_reports_no_answer_candidates():
    question = "What treatments are supported for generalized anxiety disorder?"
    retrieval = RetrievalResult(
        query_analysis=QueryAnalyzer().analyze(question),
        hits=[], confidence=0.7, should_abstain=True,
        abstention_reason="no_relevant_evidence_retrieved",
    )
    result = TOOLS["inspect"](retrieval)
    assert result["eligible_hit_count"] == 0
    assert result["retrieval_abstained"] is True
    assert result["retrieval_abstention_reason"] == "no_relevant_evidence_retrieved"


def test_professional_question_ids_are_unique():
    questions = TOOLS["QUESTIONS"]
    assert len(questions) == 7
    assert [case for case, _ in questions] == [
        f"P{i:02}" for i in range(1, 8)
    ]
