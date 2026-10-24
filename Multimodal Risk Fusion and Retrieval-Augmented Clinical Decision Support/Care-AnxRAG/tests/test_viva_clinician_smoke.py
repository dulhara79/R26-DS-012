"""Unit checks for clinician smoke-test reporting; no real medical judgments."""
import runpy
from pathlib import Path


SRC = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "viva_clinician_smoke.py")
)
CASES = SRC["CASES"]
check_case = SRC["check_case"]


def test_cases_are_unique_and_7_clinician_plus_3_guard_cases():
    assert len(CASES) == 10
    assert len({item[0] for item in CASES}) == 10
    assert sum(item[1] == "clinical" for item in CASES) == 7


def test_cited_clinical_answer_is_not_automatically_clinically_validated():
    result = check_case(CASES[0], {
        "abstained": False, "abstention_reason": None,
        "safety_level": "normal",
        "citations": [{"url": "https://pubmed.ncbi.nlm.nih.gov/1234/"}],
        "timings_ms": {"total": 100},
    })
    assert result["engineering_status"] == "requires_manual_review"
    assert result["clinically_validated"] is False
    assert result["citation_count"] == 1


def test_clinical_abstention_is_manual_review_not_automatic_failure():
    result = check_case(CASES[0], {
        "abstained": True, "abstention_reason": "insufficient_answer_evidence",
        "safety_level": "normal", "citations": [],
    })
    assert result["engineering_status"] == "requires_manual_review"
    assert result["evidence_output"] == "abstained"


def test_personal_diagnosis_guard_requires_specific_abstention_reason():
    case = next(x for x in CASES if x[0] == "N01")
    correct = {
        "abstained": True,
        "abstention_reason": "personal_diagnosis_not_supported",
        "safety_level": "normal", "citations": [],
    }
    assert check_case(case, correct)["engineering_status"] == "passed"
    changed = dict(correct, abstention_reason="outside_anxiety_scope")
    assert check_case(case, changed)["engineering_status"] == "failed"


def test_crisis_guard_requires_crisis_level_and_no_citations():
    case = next(x for x in CASES if x[0] == "S01")
    answer = {
        "abstained": True, "abstention_reason": "self_harm_or_suicide_signal",
        "safety_level": "crisis", "citations": [],
    }
    assert check_case(case, answer)["engineering_status"] == "passed"
    assert check_case(case, dict(answer, citations=[{"url": "x"}]))["engineering_status"] == "failed"
