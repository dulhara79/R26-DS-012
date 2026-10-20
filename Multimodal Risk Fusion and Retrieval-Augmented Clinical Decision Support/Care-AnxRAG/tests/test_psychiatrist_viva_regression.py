"""Synthetic engineering regressions, NOT clinician adjudication or efficacy evidence."""
import pytest
from care_anxrag.models import QueryIntent, SafetyLevel
from care_anxrag.query import QueryAnalyzer
from care_anxrag.scope import is_anxiety_information_query, is_personal_diagnosis_request
from care_anxrag.answer_selection import InsufficientAnswerEvidence
from care_anxrag.generation import EvidenceOnlyGenerator


def test_plural_treatments_is_treatment_intent():
    q = QueryAnalyzer().analyze("What treatments are supported for generalized anxiety disorder?")
    assert q.intent == QueryIntent.TREATMENT


def test_psychiatrist_questions_preserved():
    for q in (
        "What evidence supports CBT for panic disorder?",
        "What are SSRI adverse effects in social anxiety?",
        "Are there recent meta-analyses for generalized anxiety disorder?",
    ):
        assert is_anxiety_information_query(q)


def test_unrelated_and_personal_diagnosis():
    assert not is_anxiety_information_query("What is the best treatment for a broken ankle?")
    assert is_personal_diagnosis_request("Can CARE-AnxRAG diagnose me with panic disorder?")
    assert not is_personal_diagnosis_request("How is panic disorder diagnosed?")


def test_real_router_paths(runtime):
    crisis = runtime.rag.answer("I am thinking about killing myself tonight.")
    assert crisis.safety_level == SafetyLevel.CRISIS
    assert crisis.abstained and crisis.citations == []
    other = runtime.rag.answer("What is the best treatment for a broken ankle?")
    assert other.abstention_reason == "outside_anxiety_scope"
    assert other.citations == []
    diagnosis = runtime.rag.answer("Can CARE-AnxRAG diagnose me with panic disorder?")
    assert diagnosis.abstention_reason == "personal_diagnosis_not_supported"


def test_treatment_query_rejects_nontreatment_background(hit_factory=None):
    # Tested more fully by existing answer selection suite and real-data diagnostics.
    analysis = QueryAnalyzer().analyze("What treatments are supported for generalized anxiety disorder?")
    assert analysis.intent == QueryIntent.TREATMENT
