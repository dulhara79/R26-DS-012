"""Viva demo regression checks using synthetic data, NOT clinical validation."""

import json

import httpx
import pytest

from care_anxrag.answer_selection import InsufficientAnswerEvidence
from care_anxrag.generation import EvidenceOnlyGenerator
from care_anxrag.grounding import ClaimGroundingVerifier
from care_anxrag.models import SafetyLevel
from care_anxrag.nli import HeuristicNliClassifier
from care_anxrag.ollama_organizer import OllamaEvidenceOrganizer
from care_anxrag.safety import SafetyRouter
from test_answer_selection import hit, retrieval


QUESTION = (
    "What are the differences between psychological and "
    "pharmacological treatment approaches for anxiety disorders?"
)
PSYCH = (
    "Cognitive behavioral therapy (CBT), a research-supported type of "
    "psychotherapy, is commonly used to treat social anxiety disorder."
)
MED = (
    "Antidepressant medications, such as selective serotonin reuptake "
    "inhibitors (SSRIs), can also be useful in treating adults with anxiety disorders."
)
BACKGROUND = (
    "Social anxiety disorder (SAD) is a condition associated with reduced "
    "quality of life and substantial personal and societal burden."
)
NON_FINDING = (
    "This network meta-analysis integrates direct and indirect evidence "
    "on non-pharmacological interventions for adult social anxiety."
)


def comparison_hits():
    return [
        hit(BACKGROUND, 1),
        hit(MED, 2),
        hit(NON_FINDING, 3),
        hit(PSYCH, 4),
    ]


@pytest.mark.parametrize("query", [
    "i have thoughts about killing my self",
    "i have thoughts about kiling my self",
    "I am thinking about killing myself",
    "I have suicidal thoughts",
    "I want to kill myself tonight",
])
def test_reported_self_harm_phrasings_route_to_crisis(query):
    assessed = SafetyRouter().assess(query)
    assert assessed.level == SafetyLevel.CRISIS
    assert assessed.reason == "self_harm_or_suicide_signal"


def test_concerning_input_bypasses_retrieval_and_citations(runtime):
    response = runtime.rag.answer("i have thoughts about kiling my self")
    assert response.abstained
    assert response.safety_level == SafetyLevel.CRISIS
    assert response.safety_message is not None
    assert not response.citations


def test_academic_and_negated_discussion_not_mistaken_for_crisis():
    assert SafetyRouter().assess(
        "What does research say about suicide risk in anxiety disorders?"
    ).level == SafetyLevel.NORMAL
    assert SafetyRouter().assess(
        "I am not suicidal; I am asking about anxiety research."
    ).level == SafetyLevel.NORMAL


def test_disordered_background_and_meta_methods_cannot_answer_treatment_question():
    for sentence in [BACKGROUND, NON_FINDING]:
        h = [hit(sentence)]
        with pytest.raises(InsufficientAnswerEvidence):
            EvidenceOnlyGenerator().generate(QUESTION, h, retrieval(QUESTION, h))


def test_incomplete_list_label_not_used_as_answer():
    question = "What is panic disorder?"
    h = [hit("People with panic disorder may have:")]
    with pytest.raises(InsufficientAnswerEvidence):
        EvidenceOnlyGenerator().generate(question, h, retrieval(question, h))


def test_general_comparison_uses_both_direct_treatment_findings():
    h = comparison_hits()
    payload = EvidenceOnlyGenerator().generate(QUESTION, h, retrieval(QUESTION, h))
    assert BACKGROUND not in payload.answer
    assert NON_FINDING not in payload.answer
    assert payload.cited_source_ids == ["S4", "S2"]
    assert f"{PSYCH} [S4]" in payload.answer
    assert f"{MED} [S2]" in payload.answer
    assert ClaimGroundingVerifier(HeuristicNliClassifier()).verify(payload, h).supported


def test_general_comparison_abstains_if_either_approach_is_missing():
    h = [hit(MED)]
    with pytest.raises(InsufficientAnswerEvidence):
        EvidenceOnlyGenerator().generate(QUESTION, h, retrieval(QUESTION, h))


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": json.dumps(self.payload)}}


def test_ollama_can_only_reorder_immutable_source_statements(monkeypatch):
    def fake_post(url, **kwargs):
        assert url.endswith("/api/chat")
        assert "ordered_ids" in str(kwargs["json"]["format"])
        return FakeResponse({"ordered_ids": ["S2", "S4"]})

    monkeypatch.setattr("care_anxrag.ollama_organizer.httpx.post", fake_post)
    h = comparison_hits()
    payload = OllamaEvidenceOrganizer("http://localhost:11434", "gemma3:4b").generate(
        QUESTION, h, retrieval(QUESTION, h)
    )

    assert payload.cited_source_ids == ["S2", "S4"]
    assert "Psychological approaches\n" in payload.answer
    assert "Medication approaches\n" in payload.answer
    assert f"{MED} [S2]" in payload.answer
    assert f"{PSYCH} [S4]" in payload.answer
    assert BACKGROUND not in payload.answer
    assert ClaimGroundingVerifier(HeuristicNliClassifier()).verify(payload, h).supported


@pytest.mark.parametrize("malicious", [
    {"ordered_ids": ["S2"], "answer": "Stop taking medications."},
    {"ordered_ids": ["S999", "S4"]},
    {"ordered_ids": ["S2", "S2"]},
    {"ordered_ids": ["S2"]},
])
def test_ollama_malicious_or_invalid_output_is_discarded(monkeypatch, malicious):
    monkeypatch.setattr(
        "care_anxrag.ollama_organizer.httpx.post",
        lambda *args, **kwargs: FakeResponse(malicious),
    )
    h = comparison_hits()
    result = OllamaEvidenceOrganizer("http://localhost:11434", "gemma3:4b").generate(
        QUESTION, h, retrieval(QUESTION, h)
    )
    assert "Stop taking medications" not in result.answer
    assert "S999" not in result.answer
    assert result.cited_source_ids == ["S4", "S2"]
    assert ClaimGroundingVerifier(HeuristicNliClassifier()).verify(result, h).supported


def test_ollama_outage_falls_back_to_source_only_output(monkeypatch):
    def offline(*args, **kwargs):
        raise httpx.ConnectError("synthetic disconnected Ollama")

    monkeypatch.setattr("care_anxrag.ollama_organizer.httpx.post", offline)
    h = comparison_hits()
    result = OllamaEvidenceOrganizer("http://localhost:11434", "gemma3:4b").generate(
        QUESTION, h, retrieval(QUESTION, h)
    )
    assert result.cited_source_ids == ["S4", "S2"]
    assert ClaimGroundingVerifier(HeuristicNliClassifier()).verify(result, h).supported


def test_outside_scope_remains_abstention(runtime):
    result = runtime.rag.answer("How do I repair a refrigerator compressor?")
    assert result.abstained
    assert result.abstention_reason == "outside_anxiety_scope"
    assert result.citations == []


def test_who_style_chunk_selects_both_treatment_facts():
    """Regression for observed live WHO retrieval: two facets in one chunk."""
    from care_anxrag.models import EvidenceLevel
    from care_anxrag.query import QueryAnalyzer
    from care_anxrag.answer_selection import select_answer_sentence

    statement_psych = (
        "Psychological interventions are essential treatments for anxiety disorders "
        "and refer primarily to talk therapy with professionals or supervised lay therapists."
    )
    statement_med = (
        "Antidepressant medications, such as selective serotonin reuptake inhibitors "
        "(SSRIs), can also be useful in treating adults with anxiety disorders."
    )
    who = hit(
        "There are several effective treatments for anxiety disorders. "
        + statement_psych + " " + statement_med,
        1,
        title="Anxiety disorders",
    )
    who.chunk.evidence_level = EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION
    question = QUESTION
    analysis = QueryAnalyzer().analyze(question)
    assert analysis.treatments == []
    assert select_answer_sentence(who, analysis, required_topic="psychological") == statement_psych
    assert select_answer_sentence(who, analysis, required_topic="medication") == statement_med

    payload = EvidenceOnlyGenerator().generate(
        question, [who], retrieval(question, [who])
    )
    assert statement_psych + " [S1]" in payload.answer
    assert statement_med + " [S1]" in payload.answer
    assert ClaimGroundingVerifier(HeuristicNliClassifier()).verify(
        payload, [who]
    ).supported


def test_acupuncture_study_objective_not_misclassified_as_result():
    objective = (
        "Evaluate the effectiveness of acupuncture and electroacupuncture in "
        "the treatment of anxiety to verify if people report reduced scores "
        "after five sessions."
    )
    rows = [hit(objective)]
    with pytest.raises(InsufficientAnswerEvidence):
        EvidenceOnlyGenerator().generate(QUESTION, rows, retrieval(QUESTION, rows))
