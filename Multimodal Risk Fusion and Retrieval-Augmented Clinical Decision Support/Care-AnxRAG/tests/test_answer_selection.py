"""Synthetic source fixtures for answer selection; not clinical evidence or gold labels."""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from care_anxrag.api import create_app
from care_anxrag.generation import EvidenceOnlyGenerator
from care_anxrag.grounding import ClaimGroundingVerifier
from care_anxrag.models import (
    ChunkRecord,
    DocumentStatus,
    EvidenceLevel,
    KnowledgeLayer,
    RetrievalResult,
    SearchHit,
)
from care_anxrag.nli import HeuristicNliClassifier
from care_anxrag.query import QueryAnalyzer
from care_anxrag.rag import CareAnxRag


QUESTION = "What does the evidence say about cognitive behavioural therapy for anxiety?"


def hit(text: str, index: int = 1, *, title: str = "Synthetic evidence") -> SearchHit:
    return SearchHit(
        chunk=ChunkRecord(
            chunk_id=f"chunk-{index}",
            document_id=f"doc-{index}",
            version_id=f"version-{index}",
            source_id=f"source-{index}",
            source_name=f"Synthetic source {index}",
            title=title,
            layer=KnowledgeLayer.CLINICAL_CORE,
            status=DocumentStatus.ACTIVE,
            section_path="evidence",
            section_heading="Evidence",
            ordinal=0,
            text=text,
            text_hash="synthetic",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            authority_score=0.9,
            evidence_level=EvidenceLevel.CLINICAL_GUIDELINE,
            evidence_score=0.9,
            metadata={"synthetic": True},
        ),
        care_score=0.9,
        relevance_score=0.9,
    )


def retrieval(question: str, hits: list[SearchHit], **overrides) -> RetrievalResult:
    return RetrievalResult(
        query_analysis=QueryAnalyzer().analyze(question),
        hits=hits,
        confidence=0.9,
        **overrides,
    )


class StaticRetriever:
    """Supply a fixed retrieval result to exercise real presentation and grounding."""

    def __init__(self, result: RetrievalResult):
        self.result = result

    def retrieve(self, question: str) -> RetrievalResult:
        return self.result


def rag_for(settings, result: RetrievalResult) -> CareAnxRag:
    return CareAnxRag(
        settings,
        StaticRetriever(result),
        EvidenceOnlyGenerator(),
        ClaimGroundingVerifier(HeuristicNliClassifier()),
    )


def test_findings_are_selected_instead_of_more_lexically_similar_objectives():
    objective = (
        "The purpose of this study was to investigate the effectiveness of cognitive "
        "behavioural therapy for anxiety."
    )
    finding = "CBT reduced anxiety symptoms compared with the control condition."
    hits = [hit(f"{objective} {finding}")]
    payload = EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))

    assert payload.answer == f"- {finding} [S1]"
    assert payload.cited_source_ids == ["S1"]
    assert ClaimGroundingVerifier(HeuristicNliClassifier()).verify(payload, hits).supported


@pytest.mark.parametrize("text", [
    "The aim was to determine whether CBT reduced anxiety symptoms.",
    "Patients with anxiety were randomly assigned to CBT or a control condition.",
    "Internet-based cognitive behavioral therapy programs are offered with varying therapist support.",
    "We hypothesize that CBT will improve anxiety symptoms.",
    "This study evaluated CBT effectiveness for anxiety using symptom scores.",
    "CBT effectiveness for anxiety was assessed using symptom questionnaires.",
])
def test_objectives_methods_and_delivery_background_are_not_effectiveness_answers(text):
    hits = [hit(text)]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))


def test_scans_remaining_context_and_keeps_original_citation_ids(settings):
    hits = [hit("The purpose was to study CBT for anxiety.", i) for i in range(1, 4)]
    finding = "CBT can reduce anxiety symptoms."
    hits.append(hit(finding, 4))
    result = rag_for(settings, retrieval(QUESTION, hits)).answer(QUESTION)

    assert not result.abstained
    assert result.answer == f"- {finding} [S4]"
    assert [(c.citation_id, c.chunk_id) for c in result.citations] == [("S4", "chunk-4")]


def test_duplicate_source_sentences_are_not_repeated():
    finding = "CBT can reduce anxiety symptoms."
    hits = [hit(finding, 1), hit(finding, 2)]
    payload = EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))

    assert payload.answer == f"- {finding} [S1]"
    assert payload.cited_source_ids == ["S1"]


@pytest.mark.parametrize("finding", [
    "CBT did not reduce anxiety symptoms compared with the control condition.",
    "The evidence for CBT effectiveness in anxiety remains uncertain.",
    "CBT had no effect on anxiety symptoms.",
    "CBT reduced anxiety scores by 2.5 points, but the estimate was imprecise.",
])
def test_negative_and_uncertain_findings_remain_verbatim(finding):
    hits = [hit(finding)]
    payload = EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))

    assert payload.answer == f"- {finding} [S1]"


def test_population_qualification_is_kept_in_the_selected_sentence():
    finding = "CBT reduced anxiety symptoms in adolescents in this pilot trial."
    hits = [hit(finding)]
    payload = EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))

    assert finding in payload.answer


@pytest.mark.parametrize("title,text", [
    ("CBT for adolescents", "CBT reduced anxiety symptoms."),
    ("Synthetic evidence", "Participants were adolescents. CBT reduced anxiety symptoms."),
])
def test_population_scope_cannot_be_lost_when_extracting_a_finding(title, text):
    hits = [hit(text, title=title)]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))


def test_adolescent_finding_does_not_answer_an_explicit_adult_question(settings):
    question = "Does CBT reduce anxiety in adults?"
    hits = [hit("CBT reduced anxiety symptoms in adolescents in this pilot trial.")]
    result = rag_for(settings, retrieval(question, hits)).answer(question)

    assert result.abstained
    assert result.abstention_reason == "insufficient_answer_evidence"
    assert result.citations == []


def test_unrelated_finding_cannot_borrow_treatment_context_from_an_objective():
    hits = [hit(
        "The aim was to study CBT for anxiety. Medication reduced anxiety symptoms."
    )]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))


def test_population_scope_from_ingestion_facets_is_preserved():
    evidence = hit("CBT reduced anxiety symptoms.")
    evidence.chunk.metadata["clinical_facets"] = {"populations": ["children_and_adolescents"]}
    hits = [evidence]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))


def test_unrecognized_population_metadata_does_not_crash():
    finding = "CBT can reduce anxiety symptoms."
    evidence = hit(finding)
    evidence.chunk.metadata["clinical_facets"] = {"populations": [{"unknown": "value"}]}
    hits = [evidence]
    payload = EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))

    assert payload.answer == f"- {finding} [S1]"


def test_findings_do_not_borrow_the_requested_outcome_from_an_objective(settings):
    question = "Does CBT improve quality of life for anxiety?"
    hits = [hit(
        "The aim was to assess CBT effects on quality of life for anxiety. "
        "CBT reduced anxiety symptoms."
    )]
    result = rag_for(settings, retrieval(question, hits)).answer(question)

    assert result.abstained
    assert result.abstention_reason == "insufficient_answer_evidence"


def test_general_information_queries_still_return_source_definitions():
    question = "What are anxiety disorders?"
    definition = "Anxiety disorders involve persistent fear or worry."
    hits = [hit(definition)]
    payload = EvidenceOnlyGenerator().generate(question, hits, retrieval(question, hits))

    assert payload.answer == f"- {definition} [S1]"


def test_descriptive_treatment_question_can_return_a_discussed_intervention():
    question = "What evidence-based psychological intervention is discussed for panic disorder?"
    discussion = "The fixture discusses cognitive behavioural therapy for panic disorder."
    hits = [hit(discussion)]
    payload = EvidenceOnlyGenerator().generate(question, hits, retrieval(question, hits))

    assert payload.answer == f"- {discussion} [S1]"


def test_helpful_question_cannot_be_answered_with_delivery_background():
    question = "Is CBT helpful for anxiety?"
    hits = [hit("Internet-based CBT programs for anxiety vary in therapist support.")]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(question, hits, retrieval(question, hits))


@pytest.mark.parametrize("finding", [
    "In this trial CBT did not reduce anxiety symptoms.",
    "CBT was not effective for anxiety in this trial.",
    "CBT effectiveness for anxiety remains uncertain.",
])
def test_actual_negative_result_is_preferred_over_generic_background(finding):
    background = "CBT is an effective treatment for anxiety."
    hits = [hit(f"{background} {finding}")]
    payload = EvidenceOnlyGenerator().generate(QUESTION, hits, retrieval(QUESTION, hits))

    assert payload.answer == f"- {finding} [S1]"


@pytest.mark.parametrize("finding", [
    "CBT reduced anxiety symptoms vs. placebo, but the estimate was imprecise.",
    "CBT did not improve anxiety outcomes, e.g. symptom severity, in this trial.",
])
def test_abbreviations_do_not_strip_comparators_or_qualifiers(settings, finding):
    hits = [hit(finding)]
    result = rag_for(settings, retrieval(QUESTION, hits)).answer(QUESTION)

    assert not result.abstained
    assert result.answer == f"- {finding} [S1]"
    assert result.citations[0].chunk_id == "chunk-1"


def test_objectives_only_returns_specific_api_abstention(runtime):
    hits = [hit("The purpose was to study CBT for anxiety.")]
    runtime.rag = rag_for(runtime.settings, retrieval(QUESTION, hits))
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.post("/v1/ask", json={"question": QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is True
    assert body["abstention_reason"] == "insufficient_answer_evidence"
    assert body["citations"] == []
    assert "findings" in body["answer"]
    assert "format" not in body["answer"]


def test_existing_unresolved_conflict_abstention_is_preserved(settings):
    hits = [hit("CBT can reduce anxiety symptoms.")]
    result = rag_for(settings, retrieval(
        QUESTION, hits, should_abstain=True,
        abstention_reason="unresolved_high_confidence_evidence_conflict",
    )).answer(QUESTION)

    assert result.abstained
    assert result.abstention_reason == "unresolved_high_confidence_evidence_conflict"
    assert result.citations == []


# ---------------------------------------------------------------------------
# Viva regressions: question-directed factuality vs merely topical citations.
# Synthetic engineering examples; no clinician-reviewed labels or outcomes.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question,irrelevant", [
    (
        "What treatments are supported for generalized anxiety disorder?",
        "Generalized anxiety disorder (GAD) is a prevalent anxiety disorder "
        "characterized by uncontrollable worry, muscle tension, and irritability.",
    ),
    (
        "What treatments are supported for generalized anxiety disorder?",
        "Generalized anxiety disorder (GAD) has important implications "
        "for patients and healthcare resources.",
    ),
    (
        "How is panic disorder diagnosed?",
        "Panic disorder occurs in 1-3% of adolescents and may persist.",
    ),
    (
        "How is panic disorder diagnosed?",
        "Cognitive behavioral therapy (CBT) is efficacious for panic disorder.",
    ),
    (
        "What evidence contradicts an intervention recommendation for GAD?",
        "However, there is a lack of research exploring online check-ins for GAD.",
    ),
    (
        "What evidence contradicts an intervention recommendation for GAD?",
        "CBT is an effective treatment for generalized anxiety disorder.",
    ),
])
def test_clinician_queries_do_not_return_topical_non_answers(question, irrelevant):
    rows = [hit(irrelevant)]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(question, rows, retrieval(question, rows))


@pytest.mark.parametrize("question,statement", [
    (
        "What treatments are supported for generalized anxiety disorder?",
        "CBT was effective for adults with generalized anxiety disorder.",
    ),
    (
        "What does the evidence say about CBT for social anxiety disorder?",
        "CBT reduced symptom severity in adults with social anxiety disorder.",
    ),
    (
        "Compare exposure therapy and CBT for panic disorder.",
        "CBT and exposure therapy showed no significant difference for panic disorder.",
    ),
    (
        "How is panic disorder diagnosed?",
        "Panic disorder diagnosis requires assessment of recurrent unexpected panic attacks.",
    ),
])
def test_clinician_questions_can_select_direct_source_sentences(question, statement):
    rows = [hit(statement)]
    payload = EvidenceOnlyGenerator().generate(question, rows, retrieval(question, rows))
    assert payload.answer == f"- {statement} [S1]"
    assert ClaimGroundingVerifier(HeuristicNliClassifier()).verify(payload, rows).supported


def test_single_negative_study_does_not_establish_a_contradiction():
    question = "What evidence contradicts an intervention recommendation for GAD?"
    rows = [hit("CBT did not reduce anxiety symptoms in adults with generalized anxiety disorder.")]
    # A sole negative result does not establish that two studies conflict.
    # Comparison requires a verified, independently grounded NLI pair.
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(question, rows, retrieval(question, rows))


def test_comparison_requires_comparative_outcome_not_two_treatment_mentions():
    question = "Compare exposure therapy and CBT for panic disorder."
    rows = [hit("CBT and exposure therapy are treatments for panic disorder.")]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(question, rows, retrieval(question, rows))


def test_recent_review_discovery_requires_relevant_review_title_and_publication_date():
    from datetime import UTC, datetime
    from care_anxrag.models import EvidenceLevel

    question = "What recent systematic reviews address anxiety-disorder treatment?"
    biomarker = hit(
        "This systematic review examined associations of inflammatory biomarkers "
        "and anxiety disorders."
    )
    biomarker.chunk.evidence_level = EvidenceLevel.SYSTEMATIC_REVIEW
    biomarker.chunk.published_at = datetime(2026, 7, 9, tzinfo=UTC)
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(
            question, [biomarker], retrieval(question, [biomarker])
        )

    title = "CBT treatment for anxiety disorders: a systematic review."
    therapy = hit(
        "CBT reduced symptom severity for adults with anxiety disorders "
        "in the included studies.",
        title=title,
    )
    therapy.chunk.url = "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    therapy.chunk.evidence_level = EvidenceLevel.SYSTEMATIC_REVIEW
    therapy.chunk.published_at = datetime(2026, 7, 9, tzinfo=UTC)
    generated = EvidenceOnlyGenerator().generate(
        question, [therapy], retrieval(question, [therapy])
    )
    assert generated.answer == f"- {title} [S1]"
    assert generated.cited_source_ids == ["S1"]
    # This is a SOURCE TITLE, not a claim that CBT works.
    assert not ClaimGroundingVerifier(HeuristicNliClassifier()).verify(
        generated, [therapy]
    ).supported
    assert ClaimGroundingVerifier(HeuristicNliClassifier()).verify(
        generated, [therapy], allow_metadata_titles=True
    ).supported


def test_authoritative_medication_query_excludes_unrelated_secondary_disease_reviews():
    from care_anxrag.models import EvidenceLevel

    question = "What do authoritative sources say about medication for anxiety?"
    copd = hit("SSRIs may help anxiety symptoms in chronic obstructive pulmonary disease.")
    copd.chunk.evidence_level = EvidenceLevel.META_ANALYSIS
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(question, [copd], retrieval(question, [copd]))

    who = hit(
        "Antidepressant medications, such as selective serotonin reuptake "
        "inhibitors (SSRIs), can also be useful in treating adults with anxiety disorders."
    )
    who.chunk.evidence_level = EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION
    generated = EvidenceOnlyGenerator().generate(question, [who], retrieval(question, [who]))
    assert generated.cited_source_ids == ["S1"]




def test_nimh_medication_followup_without_antecedent_is_not_an_answer():
    question = "What do authoritative sources say about medication for anxiety?"
    dangling = (
        "These medications can be very effective in rapidly decreasing anxiety, "
        "but some people build up a tolerance to them and need increasing doses "
        "to get the same effect."
    )
    h = hit(dangling)
    h.chunk.evidence_level = EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(question, [h], retrieval(question, [h]))


def test_nimh_named_medication_sentence_selected_instead_of_dangling_followup():
    question = "What do authoritative sources say about medication for anxiety?"
    named = (
        "Antidepressant medications, such as selective serotonin reuptake "
        "inhibitors (SSRIs), can also be useful in treating adults with anxiety disorders."
    )
    dangling = (
        "These medications can be very effective in rapidly decreasing anxiety, "
        "but some people build up a tolerance to them and need increasing doses "
        "to get the same effect."
    )
    h = hit(named + " " + dangling)
    h.chunk.evidence_level = EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION
    result = EvidenceOnlyGenerator().generate(question, [h], retrieval(question, [h]))
    assert named in result.answer
    assert dangling not in result.answer
    assert ClaimGroundingVerifier(HeuristicNliClassifier()).verify(result, [h]).supported

def test_gad_at_sentence_end_or_punctuation_is_detected_as_subtype():
    assert "generalized_anxiety_disorder" in QueryAnalyzer().analyze(
        "What evidence contradicts treatment for GAD?"
    ).anxiety_subtypes


def test_seventh_chunk_with_relevant_treatment_result_becomes_first():
    from care_anxrag.retrieval import CareRetriever
    question = "What treatments are supported for generalized anxiety disorder?"
    background = [
        hit("Generalized anxiety disorder involves excessive worry and distress.", i)
        for i in range(1, 7)
    ]
    supported = hit(
        "CBT reduced anxiety symptoms in adults with generalized anxiety disorder.", 7
    )
    analysis = QueryAnalyzer().analyze(question)
    reordered = CareRetriever._rank_by_answerability(background + [supported], analysis)
    assert reordered[0].chunk.chunk_id == supported.chunk.chunk_id
    six = CareRetriever._diversify(reordered, limit=6, max_per_document=2)
    generated = EvidenceOnlyGenerator().generate(question, six, retrieval(question, six))
    assert generated.answer == (
        "- CBT reduced anxiety symptoms in adults with generalized anxiety disorder. [S1]"
    )
    assert ClaimGroundingVerifier(HeuristicNliClassifier()).verify(generated, six).supported


def test_answerability_ranking_rejects_research_gap_as_contradiction():
    from care_anxrag.retrieval import CareRetriever
    question = "What evidence contradicts an intervention recommendation for GAD?"
    gap = hit(
        "There is a lack of research exploring online check-ins for GAD.", 1
    )
    finding = hit(
        "CBT did not reduce anxiety symptoms in adults with generalized anxiety disorder.", 2
    )
    reordered = CareRetriever._rank_by_answerability(
        [gap, finding], QueryAnalyzer().analyze(question)
    )
    assert [x.chunk.chunk_id for x in reordered] == ["chunk-2", "chunk-1"]


def test_answerability_ranking_preserves_abstention_if_no_direct_evidence():
    from care_anxrag.retrieval import CareRetriever
    question = "How is panic disorder diagnosed?"
    background = [
        hit("Panic disorder causes significant distress.", 1),
        hit("CBT is an effective treatment for panic disorder.", 2),
    ]
    reordered = CareRetriever._rank_by_answerability(
        background, QueryAnalyzer().analyze(question)
    )
    assert [x.chunk.chunk_id for x in reordered] == ["chunk-1", "chunk-2"]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(question, reordered, retrieval(question, reordered))


# Diagnostics from 2026-10-16 real corpus: these statements previously
# passed heuristic eligibility despite answering none of the clinician tasks.
@pytest.mark.parametrize("question,title,nonanswer", [
    (
        "What treatments are supported for generalized anxiety disorder?",
        "Comparing the Efficacy of Electronically Delivered CBT for GAD",
        "Comparing the Effectiveness of Electronically Delivered Therapy "
        "(e-CBT) to Weekly Online Mental Health Check-ins for Generalized "
        "Anxiety Disorder-A Randomized Controlled Trial.",
    ),
    (
        "What recent systematic reviews address anxiety-disorder treatment?",
        "EVIDENCE-BASED PHARMACOTHERAPY FOR ANXIETY DISORDERS: A SYSTEMATIC REVIEW",
        "To evaluate the efficacy, safety, and tolerability of pharmacological "
        "interventions for anxiety outcomes in adults with post-traumatic "
        "stress disorder (PTSD), including comorbid anxiety disorders.",
    ),
    (
        "How is panic disorder diagnosed?",
        "A feasibility trial for panic disorder in adolescents",
        "Thirty-four young people meeting diagnostic criteria for panic disorder "
        "were randomly allocated to cognitive therapy or CBT.",
    ),
])
def test_real_corpus_title_objective_and_trial_allocation_are_not_answers(
    question, title, nonanswer,
):
    rows = [hit(nonanswer, title=title)]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(question, rows, retrieval(question, rows))


def test_matching_source_title_alone_is_not_a_clinical_finding():
    question = "What treatments are supported for generalized anxiety disorder?"
    title = "CBT effectiveness for generalized anxiety disorder."
    rows = [hit(title, title=title)]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(question, rows, retrieval(question, rows))


def test_real_gad_review_finding_is_still_eligible():
    question = "What treatments are supported for generalized anxiety disorder?"
    finding = (
        "Our review confirms the usefulness of long-term pharmacological "
        "treatments for PD and GAD and suggests that they can provide further "
        "improvement over that obtained during short-term therapy."
    )
    rows = [hit(finding)]
    payload = EvidenceOnlyGenerator().generate(question, rows, retrieval(question, rows))
    assert payload.answer == f"- {finding} [S1]"
    assert ClaimGroundingVerifier(HeuristicNliClassifier()).verify(payload, rows).supported


# Regression fixtures copied from the staged NIMH educational source text.
# These are source-derived engineering tests, not clinician-adjudicated labels.
def test_nimh_social_anxiety_generic_cbt_guidance_can_answer_p02():
    question = "What does the evidence say about CBT for social anxiety disorder?"
    finding = (
        "Cognitive behavioral therapy (CBT), a research-supported type of "
        "psychotherapy, is commonly used to treat social anxiety disorder."
    )
    rows = [hit(finding, title="Social Anxiety Disorder: What You Need to Know")]
    rows[0].chunk.source_id = "nimh_social_anxiety"
    rows[0].chunk.evidence_level = EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION
    rows[0].chunk.metadata["clinical_facets"] = {
        "populations": ["adults", "children_and_adolescents"],
    }
    result = EvidenceOnlyGenerator().generate(question, rows, retrieval(question, rows))
    assert result.answer == f"- {finding} [S1]"
    assert ClaimGroundingVerifier(HeuristicNliClassifier()).verify(result, rows).supported


def test_document_wide_mixed_population_exception_never_applies_to_trials():
    question = "What does the evidence say about CBT for social anxiety disorder?"
    finding = (
        "Cognitive behavioral therapy (CBT), a research-supported type of "
        "psychotherapy, is commonly used to treat social anxiety disorder."
    )
    rows = [hit(finding, title="Mixed-age trial for social anxiety disorder")]
    rows[0].chunk.evidence_level = EvidenceLevel.RANDOMIZED_CONTROLLED_TRIAL
    rows[0].chunk.metadata["clinical_facets"] = {
        "populations": ["adults", "children_and_adolescents"],
    }
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(question, rows, retrieval(question, rows))


def test_nimh_general_guidance_does_not_answer_explicit_adult_only_question():
    question = "What evidence supports CBT for adults with social anxiety disorder?"
    finding = (
        "Cognitive behavioral therapy (CBT), a research-supported type of "
        "psychotherapy, is commonly used to treat social anxiety disorder."
    )
    rows = [hit(finding, title="Social Anxiety Disorder: What You Need to Know")]
    rows[0].chunk.evidence_level = EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION
    rows[0].chunk.metadata["clinical_facets"] = {
        "populations": ["adults", "children_and_adolescents"],
    }
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(question, rows, retrieval(question, rows))


def test_nimh_panic_diagnosis_source_statement_can_answer_p07():
    question = "How is panic disorder diagnosed?"
    diagnosis = (
        "Panic disorder may be diagnosed when a person experiences recurrent "
        "and unexpected panic attacks, along with ongoing worry or behavioral "
        "changes due to the fear of future attacks, often interfering with daily life."
    )
    rows = [hit(diagnosis, title="Panic Disorder: What You Need to Know")]
    rows[0].chunk.source_id = "nimh_panic_disorder"
    rows[0].chunk.evidence_level = EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION
    result = EvidenceOnlyGenerator().generate(question, rows, retrieval(question, rows))
    assert result.answer == f"- {diagnosis} [S1]"
    assert ClaimGroundingVerifier(HeuristicNliClassifier()).verify(result, rows).supported


def test_nimh_panic_diagnosis_does_not_accept_participant_eligibility():
    question = "How is panic disorder diagnosed?"
    rows = [hit(
        "Participants with panic disorder were enrolled after meeting "
        "diagnostic criteria.",
        title="Panic disorder feasibility trial",
    )]
    with pytest.raises(ValueError, match="insufficient_answer_evidence"):
        EvidenceOnlyGenerator().generate(question, rows, retrieval(question, rows))


def test_prefer_self_contained_panic_diagnosis_over_isolated_criterion():
    question = "How is panic disorder diagnosed?"
    criterion = (
        "To be diagnosed with panic disorder, a person must experience "
        "recurrent, unexpected panic attacks."
    )
    complete = (
        "Panic disorder may be diagnosed when a person experiences recurrent "
        "and unexpected panic attacks, along with ongoing worry or behavioral "
        "changes due to the fear of future attacks, often interfering with daily life."
    )
    rows = [hit(criterion + " " + complete, title="Panic Disorder: What You Need to Know")]
    rows[0].chunk.evidence_level = EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION
    result = EvidenceOnlyGenerator().generate(question, rows, retrieval(question, rows))
    assert result.answer == f"- {complete} [S1]"



def test_nimh_cbt_two_chunks_one_publication_prefers_explicitly_supported_evidence():
    question = "What does the evidence say about CBT for social anxiety disorder?"
    generic = (
        "CBT delivered in a group therapy format also can offer unique "
        "benefits for social anxiety disorder."
    )
    supported = (
        "Cognitive behavioral therapy (CBT), a research-supported type of "
        "psychotherapy, is commonly used to treat social anxiety disorder."
    )
    hits = [hit(generic, 1), hit(supported, 2)]
    for row in hits:
        row.chunk.document_id = "nimh-social-anxiety-single-publication"
        row.chunk.evidence_level = EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION
        row.chunk.metadata["clinical_facets"] = {
            "populations": ["children_and_adolescents", "adults"],
        }
    payload = EvidenceOnlyGenerator().generate(
        question, hits, retrieval(question, hits)
    )
    assert payload.answer == f"- {supported} [S2]"
    assert payload.cited_source_ids == ["S2"]
    assert ClaimGroundingVerifier(HeuristicNliClassifier()).verify(
        payload, hits
    ).supported


def test_nimh_panic_two_chunks_one_publication_prefers_complete_diagnostic_context(
    settings,
):
    question = "How is panic disorder diagnosed?"
    incomplete = (
        "To be diagnosed with panic disorder, a person must experience "
        "recurrent, unexpected panic attacks."
    )
    complete = (
        "Panic disorder may be diagnosed when a person experiences recurrent "
        "and unexpected panic attacks, along with ongoing worry or behavioral "
        "changes due to the fear of future attacks, often interfering with daily life."
    )
    hits = [hit(incomplete, 1), hit(complete, 2)]
    for row in hits:
        row.chunk.document_id = "nimh-panic-single-publication"
        row.chunk.evidence_level = EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION
    result = rag_for(settings, retrieval(question, hits)).answer(question)
    assert not result.abstained
    assert result.answer.startswith(f"- {complete} [S2]")
    assert incomplete not in result.answer
    assert "medication recommendation" not in result.answer
    assert "not a diagnosis of an individual" in result.answer
    assert [c.citation_id for c in result.citations] == ["S2"]
    assert [c.chunk_id for c in result.citations] == ["chunk-2"]


def test_two_independent_documents_with_distinct_findings_remain_two_citations():
    question = "What does the evidence say about CBT for social anxiety disorder?"
    hits = [
        hit(
            "Cognitive behavioral therapy (CBT), a research-supported "
            "psychotherapy, is commonly used to treat social anxiety disorder.",
            1,
        ),
        hit(
            "CBT reduced social anxiety disorder symptoms compared with "
            "a waiting-list control in the studied adult population.",
            2,
        ),
    ]
    payload = EvidenceOnlyGenerator().generate(
        question, hits, retrieval(question, hits)
    )
    assert len(payload.cited_source_ids) == 2
    assert payload.cited_source_ids == ["S1", "S2"]
    assert ClaimGroundingVerifier(HeuristicNliClassifier()).verify(
        payload, hits
    ).supported


def test_medication_information_has_only_medication_disclaimer(settings):
    question = "What do authoritative sources say about medication for anxiety?"
    rows = [
        hit(
            "Antidepressant medications, such as selective serotonin "
            "reuptake inhibitors (SSRIs), can also be useful in treating "
            "adults with anxiety disorders."
        )
    ]
    rows[0].chunk.evidence_level = EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION
    response = rag_for(settings, retrieval(question, rows)).answer(question)
    assert not response.abstained
    assert "not an individualized medication recommendation" in response.answer
    assert "not a diagnosis of an individual" not in response.answer
