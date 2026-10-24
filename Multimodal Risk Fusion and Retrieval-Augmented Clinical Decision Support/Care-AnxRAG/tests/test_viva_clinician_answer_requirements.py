"""Engineering regressions derived from clinician-style viva failures.

Synthetic source snippets are NOT actual research evidence or clinical gold labels.
"""
from datetime import UTC, datetime, timedelta

import pytest

from care_anxrag.answer_selection import InsufficientAnswerEvidence
from care_anxrag.generation import EvidenceOnlyGenerator
from care_anxrag.models import (
    ChunkRecord,
    DocumentStatus,
    EvidenceLevel,
    EvidenceRelation,
    KnowledgeLayer,
    RelationLabel,
    RetrievalResult,
    SearchHit,
)
from care_anxrag.query import QueryAnalyzer


def hit(text, *, index=1, title="Anxiety treatment study", level=EvidenceLevel.RANDOMIZED_CONTROLLED_TRIAL, days_ago=120):
    return SearchHit(
        chunk=ChunkRecord(
            chunk_id=f"chunk-{index}",
            document_id=f"doc-{index}",
            version_id=f"version-{index}",
            source_id=f"source-{index}",
            source_name="Synthetic test only",
            title=title,
            layer=KnowledgeLayer.RESEARCH_FRONTIER,
            status=DocumentStatus.ACTIVE,
            section_path="abstract/results",
            section_heading="Results",
            ordinal=0,
            text=text,
            text_hash=f"hash-{index}",
            published_at=datetime.now(UTC)-timedelta(days=days_ago),
            retrieved_at=datetime.now(UTC),
            authority_score=0.8,
            evidence_level=level,
            evidence_score=0.8,
        ),
        care_score=0.8, relevance_score=0.8,
    )


def generate(question, hits, relations=()):
    retrieval = RetrievalResult(
        query_analysis=QueryAnalyzer().analyze(question),
        hits=hits,
        relations=list(relations),
        confidence=0.8,
    )
    return EvidenceOnlyGenerator().generate(question, hits, retrieval)


def test_p01_treatments_not_answered_with_gad_background():
    question = "What treatments are supported for generalized anxiety disorder?"
    hits = [
        hit("Generalized anxiety disorder (GAD) is a prevalent condition.", index=1),
        hit("Generalized anxiety disorder is associated with trouble sleeping.", index=2),
    ]
    with pytest.raises(InsufficientAnswerEvidence):
        generate(question, hits)


def test_p01_can_return_direct_supported_treatment_finding():
    question = "What treatments are supported for generalized anxiety disorder?"
    finding = "CBT reduced symptoms among people with generalized anxiety disorder."
    result = generate(question, [hit(finding)])
    assert result.answer == f"- {finding} [S1]"


def test_p01_broad_gad_treatments_include_both_evidence_classes():
    # Medication is ranked above psychotherapy, as in the live Mac run.
    question = "What treatments are supported for generalized anxiety disorder?"
    medication = (
        "Long-term pharmacological treatments for GAD were useful "
        "in improving anxiety symptoms."
    )
    psychological = (
        "CBT reduced anxiety symptoms in adults with generalized anxiety disorder."
    )
    rows = [
        hit(medication, index=1, title="GAD medication evidence"),
        hit("GAD is a common anxiety disorder.", index=2),
        hit(psychological, index=3, title="GAD CBT treatment trial"),
    ]
    result = generate(question, rows)
    assert result.answer == (
        f"- {psychological} [S3]\\n- {medication} [S1]"
    ).replace("\\n", "\n")
    assert result.cited_source_ids == ["S3", "S1"]


def test_p01_does_not_relabel_other_disorder_evidence_as_gad():
    question = "What treatments are supported for generalized anxiety disorder?"
    medication = "SSRIs can be useful in treating adults with generalized anxiety disorder."
    other_disorder = (
        "CBT reduced anxiety symptoms in adults with social anxiety disorder."
    )
    rows = [hit(medication, index=1), hit(other_disorder, index=2)]
    # A single eligible GAD class remains a partial answer; the other
    # disorder's CBT evidence must not be brought into the GAD answer.
    result = generate(question, rows)
    assert result.cited_source_ids == ["S1"]
    assert other_disorder not in result.answer


def test_p02_social_anxiety_cbt_evidence_when_explicit_sentence_exists():
    question = "What does the evidence say about CBT for social anxiety disorder?"
    finding = "CBT reduced symptoms of social anxiety disorder in adults."
    result = generate(question, [hit(finding)])
    assert result.answer == f"- {finding} [S1]"


def test_p03_cannot_construct_comparison_from_two_separate_single_arm_studies():
    question = "Compare exposure therapy and CBT for panic disorder."
    hits = [
        hit("CBT reduced symptoms in panic disorder.", index=1),
        hit("Exposure therapy reduced symptoms in panic disorder.", index=2),
    ]
    with pytest.raises(InsufficientAnswerEvidence):
        generate(question, hits)


def test_p03_accepts_explicit_head_to_head_finding():
    question = "Compare exposure therapy and CBT for panic disorder."
    finding = "CBT was more effective than exposure therapy for panic disorder."
    result = generate(question, [hit(finding)])
    assert result.answer == f"- {finding} [S1]"


def test_p04_recent_systematic_reviews_exclude_old_meta_and_nontreatment_background():
    question = "What recent systematic reviews address anxiety-disorder treatment?"
    hits = [
        hit(
            "This systematic review describes pediatric sleep interventions and anxiety outcomes.",
            level=EvidenceLevel.SYSTEMATIC_REVIEW, index=1,
        ),
        hit(
            "CBT reduced anxiety symptoms in patients with anxiety disorders.",
            level=EvidenceLevel.META_ANALYSIS, days_ago=1900, index=2,
        ),
    ]
    with pytest.raises(InsufficientAnswerEvidence):
        generate(question, hits)


def test_p04_recent_treatment_review_can_answer_with_direct_result():
    question = "What recent systematic reviews address anxiety-disorder treatment?"
    finding = "This systematic review found that CBT reduced anxiety symptoms in adults."
    title = "CBT for anxiety disorders: a systematic review."
    evidence = hit(
        finding, level=EvidenceLevel.SYSTEMATIC_REVIEW, title=title
    )
    evidence.chunk.url = "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    result = generate(question, [evidence])
    # "What recent systematic reviews address ..." is a bibliographic
    # discovery question. Return the source title, never infer effectiveness
    # from its bibliographic metadata.
    assert result.answer == f"- {title} [S1]"
    assert result.uncertainty is not None


def test_p05_authoritative_medication_excludes_off_target_and_methods():
    question = "What do authoritative sources say about medication for anxiety?"
    relevant = (
        "Antidepressant medications, such as selective serotonin reuptake "
        "inhibitors (SSRIs), can also be useful in treating adults with anxiety disorders."
    )
    hits = [
        hit(
            relevant, index=1, level=EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION,
            title="WHO Anxiety disorders",
        ),
        hit(
            "Three subclasses of anxiety medications were used including SSRIs.",
            index=2, level=EvidenceLevel.META_ANALYSIS,
            title="Chronic obstructive pulmonary disease",
        ),
    ]
    result = generate(question, hits)
    assert result.answer == f"- {relevant} [S1]"
    assert result.cited_source_ids == ["S1"]


def test_p06_absence_of_actual_conflict_cannot_invent_disagreement():
    question = "What evidence contradicts an intervention recommendation for GAD?"
    hits = [
        hit("CBT is an effective treatment for GAD.", index=1),
        hit("There is a lack of research exploring online check-ins for GAD.", index=2),
    ]
    with pytest.raises(InsufficientAnswerEvidence):
        generate(question, hits)


def test_p06_explicit_conflicting_pair_can_be_presented_verbatim():
    question = "What evidence contradicts CBT treatment for GAD?"
    first = hit("CBT is an effective treatment for GAD.", index=1)
    second = hit("CBT did not reduce symptoms in GAD.", index=2)
    relations = [EvidenceRelation(
        left_chunk_id=first.chunk.chunk_id, right_chunk_id=second.chunk.chunk_id,
        label=RelationLabel.CONTRADICTION, confidence=0.94,
    )]
    result = generate(question, [first, second], relations)
    assert result.cited_source_ids == ["S1", "S2"]


def test_p07_diagnosis_query_rejects_prevalence_and_therapy():
    question = "How is panic disorder diagnosed?"
    hits = [
        hit("Panic disorder occurs in 1-3% of adolescents.", index=1),
        hit("CBT is efficacious for panic disorder with agoraphobia.", index=2),
        hit("Panic disorder is an anxiety disorder marked by severe fear of panic attacks.", index=3),
    ]
    with pytest.raises(InsufficientAnswerEvidence):
        generate(question, hits)


def test_p07_can_answer_with_direct_diagnostic_criteria_sentence():
    question = "How is panic disorder diagnosed?"
    finding = "Diagnostic criteria for panic disorder include recurrent unexpected panic attacks."
    result = generate(question, [hit(finding)])
    assert result.answer == f"- {finding} [S1]"



# Second 10-case real-corpus regressions (2026-10-19).
# Synthetic replicas of the observed answer sentences; NOT clinical validation.
def test_p10_prevalence_without_treatment_evidence_is_not_an_answer():
    q = "What evidence supports psychotherapy for panic disorder?"
    assert QueryAnalyzer().analyze(q).intent.value == "treatment"
    background = (
        "Panic disorder occurs in 1-3% of adolescents and without treatment "
        "often has a chronic course."
    )
    with pytest.raises(InsufficientAnswerEvidence):
        generate(q, [hit(background)])


def test_p10_keeps_direct_psychotherapy_evidence():
    q = "What evidence supports psychotherapy for panic disorder?"
    evidence = (
        "Cognitive behavioral therapy (CBT), a research-supported type "
        "of psychotherapy, is commonly used to treat panic disorder."
    )
    result = generate(q, [hit(evidence)])
    assert result.answer == f"- {evidence} [S1]"


def test_p11_general_cbt_is_not_internet_delivered_cbt_evidence():
    q = "What research supports internet-delivered CBT for social anxiety disorder?"
    general_cbt = (
        "Cognitive behavioral therapy (CBT), a research-supported type of "
        "psychotherapy, is commonly used to treat social anxiety disorder."
    )
    with pytest.raises(InsufficientAnswerEvidence):
        generate(q, [hit(general_cbt)])


def test_p11_can_select_direct_internet_cbt_finding():
    q = "What research supports internet-delivered CBT for social anxiety disorder?"
    evidence = (
        "Internet-delivered CBT reduced symptoms of social anxiety disorder "
        "in adults in this randomized trial."
    )
    result = generate(q, [hit(evidence)])
    assert result.answer == f"- {evidence} [S1]"


def test_p12_acute_benzodiazepine_effect_is_not_long_term_evidence():
    q = "What is known about long-term pharmacological treatment for panic disorder?"
    acute = (
        "Benzodiazepines, which are anti-anxiety sedative medications, "
        "can be very effective in rapidly decreasing panic attack symptoms."
    )
    with pytest.raises(InsufficientAnswerEvidence):
        generate(q, [hit(acute)])


def test_p12_can_select_explicit_long_term_panic_finding():
    q = "What is known about long-term pharmacological treatment for panic disorder?"
    evidence = (
        "Long-term pharmacological treatment improved anxiety symptoms "
        "in adults with panic disorder."
    )
    result = generate(q, [hit(evidence)])
    assert result.answer == f"- {evidence} [S1]"


def test_p08_digital_cbt_question_does_not_include_generic_cbt():
    q = "What evidence supports digital CBT for generalized anxiety disorder?"
    generic = (
        "Cognitive behavioural therapy (CBT) is an effective treatment for GAD."
    )
    digital = (
        "Digital CBT reduced anxiety symptoms among adults with "
        "generalized anxiety disorder."
    )
    rows = [hit(generic, index=1), hit(digital, index=2)]
    result = generate(q, rows)
    assert result.cited_source_ids == ["S2"]
    assert result.answer == f"- {digital} [S2]"


# P12 real-corpus regression: the long-term review explicitly defines
# "panic disorder (PD)" and later states its result using "PD".
# Preserve the cited sentence unchanged; only resolve the source's acronym.
def test_p12_source_defined_pd_abbreviation_preserves_original_evidence_sentence():
    q = "What is known about long-term pharmacological treatment for panic disorder?"
    definition = (
        "This systematic review studied panic disorder (PD) and "
        "generalized anxiety disorder (GAD)."
    )
    finding = (
        "Our review confirms the usefulness of long-term pharmacological "
        "treatments for PD and GAD and suggests that they can provide further "
        "improvement over that obtained during short-term therapy."
    )
    result = generate(q, [hit(f"{definition} {finding}")])
    assert result.answer == f"- {finding} [S1]"
    assert result.cited_source_ids == ["S1"]


@pytest.mark.parametrize("intro", [
    "This study assessed depressive symptoms.",
    "This review studied Parkinson's disease (PD).",
    "PD patients were included in the review.",
])
def test_p12_undefined_or_other_pd_abbreviation_remains_rejected(intro):
    q = "What is known about long-term pharmacological treatment for panic disorder?"
    finding = (
        "Long-term pharmacological treatment for PD improved outcomes "
        "during follow-up."
    )
    with pytest.raises(InsufficientAnswerEvidence):
        generate(q, [hit(f"{intro} {finding}")])


def test_p12_source_defined_pd_abbreviation_does_not_override_other_disorder():
    q = "What is known about long-term pharmacological treatment for social anxiety disorder?"
    definition = "This review studied panic disorder (PD) and other conditions."
    finding = "Long-term pharmacological treatment for PD improved outcomes."
    with pytest.raises(InsufficientAnswerEvidence):
        generate(q, [hit(f"{definition} {finding}")])


def test_p11_symptom_study_does_not_establish_disorder_specific_efficacy():
    q = "What research supports internet-delivered CBT for social anxiety disorder?"
    finding = (
        "The web-only social anxiety CBT program appeared efficacious "
        "in participants with self-reported symptoms of social anxiety."
    )
    with pytest.raises(InsufficientAnswerEvidence):
        generate(q, [hit(finding)])
