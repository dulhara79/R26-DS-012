from care_anxrag.clinical_match import (
    extract_population_concepts,
    extract_subtype_concepts,
    extract_treatment_concepts,
    population_compatibility,
    supports_explicit_treatment_query,
    treatment_compatibility,
)


def test_extracts_multiple_anxiety_treatments_without_confusing_them() -> None:
    assert extract_treatment_concepts(
        "CBT for generalized anxiety disorder"
    ) == {"cognitive_behavioral_therapy"}
    assert extract_treatment_concepts(
        "Metacognitive therapy for GAD"
    ) == {"metacognitive_therapy"}
    assert "exposure_therapy" in extract_treatment_concepts(
        "Exposure-based therapy for panic disorder"
    )


def test_treatment_compatibility_penalizes_explicit_other_treatment() -> None:
    requested = {"cognitive_behavioral_therapy"}
    assert treatment_compatibility(
        requested,
        "A trial of cognitive behavioural therapy",
    ) == 1.0
    assert treatment_compatibility(
        requested,
        "A trial of metacognitive therapy",
    ) < 0.70
    assert 0.70 <= treatment_compatibility(
        requested,
        "General anxiety guidance",
    ) < 1.0


def test_population_compatibility_penalizes_explicit_mismatch() -> None:
    assert population_compatibility(
        "older_adults",
        "Treatment in older adults aged 65 years and above",
    ) == 1.0
    assert population_compatibility(
        "older_adults",
        "Treatment in adolescents aged 13 to 17 years",
    ) < 0.70
    assert 0.70 <= population_compatibility(
        "older_adults",
        "General treatment guidance",
    ) < 1.0


def test_joint_support_requires_same_evidence_for_subtype_and_treatment() -> None:
    requested_subtypes = {"generalized_anxiety_disorder"}
    requested_treatments = {"cognitive_behavioral_therapy"}

    assert not supports_explicit_treatment_query(
        requested_subtypes,
        requested_treatments,
        None,
        [],
        "Metacognitive therapy for adults with generalized anxiety disorder.",
    )
    assert not supports_explicit_treatment_query(
        requested_subtypes,
        requested_treatments,
        None,
        [],
        "Cognitive behavioural therapy for adults with social anxiety disorder.",
    )
    assert supports_explicit_treatment_query(
        requested_subtypes,
        requested_treatments,
        None,
        [],
        "Cognitive behavioural therapy for adults with generalized anxiety disorder.",
    )


def test_topics_may_support_subtype_but_not_treatment() -> None:
    assert supports_explicit_treatment_query(
        {"generalized_anxiety_disorder"},
        {"cognitive_behavioral_therapy"},
        None,
        ["generalized_anxiety_disorder", "Cognitive Behavioral Therapy"],
        "This abstract evaluates cognitive behavioural therapy outcomes.",
    )
    assert not supports_explicit_treatment_query(
        {"generalized_anxiety_disorder"},
        {"cognitive_behavioral_therapy"},
        None,
        ["generalized_anxiety_disorder", "Cognitive Behavioral Therapy"],
        "This abstract evaluates metacognitive therapy outcomes.",
    )


def test_population_and_subtype_extractors_handle_common_clinical_phrasing() -> None:
    assert extract_population_concepts(
        "older adults with anxiety"
    ) == {"older_adults"}
    assert extract_population_concepts(
        "adults with anxiety"
    ) == {"adults"}
    assert "generalized_anxiety_disorder" in extract_subtype_concepts(
        "patients with GAD"
    )


def test_specific_treatment_variants_retain_parent_concepts() -> None:
    vret = extract_treatment_concepts(
        "virtual reality exposure therapy for social anxiety"
    )
    assert {
        "virtual_reality_exposure_therapy",
        "exposure_therapy",
    } <= vret

    internet_cbt = extract_treatment_concepts(
        "internet-delivered cognitive behavioural therapy for panic disorder"
    )
    assert {
        "internet_delivered_cbt",
        "cognitive_behavioral_therapy",
    } <= internet_cbt
