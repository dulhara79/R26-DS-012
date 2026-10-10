from care_anxrag.clinical_match import build_clinical_evidence_facets


def test_clinical_facets_use_deterministic_source_text_matches() -> None:
    facets = build_clinical_evidence_facets(
        text=(
            "Older adults with generalized anxiety disorder received "
            "cognitive behavioural therapy."
        ),
        topics=["Anxiety Disorders"],
        annotated_pico=None,
    )

    assert facets["anxiety_subtypes"] == [
        "generalized_anxiety_disorder"
    ]
    assert facets["treatments"] == [
        "cognitive_behavioral_therapy"
    ]
    assert facets["populations"] == ["older_adults"]
    assert facets["pico"] == {
        "population": [],
        "intervention": [],
        "comparator": [],
        "outcome": [],
    }
    assert facets["provenance"]["normalized_concepts"] == (
        "deterministic_phrase_match"
    )


def test_clinical_facets_preserve_explicit_pico_annotations_verbatim() -> None:
    facets = build_clinical_evidence_facets(
        text="A randomized anxiety treatment study.",
        topics=["generalized_anxiety_disorder"],
        annotated_pico={
            "population": ["Adults with GAD"],
            "intervention": ["Manualized CBT"],
            "comparator": ["Treatment as usual"],
            "outcome": ["GAD-7 symptom severity at 12 weeks"],
        },
    )

    assert facets["pico"] == {
        "population": ["Adults with GAD"],
        "intervention": ["Manualized CBT"],
        "comparator": ["Treatment as usual"],
        "outcome": ["GAD-7 symptom severity at 12 weeks"],
    }
    assert facets["provenance"]["pico"] == "source_or_reviewer_metadata"


def test_clinical_facets_do_not_infer_missing_comparator_or_outcome() -> None:
    facets = build_clinical_evidence_facets(
        text=(
            "CBT was evaluated in adults with generalized anxiety disorder."
        ),
        topics=[],
        annotated_pico={},
    )

    assert facets["pico"]["comparator"] == []
    assert facets["pico"]["outcome"] == []



def test_clinical_facets_capture_outcomes_and_comorbidities_deterministically() -> None:
    facets = build_clinical_evidence_facets(
        text=(
            "Adults with generalized anxiety disorder and major depressive "
            "disorder received cognitive behavioural therapy. Remission and "
            "quality of life were measured."
        ),
        topics=["generalized_anxiety_disorder"],
        annotated_pico=None,
    )

    assert facets["comorbidities"] == [
        "major_depressive_disorder"
    ]
    assert facets["outcomes"] == [
        "quality_of_life",
        "remission",
    ]
