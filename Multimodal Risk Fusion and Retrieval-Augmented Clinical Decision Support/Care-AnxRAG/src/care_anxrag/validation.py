from __future__ import annotations

import re
from dataclasses import dataclass

from .evidence import is_retracted
from .models import RawDocument, SourceConfig
from .util import normalize_whitespace


# ============================================================================
# CARE-AnxRAG Domain Validation
#
# This validator answers a different question from retrieval relevance:
#
#     INGESTION VALIDATION:
#       "Does this paper belong in an anxiety-disorder evidence corpus?"
#
#     RETRIEVAL RELEVANCE:
#       "How relevant is an already-approved chunk to the user's question?"
#
# These must remain separate.
# ============================================================================


# ---------------------------------------------------------------------------
# Strong anxiety-disorder concepts
# ---------------------------------------------------------------------------

CORE_ANXIETY_PHRASES = {
    "anxiety disorder",
    "anxiety disorders",
    "generalized anxiety disorder",
    "generalised anxiety disorder",
    "generalized anxiety",
    "generalised anxiety",
    "social anxiety disorder",
    "social anxiety",
    "panic disorder",
    "panic attacks",
    "agoraphobia",
    "specific phobia",
    "specific phobias",
    "separation anxiety disorder",
    "separation anxiety",
    "selective mutism",
    "pathological worry",
    "excessive worry",
    "chronic worry",
    "high worriers",
    "anxiety intervention",
    "anxiety treatment",
    "anxiety therapy",
    "anxiety-focused intervention",
    "anxiety-focused treatment",
}

# ---------------------------------------------------------------------------
# Anxiety-specific treatment/intervention concepts
# ---------------------------------------------------------------------------

ANXIETY_INTERVENTION_PHRASES = {
    "cognitive behavioral therapy",
    "cognitive behavioural therapy",
    "cbt",
    "exposure therapy",
    "exposure-based therapy",
    "exposure-based treatment",
    "internet-based cognitive behavioral therapy",
    "internet-based cognitive behavioural therapy",
    "internet-delivered cognitive behavioral therapy",
    "internet-delivered cognitive behavioural therapy",
    "metacognitive therapy",
    "acceptance and commitment therapy",
    "anxiety intervention",
    "worry intervention",
    "worry intervention trial",
    "cognitive bias modification",
    "cbm-i",
    "virtual reality exposure therapy",
}


# ---------------------------------------------------------------------------
# Situations where anxiety is usually temporary/procedural rather than
# an anxiety disorder being studied.
# ---------------------------------------------------------------------------

PROCEDURAL_PHRASES = {
    "preoperative",
    "pre-operative",
    "postoperative",
    "post-operative",
    "perioperative",
    "procedure-related anxiety",
    "procedural anxiety",
    "procedural fear",
    "surgical anxiety",
    "dental anxiety",
    "dental procedure",
    "dental treatment",
    "biopsy",
    "urodynamic",
    "endoscopy",
    "colonoscopy",
    "lumbar puncture",
    "mri examination",
    "ct examination",
    "computed tomography",
    "magnetic resonance imaging",
    "caesarean",
    "cesarean",
    "inhaler treatment",
    "root canal",
    "postoperative pain",
    "needle fear",
    "sore throat",
}


# ---------------------------------------------------------------------------
# Study protocol indicators
#
# Protocols are not completed outcome evidence.
# ---------------------------------------------------------------------------

PROTOCOL_PHRASES = {
    "study protocol",
    "trial protocol",
    "protocol paper",
    "protocol for the prevention",
    "protocol for a randomized",
    "protocol for a randomised",
    "protocol for an rct",
    "protocol for a clinical trial",
    "protocol of a randomized",
    "protocol of a randomised",
}


# ---------------------------------------------------------------------------
# Terms indicating anxiety may be secondary to another primary condition.
#
# These do NOT automatically cause rejection.
# They help determine whether anxiety is the actual clinical target.
# ---------------------------------------------------------------------------

OTHER_PRIMARY_CONDITION_PHRASES = {
    "rheumatoid arthritis",
    "temporomandibular",
    "major depression",
    "major depressive disorder",
    "breast cancer",
    "prostate cancer",
    "cancer patients",
    "dementia",
    "psychosis",
    "first-episode psychosis",
    "irreversible vision loss",
    "asthma",
    "anorexia nervosa",
    "eating disorder",
    "copd",
    "chronic obstructive pulmonary disease",
    "diabetic",
    "vitamin d",
}


# ---------------------------------------------------------------------------
# Wording indicating that anxiety is only an outcome/measurement.
# ---------------------------------------------------------------------------

SECONDARY_OUTCOME_PHRASES = {
    "secondary outcome",
    "secondary outcomes",
    "anxiety levels",
    "anxiety score",
    "anxiety scores",
    "anxiety symptoms were assessed",
    "anxiety was assessed",
    "gad-7 questionnaire",
    "generalized anxiety disorder-7",
    "generalised anxiety disorder-7",
    "hospital anxiety and depression scale",
}


# ---------------------------------------------------------------------------
# Mixed mental-health evidence may still be useful in Research Frontier.
# ---------------------------------------------------------------------------

MIXED_MENTAL_HEALTH_PHRASES = {
    "depression and anxiety",
    "anxiety and depression",
    "depression, anxiety",
    "anxiety, depression",
    "post-traumatic stress disorder",
    "posttraumatic stress disorder",
    "ptsd",
}



@dataclass(slots=True)
class DomainAssessment:
    role: str
    evidence_role: str
    score: float
    reasons: list[str]


@dataclass(slots=True)
class ValidationResult:
    accepted: bool
    reasons: list[str]
    relevance_score: float


class DocumentValidator:
    def __init__(
        self,
        minimum_characters: int = 300,
        minimum_relevance: float = 0.35,
    ):
        self.minimum_characters = minimum_characters
        self.minimum_relevance = minimum_relevance

    def validate(
        self,
        document: RawDocument,
        source: SourceConfig,
    ) -> ValidationResult:
        reasons: list[str] = []

        title = normalize_whitespace(
            document.title or ""
        ).lower()

        text = normalize_whitespace(
            document.text or ""
        ).lower()

        assessment = assess_anxiety_domain(
            title=title,
            text=text,
        )

        # ------------------------------------------------------------------
        # General document quality
        # ------------------------------------------------------------------

        if len(document.text.strip()) < self.minimum_characters:
            reasons.append("document_too_short")

        if document.language.lower() not in {
            "en",
            "eng",
            "english",
        }:
            reasons.append("unsupported_language")

        # ------------------------------------------------------------------
        # PubMed / PMC domain gate
        # ------------------------------------------------------------------

        if source.connector in {"pubmed", "pmc"}:

            # Procedural anxiety should not enter the anxiety-disorder corpus.
            if assessment.role == "procedural_anxiety":
                reasons.append("procedural_anxiety_not_disorder_evidence")

            # Anxiety appearing only as a secondary outcome is not enough.
            elif assessment.role == "secondary_anxiety_outcome":
                reasons.append("anxiety_is_secondary_outcome")

            # No meaningful anxiety-disorder domain fit.
            elif assessment.role == "out_of_domain":
                reasons.append("low_anxiety_relevance")

            # Protocols describe planned trials rather than completed results.
            if assessment.evidence_role == "study_protocol":
                reasons.append("study_protocol_not_result_evidence")

            # Final numerical safety gate.
            if (
                assessment.score < self.minimum_relevance
                and assessment.role
                not in {
                    "core_anxiety",
                    "anxiety_comorbid",
                }
            ):
                if "low_anxiety_relevance" not in reasons:
                    reasons.append("low_anxiety_relevance")

        # ------------------------------------------------------------------
        # Retraction / withdrawal guardrail
        # ------------------------------------------------------------------

        if is_retracted(
            document.publication_types,
            document.metadata,
        ):
            reasons.append("retracted_or_withdrawn")

        # ------------------------------------------------------------------
        # PMC reuse/licensing guardrail
        # ------------------------------------------------------------------

        if (
            source.connector == "pmc"
            and not bool(
                document.metadata.get(
                    "reuse_allowed",
                    False,
                )
            )
        ):
            reasons.append(
                "full_text_reuse_not_confirmed"
            )

        # Remove duplicate reason codes while preserving order.
        reasons = list(dict.fromkeys(reasons))

        return ValidationResult(
            accepted=not reasons,
            reasons=reasons,
            relevance_score=assessment.score,
        )


def assess_anxiety_domain(
    title: str,
    text: str,
) -> DomainAssessment:
    """
    Classify how a document relates to anxiety.

    Possible domain roles:

        core_anxiety
            Anxiety disorder / pathological anxiety is the primary topic.

        anxiety_comorbid
            Anxiety is an important target, but the study population or
            intervention is mixed with another disorder/population.

        secondary_anxiety_outcome
            Anxiety is mainly a measurement or secondary endpoint.

        procedural_anxiety
            Temporary anxiety/fear related to surgery, imaging, biopsy,
            dental treatment, etc.

        out_of_domain
            Insufficient evidence that the document belongs in the
            anxiety-disorder corpus.

    Evidence roles:

        result_evidence
        review_evidence
        study_protocol
        other
    """

    title = normalize_whitespace(title or "").lower()
    text = normalize_whitespace(text or "").lower()
    combined = f"{title} {text}"

    reasons: list[str] = []

    # ======================================================================
    # 1. Evidence role
    # ======================================================================

    protocol_detected = _contains_any(
        combined,
        PROTOCOL_PHRASES,
    )

    if protocol_detected:
        evidence_role = "study_protocol"
        reasons.append("protocol_language_detected")

    elif (
        "meta-analysis" in title
        or "meta analysis" in title
        or "systematic review" in title
        or "umbrella review" in title
    ):
        evidence_role = "review_evidence"

    else:
        evidence_role = "result_evidence"

    # ======================================================================
    # 2. Core anxiety signals
    # ======================================================================

    title_core_matches = _count_phrase_matches(
        title,
        CORE_ANXIETY_PHRASES,
    )

    text_core_matches = _count_phrase_matches(
        text,
        CORE_ANXIETY_PHRASES,
    )

    title_has_anxiety = _word_present(
        title,
        "anxiety",
    )

    text_has_anxiety = _word_present(
        text,
        "anxiety",
    )

    title_has_worry = _word_present(
        title,
        "worry",
    ) or "worriers" in title

    anxiety_count = len(
        re.findall(
            r"\banxiety\b",
            combined,
        )
    )

    gad_abbreviation = bool(
        re.search(
            r"\bgad\b",
            combined,
        )
    )

    intervention_match = _contains_any(
        combined,
        ANXIETY_INTERVENTION_PHRASES,
    )

    # ======================================================================
    # 3. Procedural anxiety detection
    # ======================================================================

    procedural_title = _contains_any(
        title,
        PROCEDURAL_PHRASES,
    )

    procedural_text = _contains_any(
        text,
        PROCEDURAL_PHRASES,
    )

    strong_core_title = (
        title_core_matches > 0
        or "generalized anxiety disorder" in title
        or "generalised anxiety disorder" in title
        or "social anxiety disorder" in title
        or "panic disorder" in title
        or "anxiety disorders" in title
    )

    # If the title itself clearly describes a procedure and there is no
    # anxiety-disorder title anchor, procedural anxiety wins.
    if (
        procedural_title
        and not strong_core_title
    ):
        return DomainAssessment(
            role="procedural_anxiety",
            evidence_role=evidence_role,
            score=0.0,
            reasons=[
                *reasons,
                "procedural_context_in_title",
            ],
        )

    # ======================================================================
    # 4. Detect another primary medical/psychiatric condition
    # ======================================================================

    other_condition_title = _contains_any(
        title,
        OTHER_PRIMARY_CONDITION_PHRASES,
    )

    secondary_measure_language = _contains_any(
        combined,
        SECONDARY_OUTCOME_PHRASES,
    )

    mixed_mental_health = _contains_any(
        combined,
        MIXED_MENTAL_HEALTH_PHRASES,
    )

    # ======================================================================
    # 5. Score domain fit
    # ======================================================================

    score = 0.0

    # Strongest signal: anxiety-disorder concept in title.
    if title_core_matches > 0:
        score += 0.55
        reasons.append("core_anxiety_phrase_in_title")

    if title_core_matches >= 2:
        score += 0.10

    # Anxiety-disorder concepts throughout abstract.
    if text_core_matches > 0:
        score += 0.15

    if text_core_matches >= 3:
        score += 0.05

    # Generic anxiety in title is useful, but insufficient alone.
    if title_has_anxiety:
        score += 0.15

    # Worry-focused title.
    if title_has_worry:
        score += 0.15

    # Explicit GAD abbreviation.
    if gad_abbreviation:
        score += 0.10

    # Anxiety-specific intervention combined with anxiety anchor.
    anxiety_anchor = (
        title_has_anxiety
        or text_has_anxiety
        or title_has_worry
        or _word_present(combined, "worry")
        or _word_present(combined, "panic")
        or _word_present(combined, "phobia")
        or "agoraphobia" in combined
    )

    if (
        intervention_match
        and anxiety_anchor
    ):
        score += 0.10
        reasons.append("anxiety_targeted_intervention")

    # Repeated use of anxiety contributes only a little.
    if anxiety_count >= 4:
        score += 0.03

    if anxiety_count >= 10:
        score += 0.02

    # ======================================================================
    # 6. Penalties
    # ======================================================================

    # Procedure appears in abstract but not title.
    if (
        procedural_text
        and not strong_core_title
    ):
        score -= 0.12
        reasons.append("procedural_context_in_abstract")

    # Another condition dominates the title.
    if (
        other_condition_title
        and not strong_core_title
    ):
        score -= 0.20
        reasons.append("other_primary_condition_in_title")

    # Anxiety clearly described as a measurement/secondary endpoint.
    if (
        secondary_measure_language
        and not strong_core_title
    ):
        score -= 0.20
        reasons.append("secondary_anxiety_measure")

    # Mixed-condition mental-health studies are allowed but slightly
    # discounted unless anxiety is explicitly the primary title concept.
    if (
        mixed_mental_health
        and not strong_core_title
    ):
        score -= 0.08
        reasons.append("mixed_mental_health_topic")

    score = _clamp_score(score)

    # ======================================================================
    # 7. Domain-role decision
    # ======================================================================

    # ---------------------------------------------------------------
    # Core anxiety
    # ---------------------------------------------------------------

    if (
        strong_core_title
        and other_condition_title
    ):
        role = "anxiety_comorbid"
        reasons.append("anxiety_with_other_primary_condition")

    elif strong_core_title:
        role = "core_anxiety"

    # Worry-focused research can be core even without the exact phrase
    # "anxiety disorder".
    elif (
        title_has_worry
        and intervention_match
        and score >= 0.35
    ):
        role = "core_anxiety"

    # ---------------------------------------------------------------
    # Anxiety comorbidity / mixed-population evidence
    # ---------------------------------------------------------------

    elif (
        title_has_anxiety
        and (
            mixed_mental_health
            or other_condition_title
        )
        and score >= 0.35
        and not secondary_measure_language
    ):
        role = "anxiety_comorbid"

    # Meta-analyses/reviews directly addressing anxiety treatment can
    # remain useful Research Frontier evidence even when another condition
    # such as PTSD is also included.
    elif (
        evidence_role == "review_evidence"
        and title_has_anxiety
        and intervention_match
        and score >= 0.35
    ):
        role = "anxiety_comorbid"

    # ---------------------------------------------------------------
    # Secondary anxiety outcome
    # ---------------------------------------------------------------

    elif (
        secondary_measure_language
        or (
            other_condition_title
            and not strong_core_title
        )
    ):
        role = "secondary_anxiety_outcome"

    # ---------------------------------------------------------------
    # Weak/incidental anxiety mention
    # ---------------------------------------------------------------

    elif score < 0.35:
        role = "out_of_domain"

    else:
        role = "anxiety_comorbid"

    return DomainAssessment(
        role=role,
        evidence_role=evidence_role,
        score=score,
        reasons=reasons,
    )


def _contains_any(
    text: str,
    phrases: set[str],
) -> bool:
    return any(
        phrase in text
        for phrase in phrases
    )


def _count_phrase_matches(
    text: str,
    phrases: set[str],
) -> int:
    return sum(
        1
        for phrase in phrases
        if phrase in text
    )


def _word_present(
    text: str,
    word: str,
) -> bool:
    return bool(
        re.search(
            rf"\b{re.escape(word)}\b",
            text,
        )
    )


def _clamp_score(
    value: float,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            round(value, 4),
        ),
    )
