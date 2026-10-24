"""Conservative, deterministic selection of complete source sentences.

These English phrase rules filter obvious non-answers; they are not a clinical
correctness classifier. Retrieval quality and exact-source grounding still apply.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

from .clinical_match import (
    extract_population_concepts,
    extract_subtype_concepts,
    supports_explicit_clinical_context,
)
from .models import EvidenceLevel, QueryAnalysis, QueryIntent, SearchHit
from .util import content_tokens, source_sentences
from .question_requirements import eligible_document_for_question, is_treatment_review_request


class InsufficientAnswerEvidence(ValueError):
    """Relevant passages contained no eligible answer sentence."""

    def __init__(self) -> None:
        super().__init__("insufficient_answer_evidence")


_NON_FINDING = re.compile(
    r"\b(?:aims?|objectives?|purpose|goal)\b.{0,120}"
    r"\b(?:study|investigat\w*|assess\w*|evaluat\w*|compar\w*|determin\w*|"
    r"examin\w*|test\w*|was|were|is|are)\b"
    r"|\b(?:aimed|sought|intend(?:ed)?|plan(?:ned)?) to\b"
    r"|\b(?:study|trial|review|we) (?:investigat\w*|evaluat\w*|assess\w*|compar\w*|examin\w*)\b"
    r"|\b(?:was|were) (?:evaluated|assessed|measured|examined)\b"
    r"|\bhypothes(?:is|es|iz\w*|is\w*)\b"
    r"|\b(?:will|expected to)\b"
    r"|^(?:objectives?|aims?|methods?|design|protocol)\s*:"
    r"|\b(?:randomly assigned|randomi[sz]ed to|recruited|enrolled)\b",
    re.I,
)
_FINDING = re.compile(
    r"\b(?:reduc(?:e[sd]?|ing|tion)|improv(?:e[sd]?|ing|ement)|benefit(?:s|ed)?|"
    r"effective(?:ness)?|efficacy|efficacious|research[- ]supported|useful in treat\w*|remission|response rate|relapse|"
    r"adverse (?:effects?|events?)|side effects?|recommended|first[- ]line|essential treatments?|"
    r"uncertain|inconclusive|insufficient|no (?:significant )?(?:difference|effect)|"
    r"not differ|equivalent|most evidence)\b"
    r"|\bevidence (?:supports?|suggests?)\b"
    r"|\bevidence[- ](?:based|supported) (?:treatments?|approaches?|interventions?|options?)\b"
    r"|\b(?:can|may) help\b"
    r"|\b(?:can|may) (?:also )?(?:be )?(?:helpful|useful|beneficial)\b",
    re.I,
)
_EFFECTIVENESS_QUESTION = re.compile(
    r"\b(?:effective(?:ness)?|efficacy|works?|helps?|helpful|benefits?|"
    r"reduc\w*|improv\w*|outcomes?|results?|findings?|"
    r"support(?:ed|s|ing)?|recommended|recommendations?|best|first[- ]line)\b",
    re.I,
)
_DESCRIPTIVE_QUESTION = re.compile(r"\b(?:discussed|mentioned|listed|delivered|offered|used)\b", re.I)
_DIRECT_DIAGNOSIS = re.compile(
    r"\b(?:diagnostic criteria|diagnos(?:is|ed|ing) (?:requires|is based on|by|when)|"
    r"clinical (?:assessment|interview|evaluation)|"
    r"DSM[- ]?5(?:[- ]TR)?|ICD[- ]?1[01]|"
    r"recurrent unexpected panic attacks|"
    r"persistent (?:concern|worry) about (?:additional|future) panic attacks)\b",
    re.I,
)
_MEDICATION_MENTION = re.compile(
    r"\b(?:medication|medicine|pharmacolog\w*|antidepressants?|"
    r"SSRIs?|SNRIs?|selective serotonin reuptake inhibitors?|"
    r"benzodiazepines?|tricyclic antidepressants?|azapirones?)\b",
    re.I,
)
_COMPARATIVE_FINDING = re.compile(
    r"\b(?:compar(?:ed|ison) (?:with|to|between)|versus|vs\.?|"
    r"relative to|difference(?:s)? between|more effective than|"
    r"less effective than|superior to|inferior to|non[- ]inferior)\b",
    re.I,
)
_RESULT_STATEMENT = re.compile(
    r"\b(?:reduced|improved|benefited|found|observed|showed|results)\b"
    r"|\b(?:did not|had no effect|no (?:significant )?difference|"
    r"not effective|uncertain|inconclusive|insufficient)\b",
    re.I,
)



# Professional evidence questions require a *response to the requested task*,
# not merely a relevant paper or an exact citation. These conservative guards
# may abstain: they cannot establish clinical validity or evaluate study quality.
_INTERVENTION = re.compile(
    r"\b(?:CBT|cognitive[- ]behaviou?ral therap(?:y|ies)|"
    r"exposure[- ](?:based )?(?:therapy|treatment)|metacognitive therap(?:y|ies)|"
    r"psychotherap(?:y|ies)|psychological (?:therap(?:y|ies)|interventions?|treatments?)|"
    r"pharmacolog(?:y|ical)|non[- ]pharmacological interventions?|medications?|antidepressants?|SSRIs?|SNRIs?|"
    r"benzodiazepines?|sertraline|escitalopram|"
    r"repetitive transcranial magnetic stimulation|rTMS|tDCS|"
    r"mindfulness[- ]based (?:therapy|intervention)|"
    r"interventions? (?:for|to treat) anxiety disorders?)\b",
    re.I,
)
_REVIEW_REQUEST = re.compile(
    r"\b(?:systematic reviews?|meta[- ]analys(?:es|is))\b", re.I
)
_AUTHORITATIVE_REQUEST = re.compile(
    r"\b(?:authoritative|official|guidelines?|recommendations?)\b", re.I
)
_CONTRADICTION_REQUEST = re.compile(
    r"\b(?:contradict(?:s|ions?|ory)?|conflict(?:s|ing)?|"
    r"disagree(?:s|ment)?|inconsistent|opposing)\b", re.I
)
_NEGATIVE_TREATMENT_RESULT = re.compile(
    r"\b(?:did not (?:reduc\w*|improv\w*|benefit)|"
    r"no (?:statistically )?(?:significant )?(?:difference|benefit|effect)|"
    r"not effective|not superior|inferior to|"
    r"failed to (?:reduc\w*|improv\w*|show)|"
    r"contradictory findings|conflicting (?:results|evidence)|"
    r"inconsistent (?:results|findings|evidence))\b",
    re.I,
)
_COMPARISON_REQUEST = re.compile(
    r"\b(?:compar(?:e|es|ed|ing|ison|ative)|versus|vs\.?)\b", re.I
)
_COMPARATIVE_RESULT = re.compile(
    r"\b(?:compared (?:with|to)|versus|vs\.?|relative to|"
    r"superior|inferior|equivalent|no (?:significant )?difference|"
    r"equally effective|more effective than|less effective than)\b", re.I
)
_DIAGNOSIS_PROCESS = re.compile(
    r"\b(?:diagnosis (?:requires|involves|is based on)|"
    r"panic disorder (?:may be|is) diagnosed when|"
    r"to be diagnosed with panic disorder|"
    r"diagnostic criteria (?:include|require|specify)|"
    r"clinical (?:assessment|evaluation|interview)|"
    r"structured (?:clinical )?interview|"
    r"recurrent (?:and )?unexpected panic attacks|"
    r"DSM[- ]?\d+|ICD[- ]?\d+|"
    r"rule out|ruling out|assess(?:ment|ing)? (?:of )?(?:symptoms|panic attacks))\b",
    re.I,
)
_NON_ANSWER_METHOD = re.compile(
    r"\b(?:the following sources were used|databases? (?:were|was) searched|"
    r"we (?:searched|conducted|performed)|"
    r"this (?:study|review) (?:aims|aimed|investigated)|"
    r"(?:electronic |bibliographic )?databases?:|"
    r"study protocol|meeting diagnostic criteria|randomly allocated)\b"
    r"|\b(?:this|the) (?:network )?(?:meta-analysis|systematic review) (?:integrates|examines|explores|evaluates|assesses)\b"
    r"|^(?:(?:to )?(?:evaluate|assess|investigate|compare|explore|determine|examine)\b"
    r"|comparing the (?:effectiveness|efficacy|safety)\b)",
    re.I,
)
# A sentence such as "These medications can be very effective..." relies
# on an antecedent that is omitted from a one-sentence extractive answer.
# Exact-substring grounding alone cannot make such a claim self-contained.
_DEPENDENT_SOURCE_SENTENCE = re.compile(
    r"^(?:these|those|such|this)\s+(?:medications?|drugs?|treatments?|interventions?)\b",
    re.I,
)


# Delivery method and treatment duration are integral to the requested clinical
# intervention. A citation about ordinary CBT cannot answer an internet-CBT
# question, and rapid symptom relief is not evidence for long-term treatment.
_CBT_TOPIC = re.compile(
    r"\b(?:CBT|cognitive[- ]behaviou?ral(?:\s+therapy)?)\b", re.I,
)
_CBT_DELIVERY_QUERY = re.compile(
    r"\b(?:internet[- ](?:delivered|based)|online|web[- ]based|digital|"
    r"electronic(?:ally)?[- ]delivered|e[- ]CBT|iCBT)\b", re.I,
)
_CBT_DELIVERY_EVIDENCE = re.compile(
    r"\b(?:internet[- ](?:delivered|based)|online|web[- ]based|digital|"
    r"electronically[- ]delivered|electronically delivered|"
    r"e[- ]CBT|iCBT|DCBT)\b", re.I,
)
_LONG_TERM_REQUEST = re.compile(r"\blong[- ]term\b", re.I)
_LONG_TERM_EVIDENCE = re.compile(
    r"\b(?:long[- ]term|maintenance|sustained treatment)\b", re.I,
)
_PREVALENCE_BACKGROUND = re.compile(
    r"\b(?:prevalence|incidence|occurs in\s+\d|affects\s+\d)\b", re.I,
)


_DISORDER_TREATMENT = re.compile(
    r"\b(?:anxiety disorders?|"
    r"generaliz(?:ed|ised) anxiety disorder|panic disorder|social anxiety disorder|"
    r"agoraphobia|specific phobia)\b",
    re.I,
)


def _requires_clinician_task_evidence(
    sentence: str, hit: SearchHit, analysis: QueryAnalysis,
    *, allow_conflict_pair: bool = False,
) -> bool:
    """Require the sentence to address a clinician's requested evidence task.

    This supplements (but does not replace) source-substring grounding and the
    population/subtype safeguards. False negatives are preferred to false claims.
    """
    question = analysis.normalized_query

    if _NON_ANSWER_METHOD.search(sentence):
        return False
    # An article's heading/title is a topic label, not evidence that the
    # intervention worked or that diagnostic criteria were established.
    if sentence.casefold().strip(" .:;") == hit.chunk.title.casefold().strip(" .:;"):
        return False

    review_request = bool(_REVIEW_REQUEST.search(question))
    if review_request:
        if hit.chunk.evidence_level not in {
            EvidenceLevel.SYSTEMATIC_REVIEW,
            EvidenceLevel.META_ANALYSIS,
        }:
            return False
        if analysis.wants_recent:
            published = hit.chunk.published_at
            if published is None:
                return False
            # A documented operator definition for "recent": last 3 years.
            if published.date() < (datetime.now(UTC) - timedelta(days=3 * 365)).date():
                return False
        # A methodology sentence or a review of generic anxiety outcomes
        # (such as biomarkers) does not answer anxiety-disorder *treatment*.
        if re.search(r"\b(?:treatments?|interventions?)\b", question):
            if not (
                _DISORDER_TREATMENT.search(sentence)
                and _INTERVENTION.search(sentence)
                and _FINDING.search(sentence)
            ):
                return False

    if (
        analysis.intent == QueryIntent.MEDICATION
        and _AUTHORITATIVE_REQUEST.search(question)
        and hit.chunk.evidence_level not in {
            EvidenceLevel.CLINICAL_GUIDELINE,
            EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION,
        }
    ):
        return False

    if analysis.intent == QueryIntent.MEDICATION:
        if not (_INTERVENTION.search(sentence) and _FINDING.search(sentence)):
            return False

    if analysis.intent == QueryIntent.TREATMENT:
        # Background impairment is not evidence about treatments.
        if re.search(r"\b(?:treatments?|therap(?:y|ies)|interventions?|approaches?|CBT)\b", question, re.I):
            if not _INTERVENTION.search(sentence):
                return False
        # Background prevalence and symptom descriptions cannot satisfy a
        # question asking what interventions are supported.
        if (
            _EFFECTIVENESS_QUESTION.search(question)
            and not (_INTERVENTION.search(sentence) and _FINDING.search(sentence))
        ):
            return False
        if (
            _COMPARISON_REQUEST.search(question)
            and len(analysis.treatments) >= 2
            and not (_COMPARATIVE_RESULT.search(sentence) and _FINDING.search(sentence))
        ):
            return False

    if _CONTRADICTION_REQUEST.search(question) and not allow_conflict_pair:
        # This only selects explicit negative comparative findings; a research
        # gap, a study aim, or a generic "however" is not contradictory evidence.
        if not (
            _NEGATIVE_TREATMENT_RESULT.search(sentence)
            and _INTERVENTION.search(sentence)
        ):
            return False

    if analysis.intent == QueryIntent.DIAGNOSIS:
        if re.search(r"\bhow\b.*\bdiagnos(?:e|ed|is)\b", question):
            if not _DIAGNOSIS_PROCESS.search(sentence):
                return False

    return True

def select_answer_sentence(
    hit: SearchHit, analysis: QueryAnalysis, *,
    required_topic: str | None = None,
    allow_conflict_pair: bool = False,
) -> str | None:
    """Return an eligible source sentence, retaining its wording and scope.

    Treatment/effectiveness questions require a finding or evidence statement,
    rather than an objective or a delivery description. Explicit clinical facets
    must be supported by the sentence itself, not borrowed from another sentence.
    Known population scope must remain in the selected text; otherwise omit it.
    """
    if hit.excluded_due_to_conflict or not eligible_document_for_question(hit, analysis):
        return None

    candidates = [
        part for part in source_sentences(hit.chunk.text)
        if not part.lstrip().startswith("#")
    ]
    # Use the synonym-enriched retrieval query so GAD and hyphenated
    # clinical terms do not cause false-negative token overlap checks.
    query_tokens = set(content_tokens(analysis.retrieval_query.replace('-', ' ')))
    asks_for_findings = bool(_EFFECTIVENESS_QUESTION.search(analysis.normalized_query)) or (
        "evidence" in query_tokens
        and not _DESCRIPTIVE_QUESTION.search(analysis.normalized_query)
    )
    requires_finding = (
        analysis.intent in {QueryIntent.TREATMENT, QueryIntent.MEDICATION}
        and asks_for_findings
    ) or (
        bool(analysis.treatments)
        and analysis.intent == QueryIntent.RECENT_RESEARCH
    ) or is_treatment_review_request(analysis)
    evidence_populations = extract_population_concepts(
        f"{hit.chunk.title}\n{hit.chunk.text}"
    )
    requested_populations = extract_population_concepts(analysis.normalized_query)
    if analysis.population:
        requested_populations.add(analysis.population)
    facets = hit.chunk.metadata.get("clinical_facets", {})
    if isinstance(facets, dict):
        populations = facets.get("populations", [])
        if isinstance(populations, list):
            facet_populations = {
                value for value in populations
                if isinstance(value, str)
                and value in {
                    "children_and_adolescents", "adults",
                    "older_adults", "perinatal",
                }
            }
            # In general-purpose government guidance (e.g. NIMH), the
            # document-wide facet may mention both children AND adults in
            # unrelated sections. That is not evidence that every unqualified
            # psychotherapy recommendation is pediatric-only or adult-only.
            # Preserve strict qualifications for trials, single-population
            # sources, and any query naming a specific population.
            broad_government_guidance = (
                hit.chunk.evidence_level == EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION
                and facet_populations == {"children_and_adolescents", "adults"}
                and not requested_populations
                and not extract_population_concepts(
                    f"{hit.chunk.section_heading}\n{hit.chunk.text}"
                )
            )
            if not broad_government_guidance:
                evidence_populations.update(facet_populations)

    eligible: list[str] = []
    for sentence in candidates:
        # A dangling list label is not a medical answer.
        if sentence.rstrip().endswith(":"):
            continue
        if _DEPENDENT_SOURCE_SENTENCE.search(sentence.strip()):
            continue
        if _NON_FINDING.search(sentence):
            continue
        if analysis.intent == QueryIntent.TREATMENT:
            # A disease prevalence/background sentence is not evidence that
            # a specific therapy works (P10 live regression).
            if _PREVALENCE_BACKGROUND.search(sentence) and not _INTERVENTION.search(sentence):
                continue
            # Keep requested delivery modality in the *selected sentence*.
            if (
                _CBT_TOPIC.search(analysis.normalized_query)
                and _CBT_DELIVERY_QUERY.search(analysis.normalized_query)
                and not (
                    _CBT_TOPIC.search(sentence)
                    and _CBT_DELIVERY_EVIDENCE.search(sentence)
                )
            ):
                continue
            # Short-acting medication effects cannot answer long-term use (P12).
            if (
                _LONG_TERM_REQUEST.search(analysis.normalized_query)
                and not _LONG_TERM_EVIDENCE.search(sentence)
            ):
                continue
        if not _requires_clinician_task_evidence(
            sentence, hit, analysis, allow_conflict_pair=allow_conflict_pair
        ):
            continue
        # Descriptive queries may list or identify an intervention without
        # claiming effectiveness. All other treatment queries require a
        # genuine evidence/finding statement, rather than general background.
        if (
            analysis.intent == QueryIntent.TREATMENT
            and not _DESCRIPTIVE_QUESTION.search(analysis.normalized_query)
            and not _FINDING.search(sentence)
        ):
            continue
        if requires_finding and not _FINDING.search(sentence):
            continue
        if required_topic == "psychological" and not re.search(
            r"\b(?:CBT|cognitive[- ]behaviou?ral|psychotherap\w*|psychological (?:therap\w*|interventions?|treatments?)|exposure therapy)\b",
            sentence, re.I,
        ):
            continue
        if required_topic == "medication" and not re.search(
            r"\b(?:SSRIs?|SNRIs?|antidepressants?|medications?|benzodiazepines?|pharmacological treatments?)\b",
            sentence, re.I,
        ):
            continue
        # PubMed abstracts sometimes define "panic disorder (PD)" once and
        # use only "PD" in a later finding. Resolve that *source-defined*
        # abbreviation for clinical-context matching without changing or
        # synthesizing the exact sentence returned to the user. Bare "PD"
        # remains ambiguous (e.g. Parkinson's disease) and is not accepted.
        context_sentence = sentence
        if (
            "panic_disorder" in analysis.anxiety_subtypes
            and re.search(r"\bPD\b", sentence)
            and re.search(r"\bpanic disorder\s*\(\s*PD\s*\)", hit.chunk.text, re.I)
        ):
            context_sentence = f"{sentence} panic disorder"
        if not supports_explicit_clinical_context(
            analysis.anxiety_subtypes,
            analysis.treatments,
            analysis.population,
            analysis.outcomes,
            analysis.comorbidities,
            [],
            context_sentence,
        ):
            continue
        sentence_populations = extract_population_concepts(sentence)
        if evidence_populations and not (sentence_populations & evidence_populations):
            # General government guidance can state unqualified advice in one
            # sentence and discuss adults/children elsewhere in the chunk.
            # Keep strict study-population rules and explicit user population.
            if not (
                hit.chunk.evidence_level == EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION
                and not requested_populations
                and not sentence_populations
            ):
                continue
        if requested_populations and sentence_populations and not (
            requested_populations & sentence_populations
        ):
            continue
        sentence_tokens = set(content_tokens(sentence))
        if query_tokens and not query_tokens.intersection(sentence_tokens) and not analysis.treatments:
            continue
        if "anxiety" in query_tokens and not (
            "anxiety" in sentence_tokens or extract_subtype_concepts(sentence)
        ):
            continue
        eligible.append(sentence)

    if not eligible:
        return None

    def score(sentence: str) -> tuple[bool, float, int]:
        overlap = len(query_tokens & set(content_tokens(sentence))) / max(1, len(query_tokens))
        if analysis.intent == QueryIntent.DIAGNOSIS:
            # Prefer a self-contained diagnostic assessment statement over
            # a shorter isolated criterion when both are available.
            has_context = bool(re.search(
                r"\b(?:worr(?:y|ying)|behavioral changes|behavioural changes|"
                r"clinical assessment|physical exam|interfering with daily life)\b",
                sentence, re.I,
            ))
            return has_context, overlap, -len(sentence)
        return requires_finding and bool(_RESULT_STATEMENT.search(sentence)), overlap, -len(sentence)

    return max(eligible, key=score) if query_tokens else eligible[0]
