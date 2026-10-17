"""Conservative topic and personal-diagnosis safeguards for an anxiety evidence assistant.

This deterministic layer is NOT a clinically validated domain classifier.
It stops obvious unrelated requests, but clinical applicability still depends on
retrieval, evidence selection and grounding.
"""
from __future__ import annotations

import re


_ANXIETY_FOCUS = re.compile(
    r"\b(?:anxiety|anxious|panic|phobi(?:a|as|c)|agoraphobia|"
    r"worr(?:y|ied|ying)|fearful|fears?|stress|stressed|"
    r"gad(?:-7)?|cbt|icbt|mct|vret|"
    r"cognitive[- ]behaviou?ral therapy|metacognitive therapy|"
    r"exposure therapy|"
    r"ssris?|snris?|benzodiazepines?|"
    r"nice\s+cg113|mhgap)\b",
    re.I,
)
# Known unrelated *primary treatment targets*, including mixed sentences such
# as "broken ankle and I feel anxious". Differential/anxiety comorbidity is
# a broader research question and needs a richer scope policy.
_UNRELATED_TARGET = re.compile(
    r"\b(?:broken|fractured?)\s+(?:ankle|arm|leg|wrist|bone|foot)\b"
    r"|\b(?:ankle fracture|kidney stones?|toothache)\b"
    r"|\b(?:how (?:do|can) i|instructions for)\s+(?:cook|bake)\b",
    re.I,
)
_PERSONAL_DIAGNOSIS = (
    re.compile(r"\bdiagnos(?:e|es|ed|ing|is)\s+(?:me|myself)\b", re.I),
    re.compile(
        r"\b(?:can|could|will)\s+(?:you|care[- ]?anxrag)\s+diagnos(?:e|is)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:do|could|might)\s+i\s+have\s+(?:an?\s+)?"
        r"(?:(?:generalized|social|health|separation|specific)\s+)?"
        r"(?:anxiety|panic|phobia|agoraphobia|\w+\s+disorder)\b",
        re.I,
    ),
    re.compile(r"\bis this\s+(?:an?\s+)?(?:panic attack|anxiety disorder)\b", re.I),
)


def is_anxiety_information_query(question: str) -> bool:
    """Conservatively permit clinician evidence questions with anxiety context."""
    if _UNRELATED_TARGET.search(question):
        return False
    return bool(_ANXIETY_FOCUS.search(question))


def is_personal_diagnosis_request(question: str) -> bool:
    """Detect clear personalized diagnosis requests, not clinician guideline queries."""
    return any(pattern.search(question) for pattern in _PERSONAL_DIAGNOSIS)
