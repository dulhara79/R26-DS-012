"""
blinding.py — Lexical shortcut controls.

WHY THIS IS A SEPARATE, EARLY STAGE
===================================
In the archived work the blinded and unblinded runs were produced by different
notebooks over different record sets, so the two AUROCs were not comparable and
the conclusion ("TC-WPN relies more on lexical shortcuts than ProtoNet") rested
on a mismatched comparison.

Here, blinding is a pure text transform applied to the SAME cohort CSV after
splits are already fixed. Every arm therefore has:

    identical patients, identical split assignment, identical episode plans.

Only the characters inside `text` differ. That is the only way the robustness
table can be read as an effect of the shortcut rather than of the sample.

LEVELS
------
    none        no transform (reference arm)
    dx          anxiety/panic/phobia diagnosis vocabulary removed
    meds        anxiolytic / antidepressant drug names removed
    dx_meds     both                                    <- MAIN ROBUSTNESS ARM
    psych       dx + meds + broad psychiatric vocabulary (strictest)

DELETION, NOT SUBSTITUTION
--------------------------
Terms are deleted rather than replaced with a marker. A marker such as
[REDACTED] appears almost exclusively in positive notes, so its presence
becomes a new, cleaner shortcut and the "blinded" score goes UP. Deletion
leaves no positional cue. This is checked by the TF-IDF probe: if TF-IDF on
blinded text stays high, blinding did not work and the term list needs
extending -- report the probe value in the paper either way.

RESIDUAL LEAKAGE IS EXPECTED
----------------------------
Blinding cannot remove paraphrase ("patient appeared on edge", "reports
restlessness and racing thoughts"). The claim supported by this module is
narrow and should be written narrowly: performance without the explicit
diagnosis vocabulary, not performance without any anxiety signal.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# TERM LISTS
# Ordered longest-first at compile time so that multi-word terms are removed
# before their constituent words.
# ---------------------------------------------------------------------------
DX_TERMS = [
    "generalized anxiety disorder", "generalised anxiety disorder",
    "social anxiety disorder", "separation anxiety disorder",
    "panic disorder", "panic attacks", "panic attack",
    "anxiety disorder", "anxiety disorders", "anxiety state", "anxiety states",
    "social anxiety", "separation anxiety", "situational anxiety",
    "anxiolysis", "anxiolytic", "anxiolytics",
    "anxieties", "anxiety", "anxiously", "anxious",
    "panicky", "panicked", "panic",
    "agoraphobia", "agoraphobic",
    "phobias", "phobia", "phobic",
    "gad-7", "gad7", "gad",
    "nervousness", "nervous",
    "worries", "worried", "worrying", "worry",
    "restlessness", "restless",
]

MED_TERMS = [
    # SSRI
    "sertraline", "zoloft", "escitalopram", "lexapro", "fluoxetine", "prozac",
    "paroxetine", "paxil", "citalopram", "celexa", "fluvoxamine", "luvox",
    # SNRI
    "venlafaxine", "effexor", "duloxetine", "cymbalta", "desvenlafaxine", "pristiq",
    # azapirone
    "buspirone", "buspar",
    # benzodiazepines
    "lorazepam", "ativan", "clonazepam", "klonopin", "alprazolam", "xanax",
    "diazepam", "valium", "oxazepam", "temazepam", "restoril", "chlordiazepoxide",
    "midazolam", "versed",
    # other
    "hydroxyzine", "vistaril", "atarax", "pregabalin", "lyrica",
    "propranolol", "inderal", "mirtazapine", "remeron", "trazodone",
    "benzodiazepine", "benzodiazepines", "benzo", "benzos",
    "ssri", "ssris", "snri", "snris",
]

PSYCH_TERMS = [
    "psychiatric", "psychiatry", "psychiatrist", "psych consult",
    "depression", "depressed", "depressive", "dysthymia",
    "phq-9", "phq9", "phq",
    "psychotherapy", "cognitive behavioral therapy", "cognitive behavioural therapy",
    "cognitive behavioral", "cognitive behavioural", "cbt",
    "mental health", "mental status", "mood disorder", "affective disorder",
    "axis i", "axis 1", "dsm-5", "dsm-iv", "dsm",
    "ptsd", "post-traumatic stress", "posttraumatic stress",
    "obsessive compulsive", "ocd",
    "bipolar", "schizophrenia", "schizoaffective",
]

LEVELS: dict[str, list[str]] = {
    "none": [],
    "dx": DX_TERMS,
    "meds": MED_TERMS,
    "dx_meds": DX_TERMS + MED_TERMS,
    "psych": DX_TERMS + MED_TERMS + PSYCH_TERMS,
}


def _compile(terms: list[str]) -> re.Pattern | None:
    """
    One alternation, longest term first, word-boundary anchored, case
    insensitive. Hyphens inside terms are matched literally; `\b` still works
    because the terms begin and end with word characters.
    """
    if not terms:
        return None
    uniq = sorted(set(t.strip().lower() for t in terms if t.strip()),
                  key=lambda t: (-len(t), t))
    body = "|".join(re.escape(t) for t in uniq)
    return re.compile(rf"\b(?:{body})\b", flags=re.IGNORECASE)


_CACHE: dict[str, re.Pattern | None] = {}


def get_pattern(level: str) -> re.Pattern | None:
    if level not in LEVELS:
        raise ValueError(f"unknown blinding level {level!r}; "
                         f"choose from {sorted(LEVELS)}")
    if level not in _CACHE:
        _CACHE[level] = _compile(LEVELS[level])
    return _CACHE[level]


_WS = re.compile(r"\s+")
_ORPHAN_PUNCT = re.compile(r"\s+([,.;:])")


def blind_text(text: str, level: str = "dx_meds") -> str:
    """Delete every matching term, then tidy the whitespace it left behind."""
    pat = get_pattern(level)
    if pat is None or not isinstance(text, str):
        return text if isinstance(text, str) else ""
    out = pat.sub(" ", text)
    out = _ORPHAN_PUNCT.sub(r"\1", out)
    return _WS.sub(" ", out).strip()


def count_hits(text: str, level: str = "dx_meds") -> int:
    """How many blinded terms a note contained. Used for the audit table:
    report hits per class BEFORE blinding, and confirm zero AFTER."""
    pat = get_pattern(level)
    if pat is None or not isinstance(text, str):
        return 0
    return len(pat.findall(text))


def verify_blinded(texts, level: str = "dx_meds") -> dict:
    """
    Post-condition check. `residual_notes` must be 0; if it is not, a term in
    the list is being reintroduced by the whitespace tidy or a boundary case
    is being missed, and the run must not proceed.
    """
    residual = [i for i, t in enumerate(texts) if count_hits(t, level) > 0]
    return {
        "level": level,
        "n_notes": len(texts),
        "residual_notes": len(residual),
        "residual_indices": residual[:20],
    }
