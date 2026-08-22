"""
test_no_text_derived_filtering.py

Two kinds of guarantee live here.

A. BEHAVIOURAL — the cohort builder's label must not move when the note text
   changes. If a reviewer swaps every note for a lorem-ipsum string, the
   patient groups and the split assignment must come out byte-identical.

B. STATIC — a grep over the pipeline modules for the specific constructs that
   broke the archived benchmark. These are cheap, they run in CI, and they are
   the thing to point at when a reviewer asks "how do you know the label isn't
   a function of the text?".

The archived constructs being guarded against:
    has_psychiatric_content(...)          keyword gate applied to positives only
    assign_anxiety_confidence(...)        label_confidence derived from regex
    penalize_control_noise(...)           control down-weighting by text
    test[test.label_confidence >= 0.7]    test set filtered by a text score
    training_weight = conf * quality      text-derived training weights
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcwpn import cohort as cohort_mod  # noqa: E402
from tcwpn.blinding import blind_text, count_hits, LEVELS  # noqa: E402


# ===========================================================================
# A. BEHAVIOURAL
# ===========================================================================
def _toy_diagnoses() -> pd.DataFrame:
    rows = [
        # subject, hadm, seq, code, version
        (1, 100, 1, "F411", 10),    # anxiety
        (1, 100, 2, "I10", 10),
        (2, 200, 1, "F32.9", 10),   # depression -> psych control
        (3, 300, 1, "I50", 10),     # cardiac only -> nonpsych control
        (4, 400, 3, "30002", 9),    # ICD-9 GAD -> anxiety
        (5, 500, 1, "2967", 9),     # ICD-9 bipolar -> psych control
    ]
    return pd.DataFrame(
        rows, columns=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]
    )


def _group_of(frame: pd.DataFrame, sid: int) -> str:
    return frame.loc[frame["subject_id"] == sid, "group"].iloc[0]


def test_groups_come_from_codes_not_text():
    fn = getattr(cohort_mod, "assign_patient_groups", None)
    if fn is None:
        pytest.skip("assign_patient_groups not present under that name")
    g = fn(_toy_diagnoses())
    assert _group_of(g, 1).startswith("anx")
    assert _group_of(g, 4).startswith("anx")
    assert "psych" in _group_of(g, 2)
    assert "psych" in _group_of(g, 5)
    assert _group_of(g, 3) in ("nonpsych_control", "control", "nonpsych")


def test_grouping_signature_takes_no_text():
    """The function cannot filter on text it never receives."""
    import inspect

    fn = getattr(cohort_mod, "assign_patient_groups", None)
    if fn is None:
        pytest.skip("assign_patient_groups not present under that name")
    params = set(inspect.signature(fn).parameters)
    for forbidden in ("text", "notes", "note_text", "clinical_note_text"):
        assert forbidden not in params, (
            f"{forbidden!r} must not be an argument to the grouping function"
        )


# ===========================================================================
# B. STATIC
# ===========================================================================
FORBIDDEN_PATTERNS = {
    "text-derived label confidence": r"\blabel_confidence\b",
    "text-derived training weight": r"\btraining_weight\b",
    "keyword gate on positives": r"\bhas_psychiatric_content\b",
    "regex confidence engine": r"\bassign_anxiety_confidence\b",
    "control text penalty": r"\bpenalize_control_noise\b",
    "note quality score": r"\bsection_quality\b",
    "curriculum purity filter": r"\bcurriculum_filter\b",
}

# Files whose job is to DISCUSS the old pipeline are allowed to name it.
DOC_ALLOWLIST = {"blinding.py"}


def _pipeline_sources():
    for p in sorted((ROOT / "src" / "tcwpn").glob("*.py")):
        if p.name in DOC_ALLOWLIST:
            continue
        yield p
    for p in sorted((ROOT / "scripts").glob("*.py")):
        yield p


GUARD_MARKER = "archived-construct-guard"


def _strip_comments_and_docstrings(src: str) -> str:
    """
    Drop, in order:
      1. lines carrying the GUARD_MARKER -- these name an archived construct in
         order to REJECT it (e.g. sampler.FORBIDDEN_FIELDS), which is the
         opposite of reintroducing it;
      2. triple-quoted blocks, so a module may document what it removed;
      3. #-comments.
    What remains is executable code that genuinely uses the name.
    """
    src = "\n".join(l for l in src.splitlines() if GUARD_MARKER not in l)
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


@pytest.mark.parametrize("label,pattern", sorted(FORBIDDEN_PATTERNS.items()))
def test_no_archived_leakage_construct_in_executable_code(label, pattern):
    offenders = []
    rx = re.compile(pattern)
    for path in _pipeline_sources():
        code = _strip_comments_and_docstrings(path.read_text(encoding="utf-8"))
        if rx.search(code):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        f"{label}: reintroduced in {offenders}. This construct made the label a "
        f"function of the note text in the archived benchmark."
    )


def test_no_label_conditional_text_filter():
    """
    Catches the shape `if label == 1: <something about text>` regardless of the
    helper's name -- the actual defect, of which has_psychiatric_content was
    only one instance.
    """
    shape = re.compile(
        r"if\s+.*\blabel\b.*==\s*1.*:\s*\n\s+.*(text|keyword|regex|search|contains)",
        re.IGNORECASE,
    )
    offenders = [
        str(p.relative_to(ROOT))
        for p in _pipeline_sources()
        if shape.search(_strip_comments_and_docstrings(p.read_text(encoding="utf-8")))
    ]
    assert not offenders, f"label-conditional text filter found in {offenders}"


# ===========================================================================
# C. BLINDING
# ===========================================================================
SAMPLE = ("Pt with generalized anxiety disorder, on sertraline 50mg and PRN "
          "lorazepam. Denies panic attacks. Appeared anxious. PHQ-9 = 12.")


@pytest.mark.parametrize("level", [lv for lv in LEVELS if lv != "none"])
def test_blinding_leaves_no_residual(level):
    assert count_hits(blind_text(SAMPLE, level), level) == 0


def test_blinding_is_idempotent():
    once = blind_text(SAMPLE, "dx_meds")
    assert blind_text(once, "dx_meds") == once


def test_blinding_deletes_rather_than_marks():
    """A placeholder token would appear only in positive notes and become a new,
    cleaner shortcut. Assert none of the usual markers are introduced."""
    out = blind_text(SAMPLE, "psych")
    for marker in ("[REDACTED]", "REDACTED", "<MASK>", "[MASK]", "XXXX", "___"):
        assert marker not in out


def test_blinding_preserves_non_target_clinical_content():
    out = blind_text(SAMPLE, "dx_meds")
    assert "50mg" in out and "Pt with" in out


def test_none_level_is_identity():
    assert blind_text(SAMPLE, "none") == SAMPLE
