"""
test_blinding_vocabularies.py

Two blinding vocabularies exist in this repository:

    src/tcwpn/blinding.py        levels: none, dx, meds, dx_meds, psych
    scripts/tokenize_cohort.py   levels: none, anxiety, meds, anx_meds, psych

Only the second one produces the pkls that models are actually evaluated on.
The first is used by scripts/blind_cohort.py, which is a reporting utility.

That divergence already cost a run: a notebook passed `--blind dx_meds` to
tokenize_cohort.py, argparse rejected it, no blinded pkl was written, and two
evaluate calls silently found nothing to load.

These tests do not force the lists to be identical -- they deliberately differ,
and merging them would silently change what a blinded experiment removes. They
pin the difference so it is documented, cannot drift further unnoticed, and can
be stated accurately in the paper's methods section.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcwpn.blinding import LEVELS as MODULE_LEVELS  # noqa: E402

TOKENIZE_SRC = (ROOT / "scripts" / "tokenize_cohort.py").read_text(encoding="utf-8")


def _extract_list(name: str) -> list[str]:
    """Pull a top-level list literal out of tokenize_cohort.py without importing
    it (it needs tqdm/transformers, which need not be installed to run tests)."""
    m = re.search(rf"^{name}\s*=\s*\[(.*?)\]", TOKENIZE_SRC, re.S | re.M)
    if not m:
        raise AssertionError(f"{name} not found in tokenize_cohort.py")
    return [t.lower() for t in re.findall(r'"([^"]+)"', m.group(1))]


def _extract_levels() -> set[str]:
    m = re.search(r"^BLIND_LEVELS\s*=\s*\{(.*?)\}", TOKENIZE_SRC, re.S | re.M)
    assert m, "BLIND_LEVELS not found"
    return set(re.findall(r'"([^"]+)"\s*:', m.group(1)))


TOK_ANXIETY = _extract_list("ANXIETY_TERMS")
TOK_MEDS = _extract_list("MED_TERMS")
TOK_LEVELS = _extract_levels()


# ===========================================================================
# The names a notebook may pass
# ===========================================================================
def test_tokenize_accepts_exactly_these_levels():
    assert TOK_LEVELS == {"none", "anxiety", "meds", "anx_meds", "psych"}


def test_the_name_that_broke_stage_b_is_still_not_accepted():
    """`dx_meds` is a blinding.py name. If someone later adds it as an alias in
    tokenize_cohort.py, this test should be deleted deliberately, not silently."""
    assert "dx_meds" not in TOK_LEVELS
    assert "dx_meds" in MODULE_LEVELS


def test_notebooks_only_use_accepted_levels():
    """
    Guards the exact failure mode: a notebook passing a level argparse will
    reject, producing a missing pkl instead of a loud error.

    Code cells only. A markdown cell may name the rejected level, because
    explaining why Stage B failed requires writing it down.
    """
    import json

    offenders = []
    for p in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = json.loads(p.read_text(encoding="utf-8"))
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            for level in re.findall(r"--blind\s+([a-z_]+)", src):
                if level not in TOK_LEVELS and not level.startswith("{"):
                    offenders.append(f"{p.name} cell {i}: --blind {level}")
    assert not offenders, offenders


# ===========================================================================
# The divergence, pinned
# ===========================================================================
def test_tokenize_anxiety_terms_omit_the_symptom_family():
    """
    tokenize_cohort's anxiety list is DIAGNOSIS vocabulary. blinding.py also
    strips symptom words. So `anx_meds` is diagnosis-and-drug-name blinding, and
    the paper must not describe it as removing all anxiety-related language.
    """
    symptom_words = {"worry", "worries", "worried", "worrying",
                     "restless", "restlessness", "nervous"}
    still_present = symptom_words - set(TOK_ANXIETY)
    assert still_present == symptom_words, (
        f"tokenize_cohort now strips some symptom words {symptom_words - still_present}; "
        "update the methods section, which currently claims diagnosis-term blinding"
    )


def test_tokenize_med_terms_omit_these_drugs():
    """Anxiolytics and sedating antidepressants that survive `meds` blinding.
    Any of them can still act as a lexical cue; list them as a limitation."""
    known_omissions = {"trazodone", "mirtazapine", "remeron", "propranolol",
                       "inderal", "midazolam", "versed", "chlordiazepoxide",
                       "desvenlafaxine", "pristiq"}
    still_omitted = known_omissions - set(TOK_MEDS)
    assert still_omitted == known_omissions, (
        f"tokenize_cohort now blinds {known_omissions - still_omitted}; the "
        "robustness arm changed, so previously computed blinded results are no "
        "longer comparable and must be re-run"
    )


def test_anx_meds_is_the_union_of_the_two_narrower_levels():
    m = re.search(r'"anx_meds"\s*:\s*([^,\n]+)', TOKENIZE_SRC)
    assert m and "ANXIETY_TERMS" in m.group(1) and "MED_TERMS" in m.group(1)


def test_psych_level_is_a_different_experiment_not_a_stricter_one():
    """
    The Stage A audit measured `other_psych` terms at case 0.589 vs control
    0.713 -- a ratio of 0.83, i.e. MORE common in controls. Blinding them
    removes a control-favouring cue. This test only records that the level
    exists and is broader; the interpretation is the point.
    """
    m = re.search(r'"psych"\s*:\s*([^,\n]+)', TOKENIZE_SRC)
    assert m and "PSYCH_TERMS" in m.group(1)


@pytest.mark.parametrize("level", sorted(MODULE_LEVELS))
def test_blinding_module_levels_are_self_consistent(level):
    from tcwpn.blinding import blind_text, count_hits

    sample = ("Pt with generalized anxiety disorder on sertraline, PRN lorazepam, "
              "denies panic attacks, appeared anxious and restless.")
    assert count_hits(blind_text(sample, level), level) == 0
