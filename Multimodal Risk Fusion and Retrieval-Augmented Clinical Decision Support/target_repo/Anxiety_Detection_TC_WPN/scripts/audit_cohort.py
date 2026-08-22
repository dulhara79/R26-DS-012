"""
audit_cohort.py — Stage 2. Produce the evidence report for the paper.

    python -m scripts.audit_cohort --cohort data/clean/cohort_psych_mimic4.csv

This prints (and writes as JSON) the report a reviewer needs to satisfy
themselves that the benchmark is not contaminated. Three sections matter:

  * Patient overlap between splits           -> must be 0/0/0
  * Text-derived filtering declarations      -> must all be NO
  * Lexical shortcut prevalence per class    -> the honest number

The third section is not a pass/fail check. It quantifies how often the word
"anxiety" (and related terms) literally appears in each class. If the term
appears in 80% of case notes and 5% of control notes, then a high AUROC means
very little on its own, and the blinded evaluation becomes the headline result
rather than an appendix. Reporting this up front is far stronger than having a
reviewer discover it.

Author: Dulhara Kaushalya (IT22130648)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tcwpn import splits as S     # noqa: E402

# Terms whose literal presence would let a bag-of-words model shortcut the task.
SHORTCUT_TERMS = {
    "anxiety_terms": ["anxiety", "anxious", "panic", "phobia", "phobic",
                      "agoraphobia", "gad-7", "gad7"],
    "anxiolytic_meds": ["lorazepam", "alprazolam", "clonazepam", "diazepam",
                        "buspirone", "hydroxyzine", "temazepam", "oxazepam"],
    "ssri_snri": ["sertraline", "escitalopram", "fluoxetine", "paroxetine",
                  "citalopram", "fluvoxamine", "venlafaxine", "duloxetine"],
    "other_psych": ["psychiatry", "psychiatric", "depression", "depressed",
                    "bipolar", "ptsd", "schizophrenia"],
}


def term_rate(texts: pd.Series, terms) -> float:
    pat = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b",
                     flags=re.IGNORECASE)
    return float(texts.fillna("").str.contains(pat).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = Path(args.cohort)
    df = pd.read_csv(path, low_memory=False)
    report = {"cohort_file": str(path)}

    line = "=" * 66
    print(line)
    print("TC-WPN CLEAN COHORT AUDIT")
    print(f"{path.name}")
    print(line)

    # -- patients -------------------------------------------------------------
    pats = df.drop_duplicates("subject_id")
    arms = pats["arm"].value_counts().to_dict()
    print("\nPatients")
    print("-" * 24)
    for arm, n in sorted(arms.items()):
        print(f"  {arm:<24s} {n:>8,}")
    print(f"  {'TOTAL':<24s} {len(pats):>8,}")
    report["patients_by_arm"] = {k: int(v) for k, v in arms.items()}

    print("\nNotes")
    print("-" * 24)
    notes_by_arm = df["arm"].value_counts().to_dict()
    for arm, n in sorted(notes_by_arm.items()):
        print(f"  {arm:<24s} {n:>8,}")
    print(f"  {'TOTAL':<24s} {len(df):>8,}")
    report["notes_by_arm"] = {k: int(v) for k, v in notes_by_arm.items()}
    report["notes_per_patient_median"] = float(
        df.groupby("subject_id").size().median())

    # -- declarations ---------------------------------------------------------
    print("\nText-derived filtering")
    print("-" * 24)
    declarations = {
        "Positive keyword filtering": "NO",
        "Negative keyword filtering": "NO",
        "Text-derived test filtering": "NO",
        "Medication-derived labels": "NO",
        "Text-derived sample weights": "NO",
        "Class-differential note filtering": "NO",
    }
    for k, v in declarations.items():
        print(f"  {k:<36s} {v}")
    report["declarations"] = declarations
    print("  (These hold by construction: build_clean_cohort.py assigns labels")
    print("   and splits from structured tables before any text is read.)")

    # -- overlap --------------------------------------------------------------
    print("\nPatient overlap")
    print("-" * 24)
    overlaps = S.assert_disjoint(df)
    for k, v in overlaps.items():
        print(f"  {k.replace('_', ' / '):<36s} {v}")
    S.assert_no_duplicate_notes(df)
    n_cross = S.assert_no_duplicate_text_hashes(df)
    print(f"  {'duplicate note_id':<36s} 0")
    print(f"  {'identical text across splits':<36s} {n_cross}")
    report["overlaps"] = overlaps
    report["cross_split_identical_text"] = n_cross

    # -- split summary --------------------------------------------------------
    print("\nSplits")
    print("-" * 24)
    summ = S.split_summary(df)
    print(summ.to_string(index=False))
    report["split_summary"] = summ.to_dict(orient="records")

    # -- lexical shortcut -----------------------------------------------------
    print("\nLexical shortcut prevalence (fraction of notes containing term)")
    print("-" * 66)
    print(f"  {'group':<18s} {'case':>10s} {'control':>10s} {'ratio':>10s}")
    case_txt = df.loc[df["label"] == 1, "text"]
    ctrl_txt = df.loc[df["label"] == 0, "text"]
    shortcut = {}
    for group, terms in SHORTCUT_TERMS.items():
        r1, r0 = term_rate(case_txt, terms), term_rate(ctrl_txt, terms)
        ratio = (r1 / r0) if r0 > 0 else float("inf")
        shortcut[group] = {"case_rate": round(r1, 4),
                           "control_rate": round(r0, 4),
                           "ratio": None if r0 == 0 else round(ratio, 2)}
        rs = "inf" if r0 == 0 else f"{ratio:.2f}"
        print(f"  {group:<18s} {r1:>10.3f} {r0:>10.3f} {rs:>10s}")
    report["lexical_shortcut"] = shortcut
    print("\n  Interpretation: the larger the ratio, the more a bag-of-words")
    print("  model can solve this task without any clinical inference. Report")
    print("  these numbers in the paper and pair every headline result with")
    print("  its blinded counterpart.")

    # -- metadata that must not be used for filtering -------------------------
    if "anx_coded_this_adm" in df.columns:
        frac = float(df.loc[df["label"] == 1, "anx_coded_this_adm"].mean())
        print(f"\nCase notes from an admission where anxiety was coded: {frac:.1%}")
        print("  (reporting metadata only — never used to filter val/test)")
        report["case_notes_from_anxiety_admission"] = round(frac, 4)

    out = Path(args.out) if args.out else path.with_name(
        path.stem.replace("cohort_", "audit_report_") + ".json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")
    print(line)


if __name__ == "__main__":
    main()
