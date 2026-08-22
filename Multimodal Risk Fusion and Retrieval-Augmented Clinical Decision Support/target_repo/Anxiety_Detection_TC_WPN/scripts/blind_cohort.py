#!/usr/bin/env python
"""
blind_cohort.py — Produce blinded variants of a finished cohort CSV.

Runs AFTER build_clean_cohort.py and BEFORE tokenize_cohort.py, so every
robustness arm inherits exactly the same patients and the same split
assignment. Only `text` changes.

    python scripts/blind_cohort.py \
        --cohort data/clean/psych_mimic4_notes.csv \
        --level dx_meds \
        --out data/clean/psych_mimic4_dx_meds_notes.csv

Then tokenise and plan episodes with the SAME --stem convention, and the
robustness comparison is a paired comparison over identical patient keys.

The script refuses to write an output whose row count, patient set, or split
assignment differs from the input. That refusal is the guarantee the paper
depends on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tcwpn.blinding import LEVELS, blind_text, count_hits, verify_blinded  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True,
                    help="notes CSV from build_clean_cohort.py")
    ap.add_argument("--level", default="dx_meds", choices=sorted(LEVELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-col", default="text")
    args = ap.parse_args()

    src = pd.read_csv(args.cohort)
    if args.text_col not in src.columns:
        raise SystemExit(f"no column {args.text_col!r} in {args.cohort}")

    before_hits = src[args.text_col].fillna("").map(
        lambda t: count_hits(t, args.level))

    out = src.copy()
    out[args.text_col] = out[args.text_col].fillna("").map(
        lambda t: blind_text(t, args.level))

    # -- post-conditions -----------------------------------------------------
    assert len(out) == len(src), "row count changed"
    assert set(out["subject_id"]) == set(src["subject_id"]), "patient set changed"
    if "split" in src.columns:
        pd.testing.assert_series_equal(
            out["split"].reset_index(drop=True),
            src["split"].reset_index(drop=True),
            check_names=False,
        )
    if "label" in src.columns:
        pd.testing.assert_series_equal(
            out["label"].reset_index(drop=True),
            src["label"].reset_index(drop=True),
            check_names=False,
        )

    check = verify_blinded(out[args.text_col].tolist(), args.level)
    if check["residual_notes"] != 0:
        raise SystemExit(
            f"BLINDING INCOMPLETE: {check['residual_notes']} notes still contain "
            f"blinded terms (first indices {check['residual_indices']}). "
            "Do not train on this file."
        )

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False)

    # -- report --------------------------------------------------------------
    grp = "label" if "label" in src.columns else "group"
    stats = {
        "level": args.level,
        "input": str(args.cohort),
        "output": str(outp),
        "n_notes": int(len(out)),
        "n_patients": int(src["subject_id"].nunique()),
        "terms_removed_total": int(before_hits.sum()),
        "mean_terms_removed_per_note_by_class": {
            str(k): round(float(v), 3)
            for k, v in before_hits.groupby(src[grp]).mean().items()
        },
        "notes_containing_at_least_one_term_by_class": {
            str(k): int(v)
            for k, v in (before_hits > 0).groupby(src[grp]).sum().items()
        },
        "residual_after_blinding": check["residual_notes"],
    }
    (outp.parent / f"{outp.stem}_blinding_report.json").write_text(
        json.dumps(stats, indent=2))

    print(json.dumps(stats, indent=2))
    print(f"\nwrote {outp}")
    print("Report the per-class term counts above in the paper: they quantify "
          "how much of the shortcut existed before blinding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
