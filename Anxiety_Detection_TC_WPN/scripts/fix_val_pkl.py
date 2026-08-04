"""
fix_val_pkl.py  — PUBLICATION-SAFE VERSION
TC-WPN Research Pipeline
Author: Dulhara Kaushalya

WHAT CHANGED FROM THE OLD VERSION (critical fix):
==================================================
The old version upsampled anxiety records in both val and test PKL files
using random.choices() with replacement, then OVERWROTE the originals.

WHY THAT WAS WRONG:
  1. Test set upsampling with replacement means the same patient can appear
     2-3× in evaluation. Bootstrap CI computed over such a sample
     overestimates precision (dependent samples violate IID assumption).
  2. The "Real-World" label was no longer accurate — it had been rebalanced
     to 1:2, far from the actual clinical prevalence (~1:6 in MIMIC-IV).
  3. val PKL was also upsampled, meaning model selection (best checkpoint)
     was done on an inflated validation set — the best checkpoint may not
     be the true best under real-world conditions.

WHAT THIS VERSION DOES INSTEAD:
  - Leaves test_real_world PKL completely UNTOUCHED (true prevalence evaluation)
  - Leaves val_real_world PKL completely UNTOUCHED (honest model selection)
  - Creates a SEPARATE balanced_eval PKL (under-sampling controls, NOT
    over-sampling anxiety) for supplementary "ceiling" comparison only
  - Restores originals if old upsampled versions exist (by checking for backups)
  - All actions are clearly logged so the paper methods section is truthful

CORRECT EVALUATION STRATEGY FOR THE PAPER:
  Primary results   → test_real_world (true ~1:4–1:6 prevalence, no resampling)
  Secondary results → test_high_conf  (filtered but not resampled)
  Supplementary     → test_balanced_eval (under-sampled controls, clearly labelled)
"""

import pickle
import random
import shutil
from pathlib import Path

PKL_DIR = Path(__file__).resolve().parents[1] / "mimic_pkl"
random.seed(42)


# =============================================================================
# RESTORE ORIGINALS (if the old broken script already ran)
# =============================================================================
def restore_originals():
    """
    If fix_val_pkl.py (old version) already ran, it saved _original.pkl backups.
    We restore those so we start from ground truth.
    """
    for stem in ["mimic_anxiety_val_real_world", "mimic_anxiety_test_real_world"]:
        backup = PKL_DIR / f"{stem}_original.pkl"
        current = PKL_DIR / f"{stem}.pkl"
        if backup.exists():
            shutil.copy2(backup, current)
            print(f"  ✅ Restored original: {current.name}  (from {backup.name})")
        else:
            print(
                f"  ℹ  No backup found for {stem} — assuming current file is original."
            )


# =============================================================================
# CREATE BALANCED EVAL PKL (under-sample controls, DO NOT over-sample anxiety)
# This is SUPPLEMENTARY only — clearly named and never called "real-world"
# =============================================================================
def create_balanced_eval_pkl(
    source_pkl_path: Path,
    output_pkl_path: Path,
    target_ratio: int = 2,
    description: str = "",
):
    """
    Creates a balanced evaluation set by DOWN-SAMPLING controls only.
    Anxiety records are never duplicated.

    target_ratio: controls / anxiety  (default 2 → 1:2 ratio)

    This is appropriate for supplementary analysis showing model ceiling
    performance. It must NEVER be reported as the primary result.
    """
    print(f"\nCreating balanced eval set: {output_pkl_path.name}  ({description})")

    with open(source_pkl_path, "rb") as f:
        records = pickle.load(f)

    anxiety = [r for r in records if int(r["label"]) == 1]
    control = [r for r in records if int(r["label"]) == 0]

    print(
        f"  Source prevalence: anxiety={len(anxiety):,}  control={len(control):,}  "
        f"ratio=1:{len(control)//max(len(anxiety),1)}"
    )

    # DOWN-sample controls to target_ratio × anxiety count
    target_ctrl = min(len(control), len(anxiety) * target_ratio)
    ctrl_sampled = random.sample(control, target_ctrl)

    combined = anxiety + ctrl_sampled
    random.shuffle(combined)

    print(
        f"  Balanced:          anxiety={len(anxiety):,}  control={len(ctrl_sampled):,}  "
        f"ratio=1:{len(ctrl_sampled)//max(len(anxiety),1)}"
    )
    print(f"  NOTE: Controls were DOWN-sampled. Anxiety is UNCHANGED (no duplication).")

    with open(output_pkl_path, "wb") as f:
        pickle.dump(combined, f)
    print(f"  ✅ Saved → {output_pkl_path.name}")


# =============================================================================
# VERIFY INTEGRITY: confirm no duplicated subject_ids in a PKL
# =============================================================================
def verify_no_duplicates(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)

    subject_ids = [str(r.get("subject_id", "unknown")) for r in records]
    note_ids = [str(r.get("note_id", "unknown")) for r in records]

    dup_subjects = len(subject_ids) - len(set(subject_ids))
    dup_notes = len(note_ids) - len(set(note_ids))

    anxiety_count = sum(1 for r in records if int(r.get("label", 0)) == 1)
    control_count = len(records) - anxiety_count

    print(f"\n  Integrity check: {pkl_path.name}")
    print(
        f"    Records: {len(records):,}  |  Anxiety: {anxiety_count:,}  |  Control: {control_count:,}"
    )
    print(f"    Duplicate note_ids:    {dup_notes:,}")
    print(f"    Duplicate subject_ids: {dup_subjects:,}")

    if dup_notes > 0:
        print(
            f"    ❌ WARNING: {dup_notes:,} duplicate notes detected — this file was previously upsampled!"
        )
        return False
    else:
        print(f"    ✅ No duplicate notes — file is clean.")
        return True


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("TC-WPN PKL INTEGRITY FIX — PUBLICATION-SAFE VERSION")
    print("=" * 70)

    if not PKL_DIR.exists():
        print(f"❌ PKL directory not found: {PKL_DIR}")
        print("   Run convert_csv_to_pkl_v2.py first.")
        return

    # -------------------------------------------------------------------------
    # STEP 1: Restore originals if old broken script already ran
    # -------------------------------------------------------------------------
    print("\nSTEP 1: Checking for and restoring original (non-upsampled) PKL files...")
    restore_originals()

    # -------------------------------------------------------------------------
    # STEP 2: Verify the real-world files have no duplicates
    # -------------------------------------------------------------------------
    print("\nSTEP 2: Verifying PKL integrity (checking for upsampling artifacts)...")
    all_clean = True
    for fname in [
        "mimic_anxiety_val_real_world.pkl",
        "mimic_anxiety_test_real_world.pkl",
        "mimic_anxiety_test_high_conf.pkl",
    ]:
        p = PKL_DIR / fname
        if p.exists():
            ok = verify_no_duplicates(p)
            if not ok:
                all_clean = False
        else:
            print(f"\n  ⚠ Not found: {fname}")

    if not all_clean:
        print("\n❌ Some PKL files still contain duplicates.")
        print("   This means the original backup was also upsampled, OR")
        print("   the original CSV→PKL pipeline introduced duplicates.")
        print("   Re-run convert_csv_to_pkl_v2.py from the original CSVs.")
        return

    # -------------------------------------------------------------------------
    # STEP 3: Create separate BALANCED SUPPLEMENTARY eval sets
    # These use control DOWN-sampling only — anxiety never duplicated
    # Clearly named so they can never be confused with real-world sets
    # -------------------------------------------------------------------------
    print("\nSTEP 3: Creating balanced supplementary eval PKL files...")
    print("  (Controls are DOWN-sampled. Anxiety records are NEVER duplicated.)")

    for source_name, output_name, desc in [
        (
            "mimic_anxiety_val_real_world.pkl",
            "mimic_anxiety_val_balanced_supp.pkl",
            "val balanced supplementary",
        ),
        (
            "mimic_anxiety_test_real_world.pkl",
            "mimic_anxiety_test_balanced_supp.pkl",
            "test balanced supplementary",
        ),
    ]:
        src = PKL_DIR / source_name
        out = PKL_DIR / output_name
        if src.exists():
            create_balanced_eval_pkl(src, out, target_ratio=2, description=desc)
        else:
            print(f"\n  ⚠ Source not found: {source_name} — skipping.")

    # -------------------------------------------------------------------------
    # STEP 4: Final summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("✅ DONE")
    print("=" * 70)
    print("""
EVALUATION STRATEGY FOR YOUR PAPER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIMARY RESULTS (Table 2 in your paper):
  → mimic_anxiety_test_high_conf.pkl         (filtered, no resampling)
  → mimic_anxiety_test_real_world.pkl        (true prevalence, no resampling)
  These are the numbers you report as your main results.
  Report prevalence alongside AUROC so readers understand the setting.

SUPPLEMENTARY RESULTS (appendix/ablation):
  → mimic_anxiety_test_balanced_supp.pkl     (1:2, controls downsampled only)
  Clearly label this as "balanced supplementary evaluation".
  Note in the paper: "Anxiety samples are NOT augmented; controls are
  downsampled to 1:2 ratio for this supplementary comparison."

VAL SET (model selection only — never reported as results):
  → mimic_anxiety_val_real_world.pkl         (true prevalence, no resampling)
  The supplementary val_balanced_supp.pkl is for ablation/debugging only.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    main()
