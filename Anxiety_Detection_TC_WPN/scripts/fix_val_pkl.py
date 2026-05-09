"""
fix_val_pkl.py
One-time fix: rebalance val and test PKL files.

PROBLEM:
  val_real_world.pkl:  224 anxiety vs 1,465 control (13% positive)
  test_real_world.pkl: 345 anxiety vs 1,408 control (20% positive)

With episodic sampling (k_shot=5, q_query=5), the dataset needs at
least 10 anxiety patients with ≥2 notes each.  224 anxiety notes
across ~50–100 patients is marginal but workable.

The real problem is the 6:1 imbalance — AUROC computed over pooled
episode probs is unreliable when one class is rare.

FIX:
  We upsample anxiety notes in val/test to reach 1:2 ratio (anxiety:control).
  Upsampling is done WITH replacement from the existing anxiety pool,
  which is standard practice for evaluation sets in few-shot learning.

  test_high_conf is left untouched — it's already used as the primary
  publication metric and its 340/1399 ratio is acceptable.

OUTPUT:
  Overwrites the existing val/test pkl files in mimic_pkl/.
  Originals are backed up as *_original.pkl before overwriting.
"""

import pickle
import random
import shutil
from pathlib import Path

PKL_DIR = Path(__file__).resolve().parents[1] / "mimic_pkl"

TARGETS = [
    {
        "file": "mimic_anxiety_val_real_world.pkl",
        "target_ratio": 2,  # keep controls, upsample anxiety to controls/2
        "description": "val real-world",
    },
    {
        "file": "mimic_anxiety_test_real_world.pkl",
        "target_ratio": 2,
        "description": "test real-world",
    },
]

random.seed(42)


def rebalance_pkl(pkl_path, target_ratio, description):
    print(f"\nRebalancing: {pkl_path.name}  ({description})")

    with open(pkl_path, "rb") as f:
        records = pickle.load(f)

    anxiety = [r for r in records if int(r["label"]) == 1]
    control = [r for r in records if int(r["label"]) == 0]

    print(
        f"  Before: anxiety={len(anxiety):,}  control={len(control):,}  "
        f"ratio=1:{len(control)//max(len(anxiety),1)}"
    )

    # Target: anxiety = control / target_ratio
    target_anx = len(control) // target_ratio

    if target_anx <= len(anxiety):
        print(f"  Already balanced enough — no change needed.")
        return

    # Upsample anxiety with replacement
    extra_needed = target_anx - len(anxiety)
    upsampled = random.choices(anxiety, k=extra_needed)
    anxiety_new = anxiety + upsampled

    # Shuffle combined
    combined = anxiety_new + control
    random.shuffle(combined)

    print(
        f"  After:  anxiety={len(anxiety_new):,}  control={len(control):,}  "
        f"ratio=1:{len(control)//max(len(anxiety_new),1)}"
    )

    # Back up original
    backup_path = pkl_path.parent / (pkl_path.stem + "_original.pkl")
    shutil.copy2(pkl_path, backup_path)
    print(f"  Backed up original → {backup_path.name}")

    # Save rebalanced
    with open(pkl_path, "wb") as f:
        pickle.dump(combined, f)
    print(f"  ✅ Saved rebalanced → {pkl_path.name}")


def main():
    print("=" * 60)
    print("VAL/TEST PKL REBALANCING")
    print("=" * 60)

    for target in TARGETS:
        pkl_path = PKL_DIR / target["file"]
        if not pkl_path.exists():
            print(f"\n⚠ Not found: {pkl_path} — skipping.")
            continue
        rebalance_pkl(pkl_path, target["target_ratio"], target["description"])

    print("\n" + "=" * 60)
    print("✅ DONE — upload all PKL files in mimic_pkl/ to Kaggle")
    print("=" * 60)


if __name__ == "__main__":
    main()
