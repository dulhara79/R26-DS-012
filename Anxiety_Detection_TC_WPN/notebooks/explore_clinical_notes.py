# import pandas as pd

# df = pd.read_csv("mimic_processed/mimic_anxiety_train_high_conf.csv")

# print("=== LABEL CONFIDENCE (anxiety notes) ===")
# print(df[df["has_anxiety"] == 1]["label_confidence"].value_counts().sort_index())

# print("\n=== ANXIETY CONTEXT BREAKDOWN ===")
# print(df[df["has_anxiety"] == 1]["anxiety_context"].value_counts())

# print("\n=== CONTROL CONTAMINATION ===")
# contaminated = df[(df["has_anxiety"] == 0) & (df["training_weight"] < 0.9)]
# total_ctrl = (df["has_anxiety"] == 0).sum()
# print(
#     f"Contaminated controls: {len(contaminated):,} / {total_ctrl:,} ({100*len(contaminated)/total_ctrl:.1f}%)"
# )

# print("\n=== SOURCE TYPE BREAKDOWN ===")
# print(df["source_type"].value_counts())


from pathlib import Path
from config.settings import MIMIC_IV_NOTE_DATASET_PATH

note_path = Path(MIMIC_IV_NOTE_DATASET_PATH) / "note"
print("Available note files:")
for f in sorted(note_path.iterdir()):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name:40s} {size_mb:>8.0f} MB")
