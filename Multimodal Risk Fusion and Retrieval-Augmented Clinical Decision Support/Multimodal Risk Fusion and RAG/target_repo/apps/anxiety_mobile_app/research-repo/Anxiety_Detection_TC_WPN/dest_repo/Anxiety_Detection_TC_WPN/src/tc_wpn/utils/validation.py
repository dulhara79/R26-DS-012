import pickle
import os
from config.settings import MIMIC_PROCESSED_PKL_DIR

for split in ["train", "val", "test"]:
    path = os.path.join(MIMIC_PROCESSED_PKL_DIR, f"{split}_notes.pkl")
    if not os.path.exists(path):
        print(f"Skipping {split}: {path} not found.")
        continue
    with open(path, "rb") as f:
        data = pickle.load(f)
    anxiety = sum(1 for r in data if r["label"] == 1)
    control = sum(1 for r in data if r["label"] == 0)
    print(f"{split:5s}: {len(data):,} | anxiety={anxiety:,} control={control:,}")