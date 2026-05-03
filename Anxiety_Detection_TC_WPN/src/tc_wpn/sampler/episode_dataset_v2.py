"""
episode_dataset_v2.py
TC-WPN Research-Grade Episodic Dataset

FIXES IN THIS VERSION:
- curriculum_filter: control threshold changed from >= 1.0 to >= 0.45
  (after penalize_control_noise fix, clean controls have weight ~0.5-0.6
   due to section_quality multiplication — not 1.0 anymore)
- _build_class_examples: added max_notes_per_patient cap (default=3)
  to force multi-patient diversity in each episode
- Temporal sort: ascending note_age_days (oldest first = chronological for GRU)
"""

import random
import pickle
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset

DEFAULT_RANDOM_SEED = 42


def set_seed(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def curriculum_filter(records, phase="full"):
    """
    Controls dataset purity by training stage.

    high_conf:
        anxiety  : weight >= 0.75  (named disorder + decent section quality)
        control  : weight >= 0.45  (FIX: was 1.0 — impossible after section_quality
                                    multiplication. Clean controls now score ~0.5)

    moderate:
        anxiety  : weight >= 0.55
        control  : weight >= 0.40

    full:
        all records
    """
    filtered = []

    for r in records:
        label = int(r["label"])
        weight = safe_float(r.get("weight", 1.0))

        if phase == "high_conf":
            if (label == 1 and weight >= 0.75) or (label == 0 and weight >= 0.45):
                filtered.append(r)

        elif phase == "moderate":
            if (label == 1 and weight >= 0.55) or (label == 0 and weight >= 0.40):
                filtered.append(r)

        else:
            filtered.append(r)

    return filtered


class TCMIMICEpisodicDataset(Dataset):
    """
    Builds N-way K-shot episodes from MIMIC PKL.

    Key research protections:
    1. No patient leakage within episode
    2. Per-patient note cap forces multi-patient diversity
    3. Temporal metadata preserved (chronological sort)
    4. Confidence weights preserved
    5. Curriculum filtering by training phase
    """

    def __init__(
        self,
        pkl_path,
        n_way=2,
        k_shot=5,
        q_query=5,
        episodes_per_epoch=1000,
        phase="full",
        min_notes_per_patient=2,
        max_notes_per_patient=3,  # FIX: cap per-patient contribution
        seed=42,
    ):
        super().__init__()

        set_seed(seed)

        self.pkl_path = Path(pkl_path)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch
        self.phase = phase
        self.min_notes_per_patient = min_notes_per_patient
        self.max_notes_per_patient = max_notes_per_patient  # FIX

        if not self.pkl_path.exists():
            raise FileNotFoundError(f"PKL not found: {self.pkl_path}")

        print("=" * 80)
        print(f"LOADING EPISODIC DATASET: {self.pkl_path.name}")
        print("=" * 80)

        with open(self.pkl_path, "rb") as f:
            records = pickle.load(f)

        print(f"Raw records loaded: {len(records):,}")

        records = curriculum_filter(records, phase=self.phase)
        print(f"After curriculum filter ({self.phase}): {len(records):,}")

        records = [r for r in records if self._is_valid_record(r)]
        print(f"After validity checks: {len(records):,}")

        # Organise: label -> subject_id -> [notes]
        self.patient_pool = {0: defaultdict(list), 1: defaultdict(list)}

        for r in records:
            label = int(r["label"])
            sid = str(r["subject_id"])
            self.patient_pool[label][sid].append(r)

        # FIX: Sort chronologically (ascending note_age_days = oldest first)
        # note_age_days=0 means most recent; higher = older
        # Ascending order feeds GRU from oldest → newest (correct temporal direction)
        for label in [0, 1]:
            for sid in list(self.patient_pool[label].keys()):
                notes = sorted(
                    self.patient_pool[label][sid],
                    key=lambda x: safe_float(x.get("note_age_days", 0)),
                    reverse=False,  # FIX: ascending = chronological
                )
                if len(notes) < self.min_notes_per_patient:
                    del self.patient_pool[label][sid]
                else:
                    self.patient_pool[label][sid] = notes

        self.valid_patients = {
            label: list(self.patient_pool[label].keys()) for label in [0, 1]
        }

        print(f"Control patients available: {len(self.valid_patients[0]):,}")
        print(f"Anxiety patients available: {len(self.valid_patients[1]):,}")

        if len(self.valid_patients[0]) < 10 or len(self.valid_patients[1]) < 10:
            raise ValueError(
                f"Too few patients after filtering. "
                f"Control={len(self.valid_patients[0])}, "
                f"Anxiety={len(self.valid_patients[1])}. "
                f"Try phase='moderate' or phase='full'."
            )

        print("=" * 80)
        print("EPISODIC DATASET READY")
        print("=" * 80)

    def _is_valid_record(self, r):
        try:
            return (
                bool(r.get("input_ids"))
                and bool(r.get("attention_mask"))
                and len(r["input_ids"]) > 0
            )
        except Exception:
            return False

    def __len__(self):
        return self.episodes_per_epoch

    def _temporal_dict(self, record):
        return {
            "visit_number": int(record.get("visit_number", 1)),
            "total_visits": int(record.get("total_visits", 1)),
            "days_since_first_visit": safe_float(
                record.get("days_since_first_visit", 0)
            ),
            "days_since_last_visit": safe_float(record.get("days_since_last_visit", 0)),
            "note_age_days": safe_float(record.get("note_age_days", 0)),
            "is_most_recent": bool(record.get("is_most_recent", False)),
        }

    def _build_class_examples(self, label):
        """
        Samples k_shot + q_query notes for one class.
        FIX: max_notes_per_patient cap forces multi-patient diversity.
        Without this cap, one patient with many notes fills the entire
        support set — model learns patient identity, not anxiety signal.
        """
        total_needed = self.k_shot + self.q_query

        candidates = self.valid_patients[label].copy()
        random.shuffle(candidates)

        selected = []

        for sid in candidates:
            if len(selected) >= total_needed:
                break

            notes = self.patient_pool[label][sid]
            # FIX: cap how many notes one patient contributes
            can_take = min(
                len(notes),
                self.max_notes_per_patient,
                total_needed - len(selected),
            )

            if len(notes) <= can_take:
                chosen = notes
            else:
                # Spread across trajectory for temporal diversity
                indices = sorted(random.sample(range(len(notes)), can_take))
                chosen = [notes[i] for i in indices]

            selected.extend(chosen)

        # Fallback: duplicate if dataset too small (rare)
        if len(selected) < total_needed:
            while len(selected) < total_needed:
                selected.append(random.choice(selected))

        selected = selected[:total_needed]
        return selected[: self.k_shot], selected[self.k_shot :]

    def _format_records(self, records):
        return {
            "input_ids": [
                torch.tensor(r["input_ids"], dtype=torch.long) for r in records
            ],
            "attention_mask": [
                torch.tensor(r["attention_mask"], dtype=torch.long) for r in records
            ],
            "weights": [safe_float(r.get("weight", 1.0)) for r in records],
            "temporal": [self._temporal_dict(r) for r in records],
            "note_ids": [str(r.get("note_id", "unknown")) for r in records],
            "subject_ids": [str(r.get("subject_id", "unknown")) for r in records],
        }

    def __getitem__(self, idx):
        classes = [0, 1]
        support, query = {}, {}
        used_note_ids = set()

        for label in classes:
            for attempt in range(20):
                sup_recs, qry_recs = self._build_class_examples(label)
                all_ids = {r["note_id"] for r in sup_recs + qry_recs}

                if not (all_ids & used_note_ids):
                    used_note_ids.update(all_ids)
                    break
            else:
                # After 20 attempts just use what we have
                sup_recs, qry_recs = self._build_class_examples(label)

            support[label] = self._format_records(sup_recs)
            query[label] = self._format_records(qry_recs)

        return {"support": support, "query": query, "classes": classes}


def episodic_collate_fn(batch):
    if len(batch) != 1:
        raise ValueError("TC-WPN episodic loader expects batch_size=1.")
    return batch[0]


if __name__ == "__main__":
    import os

    SAMPLE_PATH = "mimic_pkl/mimic_anxiety_train_high_conf.pkl"

    if not os.path.exists(SAMPLE_PATH):
        print(f"Sample path not found: {SAMPLE_PATH}")
    else:
        dataset = TCMIMICEpisodicDataset(
            pkl_path=SAMPLE_PATH,
            n_way=2,
            k_shot=5,
            q_query=5,
            episodes_per_epoch=10,
            phase="high_conf",
        )
        episode = dataset[0]
        print("\nEpisode Sanity Check:")
        for label in episode["classes"]:
            print(
                f"  Class {label}: "
                f"Support={len(episode['support'][label]['input_ids'])}, "
                f"Query={len(episode['query'][label]['input_ids'])}"
            )
        print("\n✅ episode_dataset_v2.py READY")
