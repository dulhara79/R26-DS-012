# =============================================================================
# TC-WPN: Episode Sampler (With Clinical Augmentation)
# =============================================================================
# Purpose : Converts processed .pkl note lists into N-way K-shot episodes
#           for meta-training and few-shot evaluation. Features on-the-fly
#           clinically safe data augmentation to prevent dimensional collapse.
# =============================================================================

import pickle
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import numpy as np
from transformers import AutoTokenizer

from config.settings import MIMIC_PROCESSED_PKL_DIR

# =============================================================================
# CONFIG
# =============================================================================


@dataclass
class SamplerConfig:
    PROCESSED_DIR: str = MIMIC_PROCESSED_PKL_DIR
    N_WAY: int = 2  # binary: anxiety vs control
    K_SHOT: int = 5  # support examples per class
    N_QUERY: int = 15  # query examples per class

    # Patient leakage guard — support and query never share a patient
    ENFORCE_PATIENT_SEPARATION: bool = False

    # Augmentation
    USE_AUGMENTATION: bool = False  # Set to False for Test/Val sets!
    TOKENIZER_NAME: str = "emilyalsentzer/Bio_ClinicalBERT"
    MAX_LENGTH: int = 512
    WINDOW_OVERLAP: int = 128


SCFG = SamplerConfig()

# =============================================================================
# CLINICAL AUGMENTER (NEW)
# =============================================================================


class ClinicalAugmenter:
    """
    Safely augments clinical text to increase linguistic variance
    for Few-Shot support sets without destroying ground truth labels.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        print(f"  [Augmenter] Loading Tokenizer: {SCFG.TOKENIZER_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(SCFG.TOKENIZER_NAME)

        self.protected_pattern = re.compile(
            r"\b(no_.*?|not|denies|anxiety|panic|gad|ptsd|ocd|phobia|"
            r"depress|suicid|homicid)\b",
            re.IGNORECASE,
        )

    def _safe_sentence_shuffle(self, text: str) -> str:
        sections = text.split("\n\n")
        augmented_sections = []

        for section in sections:
            sentences = [s.strip() for s in re.split(r"\.\s+", section) if s.strip()]
            if len(sentences) > 2:
                header = sentences[0]
                body = sentences[1:]
                self.rng.shuffle(body)
                augmented_sections.append(header + ". " + ". ".join(body) + ".")
            else:
                augmented_sections.append(section)

        return "\n\n".join(augmented_sections)

    def _sliding_window_tokenize(self, text: str) -> dict:
        stride = SCFG.MAX_LENGTH - SCFG.WINDOW_OVERLAP - 2
        raw_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]

        if len(raw_ids) <= stride:
            enc = self.tokenizer(
                text, max_length=SCFG.MAX_LENGTH, padding="max_length", truncation=True
            )
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "n_chunks": 1,
                "raw_token_count": len(raw_ids),
            }

        chunk_ids, chunk_masks = [], []
        for start in range(0, len(raw_ids), stride):
            chunk = raw_ids[start : start + stride]
            chunk_text = self.tokenizer.decode(chunk, skip_special_tokens=True)
            enc = self.tokenizer(
                chunk_text,
                max_length=SCFG.MAX_LENGTH,
                padding="max_length",
                truncation=True,
            )
            chunk_ids.append(enc["input_ids"])
            chunk_masks.append(enc["attention_mask"])
            if start + stride >= len(raw_ids):
                break

        return {
            "input_ids": chunk_ids,
            "attention_mask": chunk_masks,
            "n_chunks": len(chunk_ids),
            "raw_token_count": len(raw_ids),
        }

    def augment_note(
        self, parent_note: "EpisodeNote", new_note_id: str
    ) -> "EpisodeNote":
        original_text = parent_note.cleaned_text
        augmented_text = self._safe_sentence_shuffle(original_text)
        tokens = self._sliding_window_tokenize(augmented_text)

        return EpisodeNote(
            note_id=new_note_id,
            subject_id=parent_note.subject_id,
            label=parent_note.label,
            weight=parent_note.weight,  # 🟢 Ensure weight survives augmentation
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            n_chunks=tokens["n_chunks"],
            raw_token_count=tokens["raw_token_count"],
            note_timestamp=parent_note.note_timestamp,
            visit_number=parent_note.visit_number,
            days_since_first_visit=parent_note.days_since_first_visit,
            days_since_last_visit=parent_note.days_since_last_visit,
            total_visits=parent_note.total_visits,
            note_age_days=parent_note.note_age_days,
            section_quality=parent_note.section_quality,
            cleaned_text=augmented_text,
        )


# =============================================================================
# DATA LOADING
# =============================================================================


# 🟢 FIX: Accept 'filename' directly instead of 'split'
def load_split(processed_dir: str, filename: str) -> list:
    path = f"{processed_dir}/{filename}"
    with open(path, "rb") as f:
        records = pickle.load(f)
    print(f"  Loaded {filename}: {len(records):,} notes")
    return records


def index_by_label(records: list) -> dict:
    index = defaultdict(list)
    for r in records:
        index[r["label"]].append(r)
    return dict(index)


def index_by_label_and_patient(records: list) -> dict:
    index = defaultdict(lambda: defaultdict(list))
    for r in records:
        index[r["label"]][r["subject_id"]].append(r)
    return {label: dict(patients) for label, patients in index.items()}


# =============================================================================
# EPISODE DATA STRUCTURES
# =============================================================================


@dataclass
class EpisodeNote:
    note_id: str
    subject_id: str
    label: int
    weight: float  # 🟢 ADDED: Required for Model B confidence weighting
    input_ids: list
    attention_mask: list
    n_chunks: int
    raw_token_count: int
    note_timestamp: str
    visit_number: int
    days_since_first_visit: float
    days_since_last_visit: float
    total_visits: int
    note_age_days: float
    section_quality: float
    cleaned_text: str

    @classmethod
    def from_record(cls, record: dict) -> "EpisodeNote":
        return cls(
            note_id=record["note_id"],
            subject_id=record["subject_id"],
            label=record["label"],
            weight=record.get("weight", 1.0),  # 🟢 EXTRACTED: Safely defaults to 1.0
            input_ids=record["input_ids"],
            attention_mask=record["attention_mask"],
            n_chunks=record["n_chunks"],
            raw_token_count=record.get("raw_token_count", 0),
            note_timestamp=record.get("note_timestamp", ""),
            visit_number=record.get("visit_number", 1),
            days_since_first_visit=record.get("days_since_first_visit", 0.0),
            days_since_last_visit=record.get("days_since_last_visit", 0.0),
            total_visits=record.get("total_visits", 1),
            note_age_days=record.get("note_age_days", 0.0),
            section_quality=record.get("section_quality", 0.5),
            cleaned_text=record.get("cleaned_text", ""),
        )

    def get_input_tensors(self) -> tuple:
        ids = self.input_ids
        mask = self.attention_mask
        if isinstance(ids[0], list):
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            mask_tensor = torch.tensor(mask, dtype=torch.long)
        else:
            ids_tensor = torch.tensor([ids], dtype=torch.long)
            mask_tensor = torch.tensor([mask], dtype=torch.long)
        return ids_tensor, mask_tensor

    def get_temporal_metadata(self) -> dict:
        return {
            "note_timestamp": self.note_timestamp,
            "visit_number": self.visit_number,
            "days_since_first_visit": self.days_since_first_visit,
            "days_since_last_visit": self.days_since_last_visit,
            "total_visits": self.total_visits,
            "note_age_days": self.note_age_days,
            "section_quality": self.section_quality,
        }


@dataclass
class Episode:
    support: dict  # { label: [EpisodeNote] }
    query: dict  # { label: [EpisodeNote] }
    classes: list  # e.g. [0, 1]

    def all_query_notes(self) -> list:
        result = []
        for idx, label in enumerate(self.classes):
            for note in self.query[label]:
                result.append((note, idx))
        return result

    def summary(self) -> str:
        n_support = sum(len(v) for v in self.support.values())
        n_query = sum(len(v) for v in self.query.values())
        return f"Episode | {len(self.classes)}-way | support={n_support} | query={n_query} | classes={self.classes}"


# =============================================================================
# EPISODE SAMPLER
# =============================================================================


class EpisodeSampler:
    def __init__(
        self, processed_dir: str, split: str, filename: str = None, seed: int = 42
    ):
        self.split = split
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Fallback to default name if exact filename isn't provided
        if filename is None:
            filename = f"{split}_notes.pkl"

        print(f"\nLoading EpisodeSampler ({split}) from {filename}...")

        # 🟢 FIX: Pass 'filename' to load_split, NOT 'split'!
        records = load_split(processed_dir, filename)

        self._label_index = index_by_label(records)
        self._patient_index = index_by_label_and_patient(records)
        self.available_labels = sorted(self._label_index.keys())

        self.augmenter = None
        if split == "train" and SCFG.USE_AUGMENTATION:
            self.augmenter = ClinicalAugmenter(seed=seed)

        for label, recs in self._label_index.items():
            label_name = "anxiety" if label == 1 else "control"
            n_patients = len(self._patient_index[label])
            print(
                f"  Label {label} ({label_name:7s}): {len(recs):,} notes | {n_patients:,} patients"
            )

        print(f"  Sampler ready. Available labels: {self.available_labels}")

    def sample_episode(
        self,
        n_way: int = SCFG.N_WAY,
        k_shot: int = SCFG.K_SHOT,
        n_query: int = SCFG.N_QUERY,
    ) -> Episode:
        episode_classes = self.rng.sample(self.available_labels, n_way)
        support = {}
        query = {}

        for label in episode_classes:
            if SCFG.ENFORCE_PATIENT_SEPARATION:
                s_notes, q_notes = self._sample_with_patient_separation(
                    label, k_shot, n_query
                )
            else:
                s_notes, q_notes = self._sample_simple(label, k_shot, n_query)

            s_ep_notes = [EpisodeNote.from_record(r) for r in s_notes]
            q_ep_notes = [EpisodeNote.from_record(r) for r in q_notes]

            if self.augmenter is not None:
                while len(s_ep_notes) < k_shot:
                    parent_note = self.rng.choice(s_ep_notes)
                    synth_id = f"{parent_note.note_id}_aug_{len(s_ep_notes)}"
                    synthetic_note = self.augmenter.augment_note(parent_note, synth_id)
                    s_ep_notes.append(synthetic_note)

                if len(s_ep_notes) == k_shot and self.rng.random() < 0.5:
                    idx_to_replace = self.rng.randint(0, k_shot - 1)
                    parent_note = s_ep_notes[idx_to_replace]
                    synth_id = f"{parent_note.note_id}_aug_var"
                    s_ep_notes[idx_to_replace] = self.augmenter.augment_note(
                        parent_note, synth_id
                    )

            support[label] = s_ep_notes[:k_shot]
            query[label] = q_ep_notes[:n_query]

        return Episode(support=support, query=query, classes=episode_classes)

    def _sample_with_patient_separation(
        self, label: int, k_shot: int, n_query: int
    ) -> tuple:
        patient_ids = list(self._patient_index[label].keys())
        self.rng.shuffle(patient_ids)

        support_pool, query_pool = [], []
        pool_toggle = "support"

        for pid in patient_ids:
            patient_notes = self._patient_index[label][pid]
            if pool_toggle == "support" and len(support_pool) < k_shot * 2:
                support_pool.extend(patient_notes)
                pool_toggle = "query"
            else:
                query_pool.extend(patient_notes)
                if len(support_pool) >= k_shot * 2 and len(query_pool) >= n_query * 2:
                    break

        if len(support_pool) < k_shot or len(query_pool) < n_query:
            # 🟢 Evaluator Fix: Print warning so you know if data is too sparse
            print(f"⚠️ Patient separation fallback triggered for Label {label}.")
            return self._sample_simple(label, k_shot, n_query)

        s_notes = self.rng.sample(support_pool, min(k_shot, len(support_pool)))
        q_notes = self.rng.sample(query_pool, min(n_query, len(query_pool)))

        return s_notes, q_notes

    def _sample_simple(self, label: int, k_shot: int, n_query: int):
        all_notes = self._label_index[label]
        n_needed = k_shot + n_query

        # 🟢 Evaluator Fix: Removed the hardcoded 20 limit that was destroying diversity
        if len(all_notes) < n_needed:
            sampled = self.rng.choices(all_notes, k=n_needed)
        else:
            sampled = self.rng.sample(all_notes, n_needed)

        return sampled[:k_shot], sampled[k_shot:]

    def generate_episodes(
        self,
        n_episodes: int,
        n_way: int = SCFG.N_WAY,
        k_shot: int = SCFG.K_SHOT,
        n_query: int = SCFG.N_QUERY,
    ):
        for _ in range(n_episodes):
            yield self.sample_episode(n_way=n_way, k_shot=k_shot, n_query=n_query)


# =============================================================================
# COLLATE FUNCTION
# =============================================================================


def collate_episode(episode: Episode) -> dict:
    def pack_notes(notes: list) -> dict:
        ids_list, mask_list, temporal_list, label_list, weight_list = [], [], [], [], []
        for note in notes:
            ids, mask = note.get_input_tensors()
            ids_list.append(ids)
            mask_list.append(mask)
            temporal_list.append(note.get_temporal_metadata())
            label_list.append(note.label)
            weight_list.append(note.weight)  # 🟢 ADDED: Pack weights into lists

        return {
            "input_ids": ids_list,
            "attention_mask": mask_list,
            "temporal": temporal_list,
            "labels": label_list,
            "weights": weight_list,  # 🟢 EXPORTED: Expose to Kaggle
        }

    return {
        "support": {
            label: pack_notes(notes) for label, notes in episode.support.items()
        },
        "query": {label: pack_notes(notes) for label, notes in episode.query.items()},
        "classes": episode.classes,
    }
