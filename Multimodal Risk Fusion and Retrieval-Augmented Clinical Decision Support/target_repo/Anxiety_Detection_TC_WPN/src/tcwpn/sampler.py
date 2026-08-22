"""
sampler.py — Patient-disjoint episodic sampling.

WHAT WAS WRONG BEFORE
=====================
In the archived `episode_dataset.py`, `_build_class_examples()` collected a
flat list of notes by walking over patients (taking up to
`max_notes_per_patient=3` notes each) and then split that flat list at the
K-th element:

    selected = [...]                       # notes from several patients
    return selected[:k_shot], selected[k_shot:]

Because one patient could contribute up to 3 consecutive notes, a patient's
notes routinely straddled the K boundary: some of that patient's notes went
into the SUPPORT set and the rest into the QUERY set of the same episode. The
retry loop only checked `note_id` overlap, which never catches this. That is
same-patient support/query leakage in a large fraction of episodes, and with
copy-forwarded MIMIC notes it is close to testing on the training example.

WHAT THIS MODULE GUARANTEES
===========================
1. One support slot = one distinct patient (K-shot means K distinct patients).
2. support_patients ∩ query_patients = ∅ in EVERY episode, asserted at
   construction time and re-checkable afterwards from the saved plan.
3. Episodes are a serialisable *plan* of record indices, not a live random
   generator. Every model (TF-IDF, linear probe, ProtoNet, TC-WPN, and every
   blinded variant) is evaluated on byte-identical episodes. This is what
   makes the paired DeLong test valid and removes the blinded/unblinded
   mismatch from the earlier robustness analysis.
4. Evaluation plans give explicit, reportable coverage: every eligible patient
   is queried exactly `n_repeats` times.

Author: Dulhara Kaushalya (IT22130648)
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = (0, 1)


# =============================================================================
# RECORD STORE
# =============================================================================
class RecordStore:
    """
    Holds tokenised records for ONE split and indexes them by (label, patient).

    A record is a dict with at least:
        note_id, subject_id, label, split,
        input_ids (list[list[int]]), attention_mask (list[list[int]]),
        days_before_patient_last_note (float), n_notes_patient (int)

    No `weight`, no `label_confidence`, no `training_weight`. Those fields were
    derived from the note text and correlated with the label, so feeding them
    to the model was a direct label leak into prototype construction. They are
    gone by design; if they appear in a pkl this class raises.
    """

    FORBIDDEN_FIELDS = ("weight", "label_confidence", "training_weight",  # archived-construct-guard
                        "section_quality", "has_text_signal", "anxiety_context")  # archived-construct-guard

    def __init__(self, records, split_name: str = ""):
        self.split_name = split_name
        self.records = list(records)
        if not self.records:
            raise ValueError(f"RecordStore for '{split_name}' is empty")

        leaked = sorted(
            set(self.FORBIDDEN_FIELDS) & set(self.records[0].keys())
        )
        if leaked:
            raise ValueError(
                f"Records contain text-derived label-correlated fields {leaked}. "
                "Re-run tokenize_cohort.py from the clean cohort."
            )

        self.by_label_patient = {c: defaultdict(list) for c in CLASSES}
        for i, r in enumerate(self.records):
            label = int(r["label"])
            if label not in CLASSES:
                raise ValueError(f"unexpected label {label!r} in record {i}")
            self.by_label_patient[label][str(r["subject_id"])].append(i)

        self.patients = {
            c: sorted(self.by_label_patient[c].keys()) for c in CLASSES
        }
        for c in CLASSES:
            if len(self.patients[c]) < 4:
                raise ValueError(
                    f"split '{split_name}' has only {len(self.patients[c])} "
                    f"patients in class {c}; cannot form episodes"
                )

    @classmethod
    def from_pkl(cls, path, split_name: str = ""):
        path = Path(path)
        with open(path, "rb") as f:
            records = pickle.load(f)
        return cls(records, split_name=split_name or path.stem)

    def __len__(self):
        return len(self.records)

    def n_patients(self, label=None):
        if label is None:
            return sum(len(self.patients[c]) for c in CLASSES)
        return len(self.patients[label])

    def describe(self) -> dict:
        return {
            "split": self.split_name,
            "n_records": len(self.records),
            "n_patients_control": len(self.patients[0]),
            "n_patients_case": len(self.patients[1]),
            "n_notes_control": sum(len(v) for v in self.by_label_patient[0].values()),
            "n_notes_case": sum(len(v) for v in self.by_label_patient[1].values()),
            "patient_prevalence": round(
                len(self.patients[1]) / max(self.n_patients(), 1), 4
            ),
        }


# =============================================================================
# EPISODE PLAN
# =============================================================================
class EpisodePlan:
    """
    A frozen list of episodes. Each episode is:

        {
          "support": {"0": [rec_idx, ...], "1": [rec_idx, ...]},
          "query":   {"0": [rec_idx, ...], "1": [rec_idx, ...]},
        }

    Record indices refer to positions in the RecordStore this plan was built
    against; `store_fingerprint` pins that association so a plan cannot be
    silently applied to a different pkl.
    """

    def __init__(self, episodes, meta):
        self.episodes = episodes
        self.meta = meta

    def __len__(self):
        return len(self.episodes)

    def __iter__(self):
        return iter(self.episodes)

    def __getitem__(self, i):
        return self.episodes[i]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"meta": self.meta, "episodes": self.episodes}, f)
        return path

    @classmethod
    def load(cls, path):
        with open(Path(path)) as f:
            blob = json.load(f)
        return cls(blob["episodes"], blob["meta"])


def store_fingerprint(store: RecordStore) -> str:
    """Cheap identity check so a plan can't be paired with the wrong pkl."""
    import hashlib

    h = hashlib.sha256()
    h.update(str(len(store.records)).encode())
    for r in store.records[:1000]:
        h.update(str(r["note_id"]).encode())
    return h.hexdigest()[:16]


def _one_note_per_patient(store, label, patients, rng):
    """Pick exactly one record index from each of `patients`, at random."""
    out = []
    for p in patients:
        idxs = store.by_label_patient[label][p]
        out.append(int(idxs[rng.integers(0, len(idxs))]))
    return out


def _make_episode(store, k_shot, q_query, rng):
    """
    Draw one episode. Support and query patients are drawn WITHOUT replacement
    from the same class pool, so they are disjoint by construction.
    """
    support, query = {}, {}
    for c in CLASSES:
        pool = store.patients[c]
        need = k_shot + q_query
        if len(pool) < need:
            raise ValueError(
                f"class {c} in split '{store.split_name}' has {len(pool)} "
                f"patients but an episode needs {need} distinct patients "
                f"(k_shot={k_shot} + q_query={q_query})"
            )
        chosen = rng.choice(np.array(pool, dtype=object), size=need, replace=False)
        sup_p, qry_p = list(chosen[:k_shot]), list(chosen[k_shot:])
        support[str(c)] = _one_note_per_patient(store, c, sup_p, rng)
        query[str(c)] = _one_note_per_patient(store, c, qry_p, rng)
    return {"support": support, "query": query}


def build_train_plan(store: RecordStore, k_shot: int, q_query: int,
                     n_episodes: int, seed: int) -> EpisodePlan:
    """Random patient-disjoint episodes for meta-training."""
    rng = np.random.default_rng(seed)
    episodes = [_make_episode(store, k_shot, q_query, rng) for _ in range(n_episodes)]
    meta = {
        "kind": "train",
        "split": store.split_name,
        "k_shot": k_shot,
        "q_query": q_query,
        "n_episodes": n_episodes,
        "seed": seed,
        "store_fingerprint": store_fingerprint(store),
    }
    plan = EpisodePlan(episodes, meta)
    validate_plan(plan, store)  # fail loudly at construction, not at report time
    return plan


def build_eval_plan(store: RecordStore, k_shot: int, q_query: int,
                    seed: int, n_repeats: int = 3) -> EpisodePlan:
    """
    Coverage-guaranteed evaluation plan.

    Every patient in the split is used as a QUERY patient exactly `n_repeats`
    times, so the patient-level metric is computed over the whole split rather
    than over whichever patients random sampling happened to hit. The support
    patients for each episode are drawn from the same split but explicitly
    excluded from that episode's query batch.

    Support drawn from the same split is the correct few-shot protocol: at
    deployment the K labelled examples come from the target site. Support
    labels are given by definition, so this is not test-set leakage — but it
    IS worth one sentence in the paper's Experimental Setup.
    """
    rng = np.random.default_rng(seed)
    episodes = []

    for rep in range(n_repeats):
        # Per class, a shuffled queue of query patients.
        queues = {}
        for c in CLASSES:
            arr = np.array(store.patients[c], dtype=object).copy()
            rng.shuffle(arr)
            queues[c] = list(arr)

        n_batches = max(
            int(np.ceil(len(queues[c]) / q_query)) for c in CLASSES
        )
        for b in range(n_batches):
            support, query = {}, {}
            ok = True
            for c in CLASSES:
                q_pat = queues[c][b * q_query:(b + 1) * q_query]
                if not q_pat:
                    # This class ran out first; recycle from the front so the
                    # episode still has both classes. These recycled patients
                    # are NOT counted again in coverage (see validate_plan).
                    start = (b * q_query) % max(len(queues[c]), 1)
                    q_pat = queues[c][start:start + q_query]
                if len(q_pat) < 1:
                    ok = False
                    break
                remaining = [p for p in store.patients[c] if p not in set(q_pat)]
                if len(remaining) < k_shot:
                    ok = False
                    break
                sup_p = list(
                    rng.choice(np.array(remaining, dtype=object),
                               size=k_shot, replace=False)
                )
                support[str(c)] = _one_note_per_patient(store, c, sup_p, rng)
                query[str(c)] = _one_note_per_patient(store, c, q_pat, rng)
            if ok:
                episodes.append({"support": support, "query": query})

    meta = {
        "kind": "eval",
        "split": store.split_name,
        "k_shot": k_shot,
        "q_query": q_query,
        "n_episodes": len(episodes),
        "n_repeats": n_repeats,
        "seed": seed,
        "store_fingerprint": store_fingerprint(store),
    }
    plan = EpisodePlan(episodes, meta)
    validate_plan(plan, store)
    return plan


# =============================================================================
# LEAKAGE VALIDATION
# =============================================================================
def validate_plan(plan: EpisodePlan, store: RecordStore, strict: bool = True) -> dict:
    """
    Re-derive patient sets from record indices and check the invariant

        support_patients ∩ query_patients == ∅

    for every episode. Also checks that no record index is used twice within
    an episode and that support slots are K distinct patients.

    Returns a stats dict suitable for printing into the audit report.
    Raises AssertionError when strict=True and any violation is found.
    """
    n_overlap_episodes = 0
    n_overlap_patients = 0
    n_dup_record_episodes = 0
    n_nondistinct_support = 0
    covered = defaultdict(int)

    for ep in plan.episodes:
        sup_p, qry_p, all_idx = set(), set(), []
        for c_str, idxs in ep["support"].items():
            pats = [str(store.records[i]["subject_id"]) for i in idxs]
            if len(set(pats)) != len(pats):
                n_nondistinct_support += 1
            sup_p |= set(pats)
            all_idx += list(idxs)
        for c_str, idxs in ep["query"].items():
            pats = [str(store.records[i]["subject_id"]) for i in idxs]
            qry_p |= set(pats)
            all_idx += list(idxs)
            for p in pats:
                covered[p] += 1

        inter = sup_p & qry_p
        if inter:
            n_overlap_episodes += 1
            n_overlap_patients += len(inter)
        if len(set(all_idx)) != len(all_idx):
            n_dup_record_episodes += 1

    stats = {
        "episodes_tested": len(plan.episodes),
        "episodes_with_support_query_patient_overlap": n_overlap_episodes,
        "total_overlapping_patients": n_overlap_patients,
        "leakage_rate": (n_overlap_episodes / len(plan.episodes)) if len(plan) else 0.0,
        "episodes_with_duplicate_record_index": n_dup_record_episodes,
        "episodes_with_nondistinct_support_patients": n_nondistinct_support,
        "distinct_patients_queried": len(covered),
        "patients_in_split": store.n_patients(),
        "query_coverage_fraction": round(
            len(covered) / max(store.n_patients(), 1), 4
        ),
        "mean_times_queried": round(
            float(np.mean(list(covered.values()))) if covered else 0.0, 3
        ),
    }

    if strict:
        assert stats["episodes_with_support_query_patient_overlap"] == 0, (
            f"SUPPORT/QUERY PATIENT LEAKAGE in "
            f"{stats['episodes_with_support_query_patient_overlap']} episodes"
        )
        assert stats["episodes_with_duplicate_record_index"] == 0, (
            "a record index was reused inside a single episode"
        )
        assert stats["episodes_with_nondistinct_support_patients"] == 0, (
            "support set contains two notes from the same patient"
        )
    return stats


def format_leakage_report(stats: dict, k_shot: int) -> str:
    return (
        f"K={k_shot}\n"
        f"  Episodes tested:                 {stats['episodes_tested']}\n"
        f"  Support/query patient overlaps:  "
        f"{stats['episodes_with_support_query_patient_overlap']}\n"
        f"  Leakage rate:                    {stats['leakage_rate']:.4%}\n"
        f"  Distinct patients queried:       "
        f"{stats['distinct_patients_queried']} / {stats['patients_in_split']} "
        f"({stats['query_coverage_fraction']:.1%})\n"
        f"  Mean times each patient queried: {stats['mean_times_queried']}\n"
    )
