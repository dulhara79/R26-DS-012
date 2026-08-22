"""
test_episode_leakage.py — the gate the supervisor set: 5,000 episodes at
K = 1, 3, 5, 10 with zero support/query patient overlap.

    pytest -q tests/test_episode_leakage.py -s

`test_old_flat_split_sampler_leaks` deliberately reimplements the ARCHIVED
sampling logic and asserts that it DOES leak. That test is the evidence that
the bug was real and that the new sampler fixes it — worth one sentence in the
paper's methods and a very strong answer if a reviewer asks why the numbers
changed.
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tcwpn.sampler import (
    RecordStore, build_eval_plan, build_train_plan,
    format_leakage_report, validate_plan,
)

K_VALUES = [1, 3, 5, 10]
N_EPISODES = 5000
Q_QUERY = 5


def make_store(n_patients_per_class=60, notes_per_patient=4, seed=0):
    """Synthetic store: token content is irrelevant to the leakage invariant."""
    rng = random.Random(seed)
    records = []
    for label in (0, 1):
        for p in range(n_patients_per_class):
            sid = f"L{label}P{p:04d}"
            for n in range(notes_per_patient):
                records.append({
                    "note_id": f"{sid}-N{n}",
                    "subject_id": sid,
                    "label": label,
                    "split": "train",
                    "input_ids": [[rng.randint(0, 100) for _ in range(8)]],
                    "attention_mask": [[1] * 8],
                    "days_before_patient_last_note": float(30 * (notes_per_patient - n)),
                    "n_notes_patient": notes_per_patient,
                })
    return RecordStore(records, split_name="synthetic")


# =============================================================================
# THE CERTIFICATE
# =============================================================================
@pytest.mark.parametrize("k", K_VALUES)
def test_zero_leakage_at_k(k):
    store = make_store()
    plan = build_train_plan(store, k_shot=k, q_query=Q_QUERY,
                            n_episodes=N_EPISODES, seed=42 + k)
    stats = validate_plan(plan, store, strict=False)
    print("\n" + format_leakage_report(stats, k))

    assert stats["episodes_tested"] == N_EPISODES
    assert stats["episodes_with_support_query_patient_overlap"] == 0
    assert stats["leakage_rate"] == 0.0
    assert stats["episodes_with_nondistinct_support_patients"] == 0
    assert stats["episodes_with_duplicate_record_index"] == 0


def test_support_is_k_distinct_patients():
    store = make_store()
    k = 5
    plan = build_train_plan(store, k, Q_QUERY, n_episodes=200, seed=1)
    for ep in plan:
        for c, idxs in ep["support"].items():
            pats = {store.records[i]["subject_id"] for i in idxs}
            assert len(pats) == k, "K-shot must mean K distinct patients"


def test_eval_plan_covers_every_patient_exactly_n_repeats():
    store = make_store(n_patients_per_class=40, notes_per_patient=2)
    plan = build_eval_plan(store, k_shot=3, q_query=5, seed=7, n_repeats=2)
    stats = validate_plan(plan, store, strict=False)
    assert stats["episodes_with_support_query_patient_overlap"] == 0
    assert stats["query_coverage_fraction"] == 1.0, (
        "every patient in the split must be queried; partial coverage means "
        "the reported metric describes a random subset, not the split"
    )


def test_plan_roundtrips_through_json(tmp_path):
    store = make_store()
    plan = build_train_plan(store, 3, Q_QUERY, n_episodes=50, seed=3)
    p = plan.save(tmp_path / "plan.json")

    from tcwpn.sampler import EpisodePlan

    loaded = EpisodePlan.load(p)
    assert len(loaded) == len(plan)
    assert loaded.episodes == plan.episodes, (
        "plans must round-trip byte-identically or two models cannot be "
        "compared on the same episodes"
    )
    validate_plan(loaded, store)


def test_episode_raises_when_pool_too_small():
    store = make_store(n_patients_per_class=6, notes_per_patient=2)
    with pytest.raises(ValueError, match="distinct patients"):
        build_train_plan(store, k_shot=5, q_query=5, n_episodes=10, seed=0)


def test_collate_refuses_a_leaked_episode():
    """Defence in depth: even a hand-edited plan must not reach the model."""
    torch = pytest.importorskip(
        "torch", reason="collate is torch-backed; run this test in the training env"
    )

    from tcwpn.collate import collate_episode

    store = make_store()
    sid = store.patients[1][0]
    idxs = store.by_label_patient[1][sid]
    bad = {
        "support": {"0": store.by_label_patient[0][store.patients[0][0]][:1],
                    "1": idxs[:1]},
        "query": {"0": store.by_label_patient[0][store.patients[0][1]][:1],
                  "1": idxs[1:2]},          # SAME patient as support
    }
    with pytest.raises(RuntimeError, match="support/query patient overlap"):
        collate_episode(bad, store, torch.device("cpu"))


# =============================================================================
# REGRESSION TEST FOR THE ARCHIVED BUG
# =============================================================================
def _archived_build_class_examples(store, label, k_shot, q_query,
                                   max_notes_per_patient=3, rng=None):
    """
    Faithful reimplementation of episode_dataset.py::_build_class_examples:
    walk patients, take up to `max_notes_per_patient` notes each into a FLAT
    list, then cut the list at k_shot. A patient's notes straddle the cut.
    """
    rng = rng or random.Random(0)
    total = k_shot + q_query
    candidates = list(store.patients[label])
    rng.shuffle(candidates)
    selected = []
    for sid in candidates:
        if len(selected) >= total:
            break
        notes = store.by_label_patient[label][sid]
        take = min(len(notes), max_notes_per_patient, total - len(selected))
        selected.extend(notes[:take])
    selected = selected[:total]
    return selected[:k_shot], selected[k_shot:]


def test_old_flat_split_sampler_leaks():
    """
    Demonstrates the defect in the archived sampler. If this test ever starts
    failing it means the synthetic fixture changed, not that the old code was
    fine.
    """
    store = make_store(notes_per_patient=3)
    rng = random.Random(123)
    leaked = 0
    trials = 500
    for _ in range(trials):
        sup, qry = _archived_build_class_examples(store, 1, 5, 5, 3, rng)
        sp = {store.records[i]["subject_id"] for i in sup}
        qp = {store.records[i]["subject_id"] for i in qry}
        if sp & qp:
            leaked += 1
    rate = leaked / trials
    print(f"\narchived sampler support/query patient overlap rate: {rate:.1%}")
    assert rate > 0.5, (
        "expected the archived flat-split sampler to leak in most episodes"
    )
