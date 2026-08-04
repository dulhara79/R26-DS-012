"""
delong_pipeline.py
Align two saved patient-level score sets and run the paired DeLong test.
Author: (for Dulhara Kaushalya's TC-WPN pipeline)

DeLong is a PAIRED test: it needs the SAME patients, in the SAME order, scored
by both models. evaluate_patient_level() samples episodes stochastically, so the
two models may cover slightly different patient sets. This helper intersects on
patient_id and aligns the vectors before testing — robust to coverage differences.

INPUT: two .npz files, each saved with keys: patient_ids, y_true, y_prob
       (see the 'how to save' snippet in the chat answer).

    python delong_pipeline.py tcwpn_rw.npz proto_rw.npz
"""
import sys
import numpy as np
from delong import delong_roc_test, delong_auc_ci


def align_and_delong(npz_model_a, npz_model_b, name_a="Model A", name_b="Model B"):
    A = np.load(npz_model_a, allow_pickle=True)
    B = np.load(npz_model_b, allow_pickle=True)

    ids_a = np.array([str(x) for x in A["patient_ids"]])
    ids_b = np.array([str(x) for x in B["patient_ids"]])

    # Map patient_id -> (y_true, prob) for each model
    a_map = {pid: (int(yt), float(yp))
             for pid, yt, yp in zip(ids_a, A["y_true"], A["y_prob"])}
    b_map = {pid: (int(yt), float(yp))
             for pid, yt, yp in zip(ids_b, B["y_true"], B["y_prob"])}

    common = sorted(set(a_map) & set(b_map))
    if len(common) < 20:
        raise ValueError(f"Only {len(common)} shared patients — too few. "
                         f"Increase n_episodes so coverage overlaps more.")

    y_true, p_a, p_b = [], [], []
    for pid in common:
        yt_a, pa = a_map[pid]
        yt_b, pb = b_map[pid]
        assert yt_a == yt_b, f"label mismatch for {pid}: {yt_a} vs {yt_b}"
        y_true.append(yt_a); p_a.append(pa); p_b.append(pb)

    y_true = np.array(y_true); p_a = np.array(p_a); p_b = np.array(p_b)

    auc_a, lo_a, hi_a = delong_auc_ci(y_true, p_a)
    auc_b, lo_b, hi_b = delong_auc_ci(y_true, p_b)
    a1, a2, z, p = delong_roc_test(y_true, p_a, p_b)

    print("=" * 64)
    print(f"PAIRED DeLong TEST  ({len(common)} shared patients, "
          f"{int(y_true.sum())} positive)")
    print("=" * 64)
    print(f"  {name_a:<22} AUROC={auc_a:.4f}  95% CI [{lo_a:.4f}, {hi_a:.4f}]")
    print(f"  {name_b:<22} AUROC={auc_b:.4f}  95% CI [{lo_b:.4f}, {hi_b:.4f}]")
    print(f"  Difference            {a1 - a2:+.4f}")
    print(f"  z = {z:.3f}   p = {p:.4f}")
    if p < 0.05:
        print(f"  -> SIGNIFICANT at alpha=0.05: {name_a} differs from {name_b}.")
    else:
        print(f"  -> NOT significant (p>=0.05): cannot claim {name_a} beats {name_b}.")
    print("=" * 64)
    return {"auc_a": auc_a, "auc_b": auc_b, "z": z, "p": p, "n": len(common)}


if __name__ == "__main__":
    if len(sys.argv) == 3:
        align_and_delong(sys.argv[1], sys.argv[2], "TC-WPN", "Standard ProtoNet")
    else:
        # ---- Self-test: build two saved files with partial overlap & verify ----
        rng = np.random.default_rng(1)
        n = 300
        ids = np.array([f"1_{i}" if i % 3 == 0 else f"0_{i}" for i in range(n)])
        y = np.array([1 if s.startswith("1_") else 0 for s in ids])
        pa = y * 0.7 + rng.normal(0, 1, n)
        pb = y * 0.68 + rng.normal(0, 1, n)
        # model B is missing 30 patients (coverage gap) and shuffled
        keepB = rng.choice(n, n - 30, replace=False)
        np.savez("a.npz", patient_ids=ids, y_true=y, y_prob=pa)
        np.savez("b.npz", patient_ids=ids[keepB], y_true=y[keepB], y_prob=pb[keepB])
        align_and_delong("a.npz", "b.npz", "TC-WPN", "Standard ProtoNet")
