"""
run_shallow_baselines.py — Stage 5b. The baselines that decide whether the
paper has a finding at all.

    python -m scripts.run_shallow_baselines --stem psych_mimic4 --k 5 \
        --baseline tfidf_lr --split test
    python -m scripts.run_shallow_baselines --stem psych_mimic4 --k 5 \
        --baseline bert_probe --split test

Both baselines are FEW-SHOT in exactly the same sense as TC-WPN: for each
episode they see only the 2K support notes, fit a classifier on them, and score
that episode's queries. They consume the same frozen episode plan and emit the
same predictions CSV format, so the comparison is paired and the DeLong test is
valid.

This matters more than the model work. If TF-IDF on 2x5 support notes reaches
AUROC 0.9 on this cohort, the task is lexical and no architecture claim is
supportable. Run these BEFORE spending GPU time on TC-WPN.

  tfidf_lr    : TF-IDF (word 1-2 grams, fit on the support notes only) +
                logistic regression. Fitting the vectoriser per episode is the
                strict choice — fitting it on the training corpus would leak
                corpus-level vocabulary statistics into a "few-shot" baseline.
  bert_probe  : frozen Bio_ClinicalBERT [CLS] + logistic regression on the
                support notes. Isolates "does episodic meta-training buy
                anything over a frozen encoder".

Author: Dulhara Kaushalya (IT22130648)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tcwpn.metrics import compute_metrics, select_threshold   # noqa: E402
from tcwpn.sampler import EpisodePlan, RecordStore            # noqa: E402


# =============================================================================
# TEXT RECOVERY
# The pkl stores token ids, not text. For TF-IDF we decode back with the same
# tokenizer, which guarantees the baseline sees exactly the tokens the neural
# models see — including any blinding that was applied at tokenisation time.
# =============================================================================
def decode_records(store, tokenizer):
    texts = []
    for r in store.records:
        ids = [t for chunk in r["input_ids"] for t in chunk]
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return texts


def bert_cls_embeddings(store, encoder_name, device, batch_size=32):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(encoder_name).to(device).eval()
    rows_ids, rows_mask, note_idx = [], [], []
    for i, r in enumerate(store.records):
        for cid, cmask in zip(r["input_ids"], r["attention_mask"]):
            rows_ids.append(cid)
            rows_mask.append(cmask)
            note_idx.append(i)

    out = np.zeros((len(store.records), model.config.hidden_size), dtype=np.float32)
    counts = np.zeros(len(store.records), dtype=np.float32)
    with torch.no_grad():
        for s in tqdm(range(0, len(rows_ids), batch_size), desc="encoding"):
            ids = torch.tensor(rows_ids[s:s + batch_size], device=device)
            mask = torch.tensor(rows_mask[s:s + batch_size], device=device)
            cls = model(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
            cls = cls.float().cpu().numpy()
            for j, vec in enumerate(cls):
                n = note_idx[s + j]
                out[n] += vec
                counts[n] += 1
    return out / np.maximum(counts, 1)[:, None]


# =============================================================================
# EPISODIC FIT/PREDICT
# =============================================================================
def episode_indices(ep):
    sup, qry = [], []
    for c in sorted(ep["support"], key=int):
        sup += [(i, int(c)) for i in ep["support"][c]]
    for c in sorted(ep["query"], key=int):
        qry += [(i, int(c)) for i in ep["query"][c]]
    return sup, qry


def run_tfidf(store, plan, texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    probs = defaultdict(list)
    labels = {}
    for ep in tqdm(plan, desc="tfidf episodes"):
        sup, qry = episode_indices(ep)
        Xs = [texts[i] for i, _ in sup]
        ys = [c for _, c in sup]
        if len(set(ys)) < 2:
            continue
        clf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                            max_features=50_000),
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        )
        clf.fit(Xs, ys)
        Xq = [texts[i] for i, _ in qry]
        pq = clf.predict_proba(Xq)[:, list(clf.classes_).index(1)]
        for (i, c), p in zip(qry, pq):
            sid = str(store.records[i]["subject_id"])
            probs[sid].append(float(p))
            labels[sid] = c
    return probs, labels


def run_probe(store, plan, emb):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    probs = defaultdict(list)
    labels = {}
    for ep in tqdm(plan, desc="probe episodes"):
        sup, qry = episode_indices(ep)
        ys = [c for _, c in sup]
        if len(set(ys)) < 2:
            continue
        Xs = emb[[i for i, _ in sup]]
        sc = StandardScaler().fit(Xs)
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(Xs), ys)
        Xq = sc.transform(emb[[i for i, _ in qry]])
        pq = clf.predict_proba(Xq)[:, list(clf.classes_).index(1)]
        for (i, c), p in zip(qry, pq):
            sid = str(store.records[i]["subject_id"])
            probs[sid].append(float(p))
            labels[sid] = c
    return probs, labels


def to_vectors(probs, labels):
    ids = sorted(probs)
    y = np.array([labels[i] for i in ids], dtype=int)
    p = np.array([float(np.mean(probs[i])) for i in ids], dtype=float)
    cov = np.array([len(probs[i]) for i in ids], dtype=int)
    return ids, y, p, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="psych_mimic4")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--baseline", choices=["tfidf_lr", "bert_probe"], required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--blind", default=None)
    ap.add_argument("--pkl-dir", default="data/clean/pkl")
    ap.add_argument("--plan-dir", default="data/clean/plans")
    ap.add_argument("--results", default="results")
    ap.add_argument("--encoder", default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    suffix = f"_blind-{args.blind}" if args.blind else ""
    pkl_dir, plan_dir = Path(args.pkl_dir), Path(args.plan_dir)

    run_dir = (Path(args.results) / args.stem /
               f"{args.baseline}_k{args.k}_seed{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- threshold is locked on validation, exactly as for the neural runs ---
    def score(split):
        store = RecordStore.from_pkl(
            pkl_dir / f"{args.stem}_{split}{suffix}.pkl", split_name=split)
        plan = EpisodePlan.load(plan_dir / f"{args.stem}_{split}_k{args.k}.json")
        if args.baseline == "tfidf_lr":
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.encoder)
            probs, labels = run_tfidf(store, plan, decode_records(store, tok))
        else:
            import torch

            device = torch.device(
                args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            probs, labels = run_probe(
                store, plan, bert_cls_embeddings(store, args.encoder, device))
        return store, plan, to_vectors(probs, labels)

    _, _, (_, y_val, p_val, _) = score("val")
    threshold = select_threshold(y_val, p_val)
    print(f"locked threshold from validation: {threshold:.4f}")

    store, plan, (ids, y, p, cov) = score(args.split)
    metrics = compute_metrics(y, p, threshold=threshold)
    metrics.pop("_ece_table")
    metrics.update({
        "run": run_dir.name, "split": args.split, "k_shot": args.k,
        "seed": args.seed, "blind": args.blind or "none",
        "baseline": args.baseline, "n_episodes": len(plan),
        "query_coverage_fraction": round(len(ids) / max(store.n_patients(), 1), 4),
    })

    tag = f"{args.split}{suffix}"
    with open(run_dir / f"eval_{tag}.json", "w") as f:
        json.dump({"metrics": metrics}, f, indent=2)
    pd.DataFrame({"patient_id": ids, "label": y, "p_anxiety": p,
                  "n_episodes": cov}).to_csv(
        run_dir / f"predictions_{tag}.csv", index=False)

    with open(run_dir / "manifest.json", "w") as f:
        json.dump({"run_name": run_dir.name, "baseline": args.baseline,
                   "k_shot": args.k, "seed": args.seed, "stem": args.stem,
                   "locked_threshold": float(threshold)}, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nwrote {run_dir/f'predictions_{tag}.csv'}")


if __name__ == "__main__":
    main()
