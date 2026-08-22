"""
collate.py — Turn a plan episode (record indices) into packed tensors.

Chunks from all notes in a side (support or query) are concatenated into a
single [n_chunks, L] tensor plus a `note_index` vector telling the embedder
which chunks belong to which note. One BERT call per side per episode.

Author: Dulhara Kaushalya (IT22130648)
"""

from __future__ import annotations

import torch


def _pack(store, indices, device):
    ids_rows, mask_rows, note_index = [], [], []
    labels, days, subject_ids, note_ids = [], [], [], []

    for slot, ridx in enumerate(indices):
        r = store.records[ridx]
        chunks_ids = r["input_ids"]
        chunks_mask = r["attention_mask"]
        for cid, cmask in zip(chunks_ids, chunks_mask):
            ids_rows.append(cid)
            mask_rows.append(cmask)
            note_index.append(slot)
        labels.append(int(r["label"]))
        days.append(float(r.get("days_before_patient_last_note", 0.0)))
        subject_ids.append(str(r["subject_id"]))
        note_ids.append(str(r["note_id"]))

    return {
        "input_ids": torch.tensor(ids_rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long, device=device),
        "note_index": torch.tensor(note_index, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "days": torch.tensor(days, dtype=torch.float32, device=device),
        "subject_ids": subject_ids,
        "note_ids": note_ids,
    }


def collate_episode(episode: dict, store, device) -> dict:
    """
    episode: {"support": {"0": [idx...], "1": [...]}, "query": {...}}
    Class keys are strings because the plan round-trips through JSON.
    """
    sup_idx, qry_idx = [], []
    for c in sorted(episode["support"].keys(), key=int):
        sup_idx += list(episode["support"][c])
    for c in sorted(episode["query"].keys(), key=int):
        qry_idx += list(episode["query"][c])

    batch = {
        "support": _pack(store, sup_idx, device),
        "query": _pack(store, qry_idx, device),
    }

    # Belt-and-braces: the invariant is enforced in sampler.validate_plan, but
    # re-checking it at the point of use costs nothing and means a hand-edited
    # or mismatched plan cannot silently produce leaked results.
    overlap = set(batch["support"]["subject_ids"]) & set(batch["query"]["subject_ids"])
    if overlap:
        raise RuntimeError(
            f"support/query patient overlap at collate time: {sorted(overlap)[:5]}"
        )
    return batch
