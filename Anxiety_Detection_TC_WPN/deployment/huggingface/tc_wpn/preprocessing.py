"""
tc_wpn/preprocessing.py — VENDORED. Do not edit here.

§3: "TC-WPN source code + checkpoint + tokenizer + inference preprocessing"
must be treated as one versioned unit. Preprocessing at serving time must be
byte-for-byte the same function that produced the training corpus, or the
deployed model is not the model that was evaluated.

`chunk_tokenize` below is copied verbatim from
    tcwpn_test/scripts/tokenize_cohort.py
at the commit recorded in deployment_config.json -> provenance.tcwpn_git_commit.

vendor.sh re-copies it. If you change the training tokeniser, re-run vendor.sh
and bump preprocessing_version in deployment_config.json.
"""

from __future__ import annotations


def chunk_tokenize(text, tokenizer, max_len, max_chunks, stride):
    """
    Sliding-window tokenisation. Returns (input_ids, attention_mask) as lists
    of lists, at most `max_chunks` windows. max_chunks=1 keeps only the first
    window — cheap, and on discharge summaries the opening sections carry the
    chief complaint and HPI.

    VERBATIM from scripts/tokenize_cohort.py.
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_overflowing_tokens=max_chunks > 1,
        stride=stride if max_chunks > 1 else 0,
    )
    ids, mask = enc["input_ids"], enc["attention_mask"]
    if isinstance(ids[0], int):  # single window -> wrap
        ids, mask = [ids], [mask]
    return ids[:max_chunks], mask[:max_chunks]


def pack_notes(texts, tokenizer, max_len, max_chunks, stride, device):
    """
    Serving-side equivalent of collate._pack, minus the record store.

    Returns (input_ids [n_chunks_total, L], attention_mask [n_chunks_total, L],
    note_index [n_chunks_total]) — the exact three tensors
    ClinicalEmbedder.forward expects.

    The note_index vector is what lets one BERT call cover every note in a
    side while the embedder still mean-pools chunks back to per-note vectors.
    Building it by hand here (rather than assuming one chunk per note) is what
    keeps max_chunks > 1 correct if the training config ever changes.
    """
    import torch

    ids_rows, mask_rows, note_index = [], [], []
    for slot, text in enumerate(texts):
        c_ids, c_mask = chunk_tokenize(text, tokenizer, max_len, max_chunks, stride)
        for cid, cmask in zip(c_ids, c_mask):
            ids_rows.append(cid)
            mask_rows.append(cmask)
            note_index.append(slot)

    return (
        torch.tensor(ids_rows, dtype=torch.long, device=device),
        torch.tensor(mask_rows, dtype=torch.long, device=device),
        torch.tensor(note_index, dtype=torch.long, device=device),
    )
