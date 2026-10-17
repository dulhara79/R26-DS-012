# CARE-AnxRAG: recall investigation after 157-document backfill

The Oct 16 PR #38 clinician smoke run answered P05 from WHO and abstained
on P01, P02, P03, P04, P06, and P07 with insufficient_answer_evidence.

## What changed

In full retrieval mode, the CARE ranker previously chose six citation
context chunks before checking if their sentences answered the requested
clinical question. Relevant-but-background passages sometimes filled all
six slots and hid a treatment-result passage ranked seventh or later.

This PR prioritizes answer-bearing evidence *within the existing relevance
threshold* and excluding conflicted chunks. Both groups preserve their
original CARE order. Source sentences, citation validation, and abstention
rules are unchanged; no gold-standard review labels are manufactured.

## Read-only real-corpus diagnosis

Using this branch's diagnostic script in your existing virtualenv:

    python scripts/diagnose_clinician_recall.py --project-root .

The audit under artifacts/final-viva/clinician-recall contains title, URL,
source excerpt, CARE score, relevance, and whether an exact source sentence
passes the current answer selector. Absence of a finding cannot be fixed
by making up an answer. The local 157-document corpus is machine-screened
and is not a clinically adjudicated reference set.

For before-and-after comparisons, keep the active corpus and model
settings unchanged, run the 7-case smoke script, and compare the answers
with the earlier Oct 16 outputs. Do not equate a citation or high retrieval
confidence with clinical answer quality.

For the safe Mac workflow, run from an isolated Git worktree and export
PYTHONPATH to this branch's src directory while preserving project-root
as the original checkout containing the local SQLite and Chroma files.
