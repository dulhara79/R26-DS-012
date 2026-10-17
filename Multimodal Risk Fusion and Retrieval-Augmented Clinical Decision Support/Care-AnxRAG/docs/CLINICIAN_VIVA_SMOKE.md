# Psychiatrist-facing viva smoke test

Following the 2026-10-16 corpus backfill, the workstation reported:
- 157 active documents: 156 PubMed, 1 WHO
- 566 active chunks: 555 Research Frontier, 11 Clinical Core
- 47 unresolved staged versions, 485 rejected versions, 12 superseded
- Self-check healthy (SQLite, Chroma, Ollama, matching embedding identities)

These figures are *user-provided workstation measurements*, not independently
replicated by GitHub CI. The corpus is **machine-screened, not clinician-approved**.

## Run against the real local corpus

With Ollama available and your usual virtual environment activated:

    python scripts/viva_clinician_smoke.py --project-root .

For an initial smaller run:

    python scripts/viva_clinician_smoke.py --project-root . --limit 3

Output is written to a timestamped directory under:
artifacts/final-viva/clinician-smoke/

- manifest.json: runtime configuration identifiers (no secrets), database stats,
  per-case status, abstentions, citation counts, timings, guard failures
- responses.json: verbatim responses and citation metadata for each case
- clinician-review-blank.csv: **blank** columns for genuine human ratings of
  relevance, source support, population applicability and notes

## Interpreting the output

Seven professional evidence questions are automatically marked
requires_manual_review, whether answered or abstained. In particular,
a well-formed citation is not proof that the text answers the psychiatrist's
question, that a comparison is supported, or that the clinical recommendation
is correct. Do not treat these as benchmark gold labels or reported research
metrics.

Three guard cases (personal diagnosis, out-of-domain fracture advice and
a synthetic crisis phrase) have deterministic expected routes and can pass
or fail mechanically. The process exits nonzero if a guard failed or an exception
occurred. Real crisis response performance and clinical suitability still need
independent expert review.

The current research freeze is blocked by unresolved staging review. Do not
claim final B0–B5 CARE_full benchmark metrics without an adjudicated test set.
