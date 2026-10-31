# Controlled PubMed backfill for psychiatrist research demonstration

**Goal:** measure 100 distinct active evidence documents (not 100 chunks, sources or staged versions). The baseline of 15 documents is historic; this tool reads the current local count.

**Clinical status:** this is an evidence-retrieval *research prototype*, not a clinically approved evidence database. Ingestion validation and machine screening are not qualified psychiatrist review or independent benchmark adjudication.

## Before you run

From your existing CARE-AnxRAG checkout and activated environment:

    git pull --ff-only
    export CARE_SOURCE_REGISTRY="$PWD/config/sources.yaml"
    care-anxrag selfcheck --project-root .

Set a real NCBI_EMAIL privately in your existing .env or shell. Do not commit credentials. Keep Ollama and the existing 256-dimensional Chroma index available. Do not reset embeddings.

## 1. Read-only PubMed counts

    python scripts/pubmed_backfill.py preview

Counts reflect the modified-date windows used by the existing NCBI connector. Choose a window in which candidate count is positive and no greater than the cap. Candidate count is not accepted-document count.

## 2. Stage research candidates with manual review default

    python scripts/pubmed_backfill.py sync --days 30 --cap 500

The canonical config/sources.yaml and .env are not changed. Candidates pass through the existing ingestion validator, retraction and duplicate checks, staging, and vector indexing. The review workflow remains docs/STAGING_REVIEW_WORKFLOW.md.

## 3. Explicit machine-screened research-demo activation only

If you specifically need to evaluate retrieval against a broader NONCLINICAL and NON-ADJUDICATED research-prototype corpus:

    python scripts/pubmed_backfill.py sync --days 30 --cap 500 --machine-screened-demo --confirm-research-only

This deliberately activates candidates passing the automated validator. **It does not establish clinician review**, fitness for clinical decisions, or any safety/accuracy claim. Include the label "machine-screened; not clinically adjudicated" in every demonstration/report describing this corpus.

## 4. Inspect the actual active counts and records

    care-anxrag stats --project-root . > artifacts/final-viva/corpus-after.json
    care-anxrag selfcheck --project-root .

The operator writes a SQLite online backup, effective temporary registry, and report.json to artifacts/final-viva/pubmed-backfill/<timestamp>/. **Chroma is not included in that SQLite backup**; do not treat it as a complete index backup. Review report.json, especially active_before, active_after, errors, and failed records. A window may produce fewer than 100 after validation; do not invent the missing documents.

If you still have 47 unresolved staging versions, the formal corpus freeze remains blocked. Machine-screened activation cannot substitute for that human review gate. Final B0-B5 and CARE_full research results still require a genuinely annotated independent benchmark and a valid formal freeze.

## 5. Clinician-question quality checks

Run the prompts described in docs/PSYCHIATRIST_VIVA_PROTOCOL.md and capture real answers, timestamps, citations and abstentions. Distinguish the engineering regression suite from clinical effectiveness measurements.
