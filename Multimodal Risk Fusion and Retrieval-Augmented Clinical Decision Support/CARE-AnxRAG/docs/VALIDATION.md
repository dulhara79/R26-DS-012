# Validation

This report is regenerated/confirmed before release packaging.

## Deterministic validation strategy

The offline test stack uses:

- SQLite provenance ledger;
- SQLite FTS5 lexical index;
- exact SQLite cosine vector store;
- deterministic feature-hash embeddings;
- heuristic reranker;
- sentence-level heuristic NLI;
- rule-based generator.

This validates orchestration and safety properties without network/model nondeterminism.

## Tested behaviors

- configuration, supported embedding dimensions, model/index identity, and weight validation;
- section chunking and safe FTS query construction;
- idempotent ingestion, immutable versioning, manual withdrawal, and safe rollback;
- active/superseded transitions;
- physical vector cleanup and orphan reconciliation;
- vector outbox completion;
- failed-record cursor preservation, dry-run timestamp integrity, disabled-source rejection, and overlapping-sync rejection;
- hybrid retrieval ranking;
- independent relevance abstention;
- contradiction detection and abstention;
- crisis, urgent, negation, and academic-context routing;
- citation generation/validation and citation-repair context retention;
- FastAPI behavior and admin protection;
- Chroma adapter configuration and pagination contracts;
- Ollama embed/chat request contracts;
- PubMed, PMC, HTTP-page, local-file, and NICE parser fixtures;
- NCBI anti-truncation/configuration guard;
- conservative PMC licence handling, including restricted-marker precedence;
- recursive secret redaction and rejection of plaintext registry credentials;
- packaged project scaffolding resources;
- evaluation accounting for abstention-only items through `retrieval_evaluable_count`;
- wheel/source-distribution contents, install/import, CLI, and packaged self-check.

## Environment limitations

The build environment does not have network access and does not contain the optional `chromadb` or `sentence-transformers` packages, an Ollama service, NICE credentials, or a clinical corpus. Therefore:

- no live NIMH/WHO/PubMed/NICE fetch was performed;
- no real Chroma persistence test was performed;
- no real embedding/reranker/NLI model inference was performed;
- no live Ollama generation was performed;
- no clinical correctness claim is made.

These boundaries are covered by mocked contract tests and must be followed by deployment acceptance testing.

## Commands

```bash
PYTHONPATH=src python -m compileall -q src
PYTHONPATH=src pytest -q
CARE_HOME=/tmp/care-selfcheck/var PYTHONPATH=src \
  python -m care_anxrag.cli selfcheck --offline --project-root .
```

The release package also includes `scripts/validate.sh`, which executes the complete local validation workflow and writes `validation-report.json`. Its stages include standard-library syntax/configuration audits, tests with a coverage floor, deterministic self-check, end-to-end acceptance, source/wheel builds, package-content inspection, isolated wheel installation, and packaged CLI/self-check execution. The generated report is the source of truth for exact test and coverage counts.
