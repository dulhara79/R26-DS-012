# CARE-AnxRAG

**Contradiction-, Authority-, Reliability-, and Evidence-aware Retrieval-Augmented Generation for anxiety information research.**

CARE-AnxRAG is a complete reference implementation for a continuously updated, versioned, evidence-gated anxiety RAG system. It separates semantic relevance from evidence quality, retains source provenance, detects conflicting evidence, abstains when evidence is weak or inconsistent, validates citations, and routes urgent/crisis messages away from ordinary RAG.

> **Research and engineering system, not a clinical device.** It does not diagnose, replace a clinician, prescribe treatment, or provide emergency care. Clinical deployment requires formal governance, localized safety resources, security review, expert evaluation, and applicable regulatory/legal review.

## What is implemented

- Incremental source synchronization with per-source state, conditional HTTP requests, retries, rate limits, and update watermarks.
- Public-page monitoring for NIMH, WHO, and NICE update notices.
- PubMed modified-date ingestion through NCBI E-utilities.
- PMC full-text ingestion only when the configured licence allowlist confirms reuse.
- NICE syndication adapter, disabled until the operator supplies a licensed endpoint and key.
- Local authorized-document ingestion for Markdown, text, HTML, and JSON.
- SQLite provenance/version ledger with active, staging, superseded, rejected, and withdrawn states.
- Section hashes, stable document identity, cached embeddings, and a durable vector outbox.
- Persistent Chroma vector search in production and an exact SQLite vector backend for deterministic offline testing.
- Persistent SQLite FTS5 BM25 lexical retrieval.
- Dense + lexical retrieval, Reciprocal Rank Fusion, CrossEncoder reranking, and CARE evidence scoring.
- Independent relevance gating so authoritative but unrelated text cannot answer a query.
- Evidence-strength-aware contradiction handling using NLI.
- Calibrated abstention for irrelevance, weak evidence, insufficient diversity, and unresolved conflict.
- Clinical Core and Research Frontier knowledge layers.
- Ollama structured generation with strict source-ID validation and one repair attempt.
- Prompt-injection boundaries around retrieved evidence.
- Crisis/urgent-message routing before retrieval.
- FastAPI service, minimal browser UI, CLI, review workflow, scheduler, health checks, reconciliation, and evaluation harness.
- Offline unit, integration-contract, API, lifecycle, safety, conflict, and retrieval tests.

## Architecture

```text
Official/public/licensed/local sources
                |
                v
     Incremental source connectors
   (ETag / Last-Modified / mdat)
                |
                v
       Validation and staging
  relevance | licence | retraction
                |
                v
      Versioned SQLite ledger
    documents / versions / sections
                |
        +-------+-------+
        |               |
        v               v
 Chroma dense index   SQLite FTS5
        |               |
        +-------+-------+
                v
     Reciprocal Rank Fusion
                v
       CrossEncoder reranker
                v
       Independent relevance gate
                v
       CARE evidence re-ranking
  authority | evidence | freshness
       applicability | relevance
                v
      NLI contradiction analysis
                v
       conflict suppression or
       calibrated abstention
                v
  Ollama grounded JSON generation
                v
       citation validation
```

The SQLite ledger is the source of truth. Vector results are always joined back to active SQLite chunks, so a stale vector cannot make superseded evidence answerable. The vector outbox and `reconcile` command provide eventual consistency and physical cleanup.

See [Architecture](docs/ARCHITECTURE.md) for the detailed component and transaction design.

## Repository layout

```text
CARE-AnxRAG/
├── config/sources.yaml             # Source registry
├── data/local/                     # Authorized local evidence only
├── data/benchmark/                 # Benchmark JSONL
├── docs/                           # Architecture, governance, deployment, evaluation
├── scripts/                        # Bootstrap and validation scripts
├── src/care_anxrag/
│   ├── sources/                    # NCBI, NICE, HTTP, and local connectors
│   ├── ingestion.py                # Versioning, staging, promotion, vector lifecycle
│   ├── retrieval.py                # Hybrid retrieval + CARE scoring + conflict logic
│   ├── generation.py               # Ollama generation + citation validation
│   ├── safety.py                   # Pre-retrieval safety router
│   ├── api.py                      # FastAPI and browser UI
│   └── cli.py                      # Operations CLI
└── tests/                          # Deterministic validation suite
```

# Step-by-step setup

## 1. Use a supported Python

Python 3.11 or 3.12 is recommended for the production stack. The deterministic core also passes under Python 3.13.

```bash
python --version
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows PowerShell
python -m pip install --upgrade pip
```

## 2. Install CARE-AnxRAG

For the fully local deterministic development stack:

```bash
pip install -e ".[dev]"
```

For Chroma and model-based reranking/NLI:

```bash
pip install -e ".[production,dev]"
```

## 3. Install and prepare Ollama

Install Ollama using its official instructions, start it, then pull the configured models:

```bash
ollama pull embeddinggemma
ollama pull gemma3:4b
ollama list
```

The defaults use:

- `embeddinggemma` for embeddings through `/api/embed` at 256 dimensions; supported configured dimensions are 128, 256, 512, 768, or `0` for the model's native output
- `gemma3:4b` for grounded structured generation through `/api/chat`
- `cross-encoder/ms-marco-MiniLM-L6-v2` for passage reranking
- `cross-encoder/nli-deberta-v3-base` for contradiction classification

Sentence Transformers downloads CrossEncoder models on first use. Pin and pre-cache exact model revisions before a controlled experiment.

## 4. Configure environment variables

```bash
cp .env.example .env
```

At minimum, set a responsible NCBI contact and a strong admin key:

```dotenv
NCBI_EMAIL=your-research-email@example.org
CARE_SOURCE_USER_AGENT=CARE-AnxRAG/0.1 your-research-email@example.org
CARE_ADMIN_KEY=replace-with-a-long-random-secret
```

Local/offline validation settings:

```dotenv
CARE_VECTOR_BACKEND=sqlite
CARE_EMBEDDING_PROVIDER=hash
CARE_GENERATOR_PROVIDER=rule
CARE_RERANKER_PROVIDER=heuristic
CARE_NLI_PROVIDER=heuristic
CARE_ALLOW_NETWORK_SYNC=false
```

Production-oriented settings:

```dotenv
CARE_VECTOR_BACKEND=chroma
CARE_EMBEDDING_PROVIDER=ollama
CARE_GENERATOR_PROVIDER=ollama
CARE_RERANKER_PROVIDER=cross_encoder
CARE_NLI_PROVIDER=cross_encoder
CARE_ALLOW_NETWORK_SYNC=true
```

Do not switch embedding models or dimensions on an existing index without an explicit reset. CARE-AnxRAG records the embedding model identity in SQLite and refuses retrieval/indexing on a mismatch. After a deliberate model or dimension change, run `care-anxrag reconcile --reset-embedding-index --project-root .`.

## 5. Review the source registry

Edit `config/sources.yaml` before the first network synchronization.

Default registry behavior:

| Source | Default | Layer | Promotion policy |
|---|---|---|---|
| Local curated files | Enabled | Clinical Core | Manual |
| NIMH anxiety page | Enabled | Clinical Core | Manual review by default |
| WHO anxiety page | Enabled | Clinical Core | Manual review by default |
| NICE update monitor | Enabled, monitor only | Clinical Core | Never published to RAG |
| NICE syndication | Disabled | Clinical Core | Licensed endpoint/key + manual review |
| PubMed anxiety stream | Enabled | Research Frontier | Manual |
| PMC open-access stream | Disabled | Research Frontier | Manual and licence-gated |

Important controls:

- `publish_to_rag: false` monitors change without indexing content.
- `auto_promote: false` leaves accepted versions in staging.
- Research Frontier versions cannot auto-promote unless both the source and `CARE_AUTO_PROMOTE_RESEARCH=true` allow it.
- The NCBI connector refuses to silently truncate a result window. Increase `max_records_per_sync` or reduce `initial_lookback_days` when a window exceeds the cap.
- Do not enable NICE syndication until the licence and AI use conditions for your project are confirmed.

## 6. Run the deterministic self-check

This verifies configuration, SQLite, FTS5, the exact local vector backend, and weight normalization without requiring Chroma, Ollama, network access, or model downloads.

```bash
care-anxrag selfcheck --offline --project-root .
```

Expected top-level status:

```json
{
  "health": {"status": "ok"},
  "database_integrity": "ok",
  "weights_sum": 1.0
}
```

## 7. Run the full validation suite

```bash
./scripts/validate.sh
```

This performs compilation, tests, an offline self-check, package installation, and a synthetic end-to-end ingestion/query smoke test. See [Validation](docs/VALIDATION.md).

## 8. Initialize production services

`init` first scaffolds missing operator-owned files (`config/sources.yaml`, `.env.example`, and data/state directories) from packaged safe defaults without overwriting existing files. Ensure Ollama is running, then initialize the ledger and Chroma collections:

```bash
care-anxrag init --project-root .
```

The JSON response contains both `scaffold` actions (`written` or `preserved`) and `health`. A degraded health result usually means Ollama, Chroma dependencies, or a configured model is unavailable.

## 9. Synchronize evidence

Start with a dry run:

```bash
care-anxrag sync --dry-run --source nimh_anxiety --source who_anxiety --project-root .
care-anxrag sync --dry-run --source pubmed_anxiety --project-root .
```

Dry-run mode performs fetching, parsing, and validation but does not insert versions,
write vectors, or advance source cursors/watermarks. It does retain a synchronization-run
audit record in SQLite.

Then perform the real synchronization:

```bash
care-anxrag sync --source nimh_anxiety --source who_anxiety --project-root .
care-anxrag sync --source pubmed_anxiety --project-root .
```

`--force` ignores the last-success watermark. Use it deliberately; content hashes still keep ingestion idempotent.

## 10. Review Research Frontier evidence

```bash
care-anxrag staging --project-root .
care-anxrag approve VERSION_ID --project-root .
# or
care-anxrag reject VERSION_ID --reason "Not applicable to target population" --project-root .
# Withdraw the currently active version for that document and remove its vectors:
care-anxrag withdraw VERSION_ID --reason "Retracted or no longer approved" --project-root .
```

Approving a superseded version is a rollback. CARE-AnxRAG reindexes it before activation and removes the previously active vectors.

## 11. Ask and inspect

```bash
care-anxrag ask "What evidence-based interventions are discussed for panic disorder?" --project-root .
care-anxrag retrieve "What recent evidence exists for social anxiety?" --project-root .
care-anxrag stats --project-root .
```

`retrieve` exposes raw evidence and is an operator/debug command. The HTTP endpoint fails closed with `503` until `CARE_ADMIN_KEY` is configured, then requires a matching `X-Admin-Key`. The local CLI remains available to an authorized operator.

## 12. Start the API

```bash
care-anxrag serve --host 127.0.0.1 --port 8000
```

Endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Minimal service health; internal paths/details are not exposed |
| `POST` | `/v1/ask` | Grounded answer |
| `POST` | `/v1/retrieve` | Raw retrieval; admin-protected |
| `POST` | `/v1/sync` | Synchronize sources; admin-protected |
| `GET` | `/v1/stats` | Corpus counts and integrity |
| `GET` | `/v1/sources` | Source states; admin-protected |
| `GET` | `/v1/review/staging` | Review queue; admin-protected |
| `POST` | `/v1/review/{id}/approve` | Promote/rollback; admin-protected |
| `POST` | `/v1/review/{id}/reject` | Reject; admin-protected |
| `POST` | `/v1/review/{id}/withdraw` | Withdraw the active version for a document; admin-protected |
| `POST` | `/v1/reconcile` | Rebuild/clean vectors; `?reset_embedding_index=true` explicitly rebinds a changed embedding model; admin-protected |

Example:

```bash
curl -s http://127.0.0.1:8000/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does the active evidence say about panic attacks?"}'
```

Admin example:

```bash
curl -s http://127.0.0.1:8000/v1/sync \
  -H "X-Admin-Key: $CARE_ADMIN_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"source_ids":["pubmed_anxiety"],"dry_run":false,"force":false}'
```

## 13. Run continuous updates

For a research workstation:

```bash
care-anxrag scheduler --poll-seconds 60 --project-root .
```

For production, use a process supervisor or external scheduler. Run one scheduler instance only, monitor failures, back up SQLite, and execute periodic reconciliation:

```bash
care-anxrag reconcile --project-root .
# Only after an intentional embedding model/dimension change:
care-anxrag reconcile --reset-embedding-index --project-root .
```

## 14. Evaluate the research system

Create a benchmark using the format in `data/benchmark/example.jsonl`, then run:

```bash
care-anxrag evaluate data/benchmark/example.jsonl --project-root .
```

Implemented metrics:

- Recall@5
- Precision@5
- Mean Reciprocal Rank
- nDCG@5
- abstention accuracy
- conflict-detection accuracy
- citation validity

Retrieval metrics are averaged only across benchmark records that contain `relevant_external_ids` or `relevant_source_ids`; intentionally out-of-domain/abstention-only records are reported separately through `retrieval_evaluable_count` and are not treated as retrieval failures.

For publishable research, add expert-rated correctness, faithfulness, clinical appropriateness, conflict resolution, authority precision, unsafe confidence, and subgroup analyses. See [Research protocol](docs/RESEARCH_PROTOCOL.md).

# CARE scoring and abstention

CARE ranking combines:

```text
semantic relevance
+ lexical relevance
+ RRF agreement
+ CrossEncoder relevance
+ source authority
+ evidence level
+ freshness
+ anxiety/population applicability
```

It is intentionally preceded by a separate relevance gate. Authority, freshness, or evidence hierarchy cannot rescue an unrelated passage.

The system abstains when:

- no active evidence is retrieved;
- the top evidence fails independent relevance;
- relevant evidence fails the CARE quality threshold;
- overall confidence is below threshold;
- independent-source diversity is insufficient;
- high-confidence evidence conflict remains unresolved.

All thresholds and weights must be calibrated on held-out development data and frozen before final evaluation.

# Source freshness and versioning

Every source has its own state:

```text
last_attempt_at
last_success_at
last_changed_at
ETag
Last-Modified
cursor
last_error
```

Every evidence version stores publication/update/retrieval timestamps, a content fingerprint, status, evidence level, authority, source, topic, and supersession link. A failed incremental record does not advance the source watermark, preventing silent data loss.

# Adding local evidence

Place only licensed, public-domain, or otherwise authorized files in `data/local/`. See [Local document format](docs/LOCAL_DOCUMENT_FORMAT.md).

No clinical corpus is bundled with this repository.

# Security and governance requirements

Before any non-local deployment:

1. Set a strong `CARE_ADMIN_KEY`; place the API behind TLS, authentication, rate limiting, and request-size controls.
2. Localize and clinically review `CARE_CRISIS_RESOURCE_TEXT` for every deployment region.
3. Keep raw retrieval, debug traces, staging content, and licensed evidence admin-only.
4. Review source terms, robots policies, licences, attribution requirements, and retention rules.
5. Do not log sensitive free-text questions unless approved, minimized, protected, and retained under a documented policy.
6. Pin container images, Python dependencies, model revisions, prompts, configuration, and benchmark versions.
7. Perform adversarial testing for prompt injection, corpus poisoning, citation mismatch, stale guidance, and unsafe confidence.
8. Obtain independent mental-health expert review and applicable ethics/regulatory approvals.

Administrative HTTP endpoints and debug retrieval fail closed with HTTP 503 until
`CARE_ADMIN_KEY` is non-empty. Ordinary CLI administration remains available locally.

See [Data governance](docs/DATA_GOVERNANCE.md), [Threat model](docs/THREAT_MODEL.md), and [Deployment](docs/DEPLOYMENT.md).

# Validation status

The repository is validated with deterministic local components so tests do not depend on network access, model downloads, or an Ollama process. External adapters are contract-tested with fakes and representative payloads.

The current environment cannot perform live NICE licensed API calls or download/run the optional Chroma and Sentence Transformers dependencies. Those production integrations are implemented and contract-tested, but must also be verified in your deployment environment with the supplied self-check and acceptance checklist.

# Licence

Apache License 2.0. Source documents, model weights, APIs, and datasets retain their own terms and licences.
