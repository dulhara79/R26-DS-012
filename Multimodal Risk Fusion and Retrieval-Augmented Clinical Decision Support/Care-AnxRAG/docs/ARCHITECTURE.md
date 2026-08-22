# CARE-AnxRAG architecture

## Design principles

1. **SQLite is authoritative.** Chroma and FTS are indexes, not the source of truth.
2. **Relevance and trust are separate.** A highly authoritative source cannot answer an unrelated query.
3. **Updates are versioned, never destructively overwritten.** Active, staging, superseded, rejected, and withdrawn states are retained.
4. **Research evidence is gated.** New does not mean clinically authoritative.
5. **Conflicts are explicit.** The system either suppresses weaker conflicting evidence or abstains.
6. **Generation is downstream of evidence selection.** The LLM cannot repair poor retrieval and is never allowed to invent citations.
7. **Safety routing precedes RAG.** Crisis and urgent signals bypass ordinary retrieval/generation.

## Components

### Source registry

`config/sources.yaml` defines connector type, schedule, layer, authority, default evidence level, publication behavior, and connector-specific settings. Source IDs are unique and validated at startup. The registry root and every source/settings mapping are type-checked, disabled-source selections are rejected, and credential-like values must be indirect `*_env` references rather than plaintext registry secrets.

### Connectors

- `HttpPageConnector`: conditional requests with ETag and Last-Modified, HTML main-content extraction, section parsing, date extraction, retry/backoff, and rate limiting.
- `PubMedConnector`: ESearch by modification date followed by EFetch XML; overlap windows protect against boundary delays; hard truncation guard prevents watermark loss.
- `PmcConnector`: PMC XML full text with explicit licence allowlist metadata.
- `NiceSyndicationConnector`: generic JSON/XML parser for a licensed NICE feed; disabled until endpoint-specific acceptance tests pass.
- `LocalFilesConnector`: authorized Markdown/text/HTML/JSON content with optional front matter.

### Provenance ledger

SQLite stores:

- source configuration and update state;
- synchronization runs;
- stable document identity;
- immutable document versions;
- sections and section hashes;
- chunks and status;
- embedding cache;
- vector outbox;
- application metadata.

`document_id = hash(source_id, external_id)` and `version_id = hash(document_id, content_fingerprint)`.

### Version lifecycle

```text
fetched
  -> rejected / withdrawn
  -> staging
       -> active
       -> rejected
active update
  -> old active becomes superseded
  -> new staging becomes active
rollback
  -> selected superseded version is reindexed and activated
manual withdrawal / retraction
  -> the document's active version becomes withdrawn
  -> its vectors are removed and no previous version is silently reactivated
```

New vectors are written before a version becomes active. Superseded/withdrawn/rejected vectors are then deleted through the durable outbox. If physical deletion fails, SQLite status still prevents stale evidence from being returned. `reconcile` re-upserts every active vector and removes all non-active or orphan vector IDs. SQLite also stores the configured embedding model identity; retrieval and indexing fail on a mismatch. Only `reconcile --reset-embedding-index` may purge, rebind, and rebuild the index after an intentional embedding model/dimension change.

### Chunking

The chunker is heading/section-aware and sentence-aware. Defaults are 180 words, 35-word overlap, and 35-word minimum tail. Unchanged section hashes reuse previous chunk text and the embedding cache avoids recomputation. New versions still receive new chunk IDs so provenance remains immutable.

### Dense index

Production uses two Chroma collections:

- `care_clinical_core`
- `care_research_frontier`

Cosine HNSW is configured explicitly. Embeddings are supplied by CARE-AnxRAG rather than an implicit collection embedding function, preventing accidental model changes.

The SQLite exact-cosine backend exists for deterministic tests and small offline smoke checks only.

### Lexical index

SQLite FTS5 stores title, section heading, text, topics, and source name. BM25 rank is used as an independent lexical channel. Query stop words are removed before FTS expression construction.

### Hybrid retrieval

1. Analyze query intent, anxiety subtype, population, desired recency, and safety.
2. Search preferred knowledge layers using dense vectors.
3. Search FTS5 using BM25.
4. Fuse ranks using Reciprocal Rank Fusion.
5. Rerank top candidates with a CrossEncoder.
6. Calculate an independent relevance score.
7. Drop candidates below `minimum_relevance_score`.
8. Calculate CARE evidence quality.
9. Detect pairwise evidence relations on a diversified candidate set.
10. Suppress weaker contradictions or mark unresolved conflict.
11. Select diverse context and calculate confidence.
12. Abstain when any configured safety/evidence condition fails.

### Scores

Independent relevance gate:

```text
0.35 * dense
+ 0.45 * reranker
+ 0.15 * meaningful lexical presence
+ 0.05 * applicability
```

Default CARE score:

```text
0.20 semantic
+ 0.08 lexical
+ 0.07 RRF
+ 0.25 reranker
+ 0.14 authority
+ 0.13 evidence level
+ 0.06 freshness
+ 0.07 applicability
```

The weights sum to 1.0 and are validated at startup. Defaults are engineering priors, not clinically validated values.

### Contradiction analysis

Production uses `cross-encoder/nli-deberta-v3-base`. CARE-AnxRAG discovers contradiction/entailment/neutral indices from the loaded model's `id2label` mapping rather than assuming a positional label order; known defaults are used only for the configured standard model, while unknown custom label schemes fail closed. The deterministic test classifier compares sentence-level propositions so unrelated negations do not mask a treatment contradiction.

For a high-confidence contradiction:

- if evidence-strength difference exceeds `dominance_margin`, the weaker passage is excluded;
- otherwise conflict remains unresolved and contributes to abstention.

Evidence strength combines source authority, evidence level, freshness, and CARE score.

### Confidence

Confidence combines top CARE score, top-three average, independent-source diversity, authority coverage, and consensus. It is not a clinical probability. It must be calibrated and interpreted only as a system-selection confidence.

### Generation

Ollama receives only selected evidence, each inside explicit untrusted-evidence delimiters. The generator must return JSON matching `GeneratedPayload`:

```json
{
  "answer": "... [S1]",
  "cited_source_ids": ["S1"],
  "uncertainty": null
}
```

Validation requires exact equality between inline citation IDs and the structured citation list. Unknown or missing citations trigger one repair call; a second failure causes abstention rather than returning an unvalidated answer.

### Safety boundary

The rule-based router detects first-person/imminent self-harm language and selected urgent physical symptoms. It intentionally does not classify an academic mention such as “research on suicide risk” as a personal crisis. This is a minimum safety layer, not a validated clinical classifier.

## Failure behavior

| Failure | Behavior |
|---|---|
| Source HTTP failure | Source state records error; watermark is not advanced |
| Individual ingestion/index failure | Run is partial; source cursor is preserved |
| NCBI result window exceeds cap | Sync fails visibly; no silent truncation |
| Dry run | Fetches/parses/validates and records an audit run, but does not ingest or advance source timestamps/cursors |
| Concurrent in-process sync | Second run is rejected instead of racing source watermarks |
| Embedding identity mismatch | Retrieval/indexing fails until an explicit reset-and-rebuild |
| Vector upsert failure | Version remains non-active; outbox retains retry |
| Vector delete failure | SQLite status blocks stale evidence; outbox retries |
| Reranker/NLI initialization failure | Startup fails rather than silently downgrading production mode |
| Irrelevant evidence | Independent relevance abstention |
| Unresolved evidence conflict | Abstention |
| Invalid LLM citations | Repair once, then abstain |
| Crisis/urgent query | Bypass RAG and return localized safety message |
