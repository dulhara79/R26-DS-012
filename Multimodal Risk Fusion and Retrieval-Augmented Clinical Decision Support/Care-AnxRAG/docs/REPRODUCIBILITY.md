# Experiment reproducibility snapshots

Formal CARE-AnxRAG experiments should be accompanied by a machine-readable snapshot of the exact evidence and runtime configuration used for that run.

## Create a snapshot

From the repository revision used for the experiment:

```bash
care-anxrag snapshot-experiment \
  artifacts/experiment-snapshot.json \
  --code-revision "$(git rev-parse HEAD)" \
  --benchmark data/benchmark/anxiety-test.jsonl \
  --project-root .
```

The command requires an explicit code revision. This prevents a formal experiment from silently omitting its source-code identity.

## Captured state

The snapshot records:

- code revision supplied by the operator;
- database schema version;
- source-registry SHA-256;
- optional benchmark path and SHA-256;
- dependency-manifest SHA-256 when `pyproject.toml` is available;
- every active evidence version's:
  - version ID;
  - document ID;
  - source ID;
  - external ID;
  - content hash;
  - status;
  - knowledge layer;
  - evidence level;
  - publication/update/retrieval timestamps;
- embedding provider/model/dimensions;
- runtime and stored embedding identities;
- reranker provider/model;
- NLI provider/model;
- extractive answer-provider identity;
- retrieval candidate counts;
- RRF parameter;
- CARE/relevance/conflict/grounding thresholds;
- freshness half-lives;
- CARE weights;
- actual runtime chunk-size and overlap settings.

The snapshot stores evidence fingerprints, not source text.

## Sensitive data

Administrative keys, source credentials, API tokens, passwords, raw user questions, and source document text are not written into the snapshot.

The source registry is represented by a cryptographic hash rather than copied into the result. Preserve the corresponding registry file in the controlled experiment archive.

## Experiment freeze procedure

Before a locked evaluation:

1. Freeze and review the benchmark test split.
2. Record the Git commit.
3. Reconcile the vector index and verify health.
4. Create the experiment snapshot.
5. Run the benchmark without tuning.
6. Save the raw evaluation report beside the snapshot.
7. Keep the active evidence corpus unchanged until the run is complete.
8. If any code, corpus, model, threshold, weight, or chunking configuration changes, create a new snapshot and treat it as a different experiment.

## Interpretation

A snapshot improves reproducibility but does not prove that the underlying evidence labels, benchmark annotations, or clinical judgments are correct. Those require their own review and adjudication records.
