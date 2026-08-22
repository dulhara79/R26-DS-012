# Deployment guide

## Recommended topology

```text
TLS / identity-aware reverse proxy
              |
          FastAPI app
       /        |        \
 SQLite      Chroma      Ollama
 ledger      vectors     models
              |
       controlled scheduler
```

For a single-node research deployment, SQLite and embedded Chroma can share persistent local storage. For higher concurrency, separate ingestion jobs from query serving and evaluate a server-backed vector deployment.

## Mandatory configuration

- non-empty, rotated admin credential;
- localized crisis/urgent response text;
- TLS and authenticated administrative routes;
- explicit CORS policy if a separate frontend is used;
- reverse-proxy request limits and rate limiting;
- restricted filesystem permissions for `.env`, SQLite, Chroma, and model cache;
- regular encrypted backups;
- monitoring for sync failures, vector outbox failures, integrity errors, and degraded Ollama health.

## Model preparation

Pre-pull Ollama models and pre-cache/pin Sentence Transformers revisions. Do not allow uncontrolled model upgrades during an experiment.

## Acceptance sequence

1. Install dependencies from a lock file.
2. Run `care-anxrag selfcheck --offline`.
3. Run all tests.
4. Start Ollama and verify models.
5. Run `care-anxrag init`, review the reported scaffold actions, and require health `ok`. Existing operator files are preserved.
6. Synchronize one source in dry-run mode.
7. Synchronize and review counts/content hashes.
8. Run a fixed acceptance benchmark.
9. Test that administrative/raw-retrieval routes return `503` with no configured admin key, `401` with a wrong key, and succeed only with the configured key.
10. Test crisis/urgent routes using the approved localized copy.
11. Back up and restore the database/index.
12. Test source failure, vector failure, rollback, manual withdrawal, and reconciliation.
13. Verify that changing the embedding model/dimension causes a fail-closed identity mismatch; then test the deliberate `reconcile --reset-embedding-index` migration path.

## Scheduler

The built-in scheduler is appropriate for research and single-process operation. Production deployments should use an external scheduler with distributed locking so only one sync job runs at a time.

## Backups

Back up:

- SQLite database including WAL checkpoint;
- Chroma directory;
- source registry;
- environment/configuration secrets through a secret manager;
- pinned model/cache manifest;
- benchmark and reports.

SQLite is authoritative; if Chroma is lost, restore SQLite and run `care-anxrag reconcile` to regenerate active vectors. If the configured embedding model or dimension has intentionally changed, use `care-anxrag reconcile --reset-embedding-index` instead; this deletes all existing vectors and explicitly binds the rebuilt index to the new embedding identity.

## Container use

A Dockerfile and Compose file are provided as a starting point, not a hardened production platform. Pin image digests and scan images before deployment.

## Administrative fail-closed behavior

The local CLI is intended for an authorized host operator. HTTP administrative routes, raw retrieval, and debug-bearing answers are disabled with HTTP 503 until `CARE_ADMIN_KEY` is non-empty. Once configured, they require a matching `X-Admin-Key`; do not expose them directly to the public internet.
