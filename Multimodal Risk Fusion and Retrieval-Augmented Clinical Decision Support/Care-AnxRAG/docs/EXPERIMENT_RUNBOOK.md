# Experiment Runbook

CARE-AnxRAG research runs should be executed as immutable bundles so the ablation,
coverage, timing, and reproducibility evidence all refer to the same code, corpus,
benchmark, and runtime configuration.

## Run

From the exact code revision intended for an experiment:

```bash
care-anxrag experiment-bundle \
  data/benchmark/anxiety-development.jsonl \
  artifacts/experiments/development-001 \
  --code-revision "$(git rev-parse HEAD)" \
  --project-root .
```

The output directory must be empty. CARE-AnxRAG refuses to overwrite a completed
non-empty run directory.

## Artifacts

Each run produces:

- `ablation.json`: B0 dense-only through B5 CARE+conflict plus `CARE_full`.
- `coverage.json`: active-corpus clinical coverage at run time.
- `timings.json`: per-item answer/retrieval timings and stage summaries.
- `snapshot.json`: code, benchmark, corpus, model, retrieval, and chunking provenance.
- `manifest.json`: benchmark fingerprint and SHA-256/size metadata for the other artifacts.

Do not edit generated experiment artifacts by hand. If the corpus, benchmark,
configuration, models, or code changes, create a new run directory.

## Development and locked test

Tune thresholds only against the development split. Freeze code, configuration,
corpus, and model state before running a human-adjudicated locked test. Locked-test
results are evidence to report, not feedback for further tuning.
