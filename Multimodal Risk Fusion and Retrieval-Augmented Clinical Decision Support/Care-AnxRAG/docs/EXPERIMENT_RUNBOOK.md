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

Prepare and adjudicate both splits using the
[reviewer workflow](BENCHMARK_REVIEW_WORKFLOW.md). Run `benchmark-validate-splits`
before recording the human test lock. That command checks metadata and duplicate
questions without evaluating the test set; human review must also check semantic
overlap and evidence correctness.

Tune thresholds only against the development split. Freeze code, configuration,
corpus, and model state before running a human-adjudicated locked test. Locked-test
results are evidence to report, not feedback for further tuning.

After the human lock and a successful final `freeze-corpus` audit on the fixed
code/configuration/model/corpus state, run:

```bash
care-anxrag experiment-final /controlled-review/anxiety-test.jsonl \
  artifacts/experiments/test-001 \
  --corpus-freeze artifacts/final-corpus.json \
  --code-revision "$(git rev-parse HEAD)" --project-root .
```

Replace the restricted path with the custodian's actual locked test location.
The existing final preflight verifies the freeze and review metadata; the split
separation and human lock above remain required procedural checks.
