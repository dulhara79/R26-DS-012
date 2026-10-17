# Benchmark data status

`example.jsonl` is synthetic schema/test data, not a research benchmark.

`review/development.csv` contains the 17 public, unreviewed development candidates.
`review/test.csv` is an empty header-only template for independently authored test
items. `review/manifest.json` records the pending status and seed strata. All human
review fields are intentionally blank. No locked test items are included.

Follow [the reviewer workflow](../../docs/BENCHMARK_REVIEW_WORKFLOW.md) to generate
working copies, annotate independently, adjudicate, compile, check split leakage,
and record a human lock. Keep real held-out test content and reviewer records in
restricted study storage. Do not treat development or synthetic examples as final
research evidence.
