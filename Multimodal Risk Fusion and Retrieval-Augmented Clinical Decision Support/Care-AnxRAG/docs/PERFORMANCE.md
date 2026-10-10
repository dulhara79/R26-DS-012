# Performance profiling

CARE-AnxRAG records stage timings in milliseconds so optimization decisions can be based on measured bottlenecks rather than assumptions.

## Retrieval timings

Every `RetrievalResult` contains:

- `analysis`
- `embedding`
- `dense_search`
- `lexical_search`
- `fusion`
- `reranking`
- `care_scoring`
- `nli_conflict`
- `selection`
- `total`

Stages that are intentionally skipped by an ablation mode report `0.0`.

## Answer timings

Every `AnswerResponse` contains:

- `retrieval`
- `presentation`
- `grounding`
- `total`

The presentation stage is extractive formatting, not free-form medical generation.

## Profiling workflow

Run a representative benchmark against the production model stack and save the JSON output. Compare medians and upper-tail latency by stage before changing configuration.

Do not reduce candidate counts, context size, reranker depth, NLI coverage, or model size solely to improve speed. Any performance change must be evaluated on the same frozen benchmark and corpus snapshot so retrieval quality, abstention, prohibited-evidence intrusion, and extractive faithfulness can be compared before and after the change.

A useful optimization record should include:

- code commit;
- corpus snapshot/version;
- benchmark version;
- retrieval mode;
- model revisions;
- hardware;
- candidate counts;
- per-stage latency;
- retrieval/evidence metrics.

This instrumentation does not itself prove production latency targets. Live measurements on the deployment hardware are still required.
