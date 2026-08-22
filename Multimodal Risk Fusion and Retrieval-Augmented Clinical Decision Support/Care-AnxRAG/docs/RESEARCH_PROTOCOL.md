# Research and evaluation protocol

## Proposed research question

Does contradiction-, authority-, reliability-, evidence-, and freshness-aware retrieval improve the faithfulness and calibrated safety of anxiety information QA compared with conventional similarity-only and hybrid RAG?

## Baselines

- B0: generator without retrieval.
- B1: dense vector top-k.
- B2: dense + BM25 + RRF.
- B3: B2 + CrossEncoder reranker.
- B4: B3 + authority/evidence/freshness/applicability CARE score.
- B5: B4 + contradiction handling.
- CARE-AnxRAG: B5 + independent relevance gate + calibrated abstention + version-aware corpus.

Keep generator, corpus snapshot, prompt, chunking, and evaluation questions constant across baselines.

## Benchmark strata

- generalized anxiety disorder;
- panic disorder;
- social anxiety disorder;
- agoraphobia/specific phobia;
- symptoms and differential caution;
- psychological interventions;
- medication information boundaries;
- recent research;
- population-specific questions;
- out-of-domain questions;
- insufficient-evidence questions;
- contradictory evidence;
- outdated/superseded evidence;
- source-poisoning/distractor evidence;
- paraphrase and lexical traps;
- prompt injection in retrieved documents;
- crisis and urgent safety cases.

## Annotation

Each item should contain:

- question and intent;
- anxiety subtype/population;
- relevant source/document/chunk IDs;
- gold evidence excerpts;
- answerable vs must-abstain;
- expected conflict status;
- gold answer or scoring rubric;
- prohibited claims;
- annotator IDs and adjudication result.

Use at least two qualified annotators for clinical/evidence labels, report agreement, and adjudicate disagreements.

## Retrieval metrics

- Recall@k
- Precision@k
- MRR
- nDCG@k
- source/evidence authority precision
- active-version accuracy
- poisoned-evidence intrusion rate

## Generation metrics

- expert answer correctness;
- faithfulness/unsupported claim rate;
- citation entailment and completeness;
- clinical appropriateness;
- uncertainty calibration;
- appropriate abstention;
- unsafe confidence rate;
- conflict identification/resolution accuracy;
- source freshness/version correctness.

Automated LLM graders may supplement but must not replace human evaluation for the primary safety/clinical outcomes.

## Calibration

Split data into development and locked test sets. Tune CARE weights, relevance threshold, confidence threshold, contradiction threshold, dominance margin, and source-diversity requirement only on development data. Freeze all values before final testing.

## Statistical analysis

- paired bootstrap confidence intervals for retrieval/generation metric differences;
- McNemar test for paired binary outcomes such as correct abstention;
- correction for multiple primary comparisons;
- subgroup performance and error analysis;
- report effect sizes and confidence intervals, not only p-values.

## Reproducibility

Record code commit, source snapshot, active version IDs, exact model revisions, hardware, seeds, prompt, configuration, and dependency lock. Run each stochastic generation condition multiple times or use deterministic settings where supported.

## Benchmark JSONL

```json
{"id":"q001","question":"Example question","relevant_external_ids":["doc-id"],"relevant_source_ids":[],"must_abstain":false,"expects_conflict":false}
```
