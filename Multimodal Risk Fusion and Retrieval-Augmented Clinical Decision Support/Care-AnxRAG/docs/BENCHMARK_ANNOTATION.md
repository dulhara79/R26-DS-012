# CARE-AnxRAG benchmark annotation guide

The benchmark must be created from the controlled evidence corpus. Do not invent gold medical claims or source IDs.

Start with the [reviewer package workflow](BENCHMARK_REVIEW_WORKFLOW.md) for blank
annotation sheets, independent development/test authoring, adjudication records,
compilation commands, and split-validation checks. The public 17-question scaffold
is development material; it is not an independently authored locked test set.

## Required workflow

1. Freeze a corpus snapshot and record the Git commit/configuration.
2. Draft questions across the predefined benchmark strata.
3. Have at least two qualified annotators independently label the questions.
4. Adjudicate disagreements before the locked test set is evaluated.
5. Tune weights and thresholds only on the development split.
6. Never change gold labels after seeing locked-test results unless the item is formally invalidated and the change is documented.

## JSONL fields

- `id`: unique benchmark item ID.
- `question`: clinician-style information need.
- `stratum`: evaluation subgroup such as `generalized_anxiety_disorder`, `wrong_treatment_trap`, `source_poisoning`, or `out_of_domain`.
- `split`: normally `development` or `test`.
- `intent`: annotation metadata for analysis; this does not control runtime routing.
- `anxiety_subtypes`: normalized subtype labels relevant to the question.
- `treatments`: normalized treatment/intervention labels explicitly requested by the question.
- `population`: normalized population label when explicit.
- `relevant_external_ids` / `relevant_source_ids`: gold evidence identifiers.
- `prohibited_external_ids` / `prohibited_source_ids`: known distractor or inappropriate evidence identifiers for this item.
- `gold_evidence_excerpts`: exact source-supported excerpts selected by annotators. Do not paraphrase.
- `prohibited_claims`: claims that the evidence does not support and that must not appear as medical conclusions.
- `must_abstain`: whether the correct system behavior is to withhold an evidence answer.
- `expects_conflict`: whether the frozen corpus contains a clinically material evidence conflict for this item.
- `annotator_ids`: pseudonymous reviewer identifiers.
- `adjudicated`: true only after disagreements have been resolved.

## Minimum strata

Cover at least:
- generalized anxiety disorder;
- panic disorder;
- social anxiety disorder;
- agoraphobia / specific phobia;
- psychological interventions;
- medication-information boundaries;
- population-specific questions;
- recent evidence;
- wrong-subtype traps;
- wrong-treatment traps;
- insufficient evidence;
- conflicting evidence;
- superseded/retracted evidence;
- source-poisoning/distractor evidence;
- paraphrase/lexical traps;
- out-of-domain questions;
- crisis/urgent routing.

## Metrics already produced

The evaluator reports:
- Recall@5;
- Precision@5;
- MRR;
- nDCG@5;
- abstention accuracy;
- conflict accuracy;
- citation validity;
- extractive faithfulness;
- gold-evidence coverage;
- prohibited-evidence intrusion rate;
- active-version accuracy;
- stale/superseded evidence intrusion rate;
- per-stratum counts and summary metrics.

`extractive_faithfulness` checks that non-abstained medical answer lines are drawn from returned citation excerpts rather than newly written medical prose.

The automated metrics do not replace expert clinical evaluation.


## Gold-evidence coverage

When `gold_evidence_excerpts` is populated, the evaluator measures the fraction of adjudicated exact evidence excerpts that appear in the returned extractive answer. These excerpts must be copied exactly from the frozen source snapshot; do not write paraphrased gold answers.

The evaluator checks extractive fidelity against the full retrieved source chunk, not the shortened citation preview shown to users.


## Version-safety metrics

For each query with retrieved evidence, the evaluator reports:
- `active_version_accuracy`: fraction of evaluated top-5 hits whose document status is `active`;
- `stale_evidence_intrusion_rate`: fraction of evaluated top-5 hits marked `superseded` or `withdrawn`.

These metrics are observational. They do not relabel evidence and do not substitute for the ingestion/version-governance controls.
