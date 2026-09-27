# Safety-router benchmark protocol

This benchmark evaluates the deterministic pre-retrieval safety router. It is not a substitute for clinical validation, emergency-service policy, or local crisis-resource review.

## Benchmark record format

Store one JSON object per line.

Required fields:

- `id`: stable unique record identifier.
- `text`: the exact input evaluated by the router.
- `expected_level`: one of `normal`, `urgent`, or `crisis`.

Recommended review metadata:

- `stratum`: scenario family used for subgroup reporting.
- `split`: `development` or locked `test`.
- `annotator_ids`: reviewer identifiers.
- `adjudicated`: true only after disagreements have been resolved.

## Required strata

A controlled benchmark should include, at minimum:

- direct first-person self-harm or suicide language;
- indirect or ambiguous first-person risk language;
- negated self-harm language;
- historical self-harm references;
- third-person statements;
- academic/research questions about suicide or self-harm;
- urgent physical-symptom language;
- negated urgent physical symptoms;
- ordinary anxiety information questions;
- adversarial wording and paraphrases.

Do not create labels by asking the same router or an LLM to label its own benchmark.

## Annotation process

1. Freeze the benchmark source text before model/rule tuning.
2. Use at least two qualified reviewers for safety-critical labels.
3. Preserve the original independent labels.
4. Adjudicate disagreements and record the final label.
5. Tune patterns only on the development split.
6. Freeze the router before evaluating the locked test split.
7. Report subgroup results, not only aggregate accuracy.

Any benchmark containing real patient/user text requires an approved privacy, consent, retention, and de-identification process. Prefer purpose-built, reviewed synthetic cases when appropriate.

## Metrics

Run:

```bash
care-anxrag evaluate-safety path/to/safety-benchmark.jsonl
```

The report includes:

- overall accuracy;
- crisis recall;
- urgent recall;
- normal false-positive rate;
- full expected-vs-predicted confusion matrix;
- per-stratum accuracy;
- per-item predictions and routing reasons.

For safety-critical evaluation, aggregate accuracy is not sufficient. A system can achieve high overall accuracy while missing rare crisis cases, so crisis/urgent recall and false-positive behavior must be reported separately.

## Interpretation

This evaluator measures the current rule-based router. It does not prove clinical safety. Final deployment decisions require localized crisis policy, independent expert review, and testing representative of the intended population and setting.
