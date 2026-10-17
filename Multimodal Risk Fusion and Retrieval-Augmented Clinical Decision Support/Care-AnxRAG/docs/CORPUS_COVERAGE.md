# Corpus coverage audit

CARE-AnxRAG is extractive-only for medical claims. That means corpus gaps must be visible rather than hidden by generated prose.

The coverage audit examines **active evidence chunks only** and reports which anxiety subtype/treatment combinations have direct support.

## Command

Audit the full active corpus:

```bash
care-anxrag coverage --project-root .
```

Check one requested combination:

```bash
care-anxrag coverage \
  --subtype generalized_anxiety_disorder \
  --treatment cognitive_behavioral_therapy \
  --project-root .
```

The JSON output includes:

- active chunk and document counts;
- observed subtype counts;
- observed treatment counts;
- direct subtype/treatment combinations;
- supporting chunk, document, and source counts;
- evidence levels represented by each combination;
- `requested_supporting_chunks` when both a subtype and treatment are supplied.

## Evidence rule

A subtype/treatment combination is counted only when the **same active chunk** supports both concepts.

Treatment support is taken from source text/title. Treatment topic tags alone are not accepted as evidence of treatment coverage.

Subtype topic metadata may help identify the anxiety subtype because subtype topics are already part of the controlled source metadata used by the retrieval pipeline.

Superseded and withdrawn chunks are excluded because the audit reads only active chunks.

## Interpretation

A zero count means the active corpus has no direct chunk-level evidence for that requested combination under the current deterministic concept matcher. It does not mean the treatment is clinically ineffective; it means the current CARE-AnxRAG corpus does not contain direct support that the system is allowed to surface.
