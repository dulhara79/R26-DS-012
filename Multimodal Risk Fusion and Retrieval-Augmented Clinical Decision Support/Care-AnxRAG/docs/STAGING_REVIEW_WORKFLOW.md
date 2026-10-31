# Staging evidence review workflow

CARE-AnxRAG blocks corpus freeze while unresolved staging versions remain. Use this
workflow to review those versions without editing SQLite directly or allowing an
old automated screening plan to make corpus decisions.

## Important rule

Historical files such as `pubmed_reject_plan.json` and
`pubmed_approve_plan.json` are **screening hints only** unless the study records
independently establish that they are authoritative human-reviewed decisions.

The current repository history shows those files were added with the initial
repository snapshot. Their `role`, `evidence_role`, and `score` values can
accelerate review, but the software never copies them into the final decision
column.

## 1. Generate a reviewer sheet

From the research workstation using the real local CARE database:

```bash
care-anxrag staging-review-package \
  artifacts/pubmed-staging-review.csv \
  --hints pubmed_reject_plan.json \
  --project-root .
```

The command includes every current staging version and writes immutable identity
fields plus optional historical hint columns. These columns remain blank until a
human reviewer completes them:

- `decision`: exactly `approve` or `reject`
- `decision_reason`: the review rationale
- `reviewer_id`: the actual reviewer identifier

The command refuses to overwrite a non-empty existing review CSV.

## 2. Review the evidence

For every row, inspect the actual staged source/evidence under the study's current
inclusion/exclusion rules. Do not accept a historical hint merely because it is
present.

Useful existing classifications include:

- core anxiety-disorder evidence;
- anxiety comorbidity/mixed-population evidence;
- procedural anxiety;
- anxiety as a secondary outcome;
- study protocols rather than completed outcome evidence;
- out-of-domain evidence.

A reviewer may agree or disagree with the historical hint.

## 3. Dry-run the completed review

Before mutation, validate every row against the current database:

```bash
care-anxrag staging-review-apply \
  artifacts/pubmed-staging-review.csv \
  --project-root .
```

Without `--apply`, the command is always dry-run. It requires:

- every current staging version is represented;
- no duplicate version IDs;
- exact current version ID;
- exact title;
- exact source ID;
- exact external ID;
- explicit `approve` or `reject`;
- non-empty decision reason;
- non-empty reviewer ID.

If staging changed after the sheet was generated, validation fails rather than
applying stale decisions.

## 4. Apply only after the dry-run is clean

```bash
care-anxrag staging-review-apply \
  artifacts/pubmed-staging-review.csv \
  --apply \
  --project-root .
```

All rows are validated before the first mutation. Approved versions use the normal
CARE activation/indexing path. Rejected versions use the normal rejection/vector
cleanup path.

## 5. Re-check the freeze gate

```bash
care-anxrag staging --project-root .

care-anxrag freeze-corpus artifacts/development-corpus.json \
  --code-revision "$(git rev-parse HEAD)" \
  --project-root .
```

Do not begin formal benchmark annotation until the corpus audit returns
`ready_to_freeze=true`.

## Scope

This workflow is an engineering guardrail for applying human review decisions.
It does not determine clinical inclusion/exclusion, authenticate reviewer
qualifications, or make an automated plan clinically authoritative.
