# Reviewer-ready development and held-out test workflow

## Current status and deliverables

This is an **unreviewed preparation package**, not a locked research benchmark.
`data/benchmark/example.jsonl` remains synthetic schema/test data. The checked-in
[review package](../data/benchmark/review/manifest.json) contains 17 public
**development candidates** and a header-only test CSV. No relevance labels,
reviewer identities, evidence excerpts, abstention/conflict decisions, or
adjudication results have been supplied. Final measured reporting remains blocked
until humans complete and lock the benchmark.

The existing 17 questions include broad prompts that need human refinement into
specific, answerable information needs. Their intent/subtype metadata are draft
scaffolding, not validated clinical annotations. A stratum name is not proof that
the corpus contains the evidence or adversarial condition needed for that case.

## Generate a working copy

From the repository root, with the project installed (`python -m pip install -e
'.[dev]'`):

```bash
care-anxrag benchmark-review-package artifacts/benchmark-review-v1
```

The output directory must be empty; generation refuses to overwrite review work.
The command requires no runtime, database, model, or network connection. It writes:

| File | Purpose | Initial state |
| --- | --- | --- |
| `development.csv` | Public questions for drafting and development tuning | 17 candidates; all human decision fields blank |
| `test.csv` | Independent held-out questions authored by the test custodian | Header only; no test items |
| `manifest.json` | Package status, candidate counts, and 17 seed strata | `awaiting_human_review` |

Do **not** generate both splits with `benchmark-scaffold`: that legacy command
copies the same public questions for either split. Changing the split or item ID
does not turn a development question into an independent test question.

## Assign roles and preserve independent records

1. The study owner names two qualified reviewers and a test custodian. Use real,
   stable pseudonymous reviewer IDs assigned by the study; retain the identity
   mapping privately. The compiler can check distinct strings, not qualifications
   or whether the people actually reviewed independently.
2. Record the intended corpus snapshot, code revision, configuration, model
   revisions, question version, inclusion criteria, split sizes, and stratum
   targets before annotation. The 17 seed strata are a starting point; consult
   [the annotation protocol](BENCHMARK_ANNOTATION.md) for broader coverage,
   including wrong-subtype and wrong-treatment traps. Record exclusions and
   shortfalls explicitly; this package does not establish sufficient sample size.
3. Refine development candidates and assign stable `dev-...` IDs. The custodian
   independently authors specific `test-...` questions in `test.csv`, sets every
   row's `split` to `test`, and completes draft intent/subtype/treatment/population
   fields. Do not copy development questions, rename them, or use paraphrases of
   the same information need as held-out cases. Humans must check semantic overlap.
4. Keep test questions, reviewer records, and gold decisions in custodian-controlled
   storage outside this public repository and outside the tuning team's workspace.
   The checked-in empty test template is not a locked set. Share it only with
   reviewers until code/configuration/model tuning is frozen.
5. Make one separate sheet per reviewer per split before they see one another's
   decisions. Each reviewer uses the `final_*` columns to record **their proposed
   decisions**, enters only their own `reviewer_1_id` or `reviewer_2_id`, and leaves
   `adjudicated` blank. Preserve these original sheets unchanged with dates and
   rationale notes keyed by item ID. They are not compilable adjudicated sheets.

## Select evidence against one controlled corpus

Resolve source-governance issues before freezing; do not promote material simply
to improve a benchmark score. Use the intended research runtime configuration,
not the synthetic offline self-check corpus, and record a successful audit:

```bash
care-anxrag freeze-corpus artifacts/annotation-corpus.json \
  --code-revision "$(git rev-parse HEAD)" --project-root .
```

A nonzero exit or `ready_to_freeze=false` blocks this workflow. The JSON is an audit
and fingerprint, **not a backup**: preserve the actual authorized evidence files,
SQLite ledger, vectors, registry, models, and configuration used for annotation.
Prevent synchronization/promotion while review is in progress. `ready_to_freeze`
is an engineering check, not approval of corpus clinical quality or completeness.

Reviewers inspect the full frozen source/chunks and independently judge relevance,
prohibited evidence/claims, abstention, and material conflict. For development
candidate inspection, retrieval output is available via:

```bash
care-anxrag retrieve 'What evidence describes panic disorder and its management?' \
  --project-root . > artifacts/development-candidate-retrieval.json
```

Ranked hits are navigation aids, not gold. Search the corpus independently too;
never use the system's answer, ranking, or abstention as its own gold label. Test
reviewers work from the controlled corpus without inspecting model/test outputs.

## CSV field contract and adjudication

Use UTF-8 CSV with the generated header. A CSV editor must preserve JSON strings
and quoting. Do not enter placeholder identities or invented excerpts.

| Columns | Human action |
| --- | --- |
| `id`, `question`, `stratum`, `split` | Assign a unique stable ID, specific question, documented stratum, and exact `development` or `test` split |
| `intent`, `anxiety_subtypes_json`, `treatments_json`, `population` | Confirm draft metadata; JSON fields are lists of strings; population may be blank |
| `reviewer_1_id`, `reviewer_2_id` | On the adjudicated copy only, enter the two distinct IDs of the actual independent reviewers |
| `final_relevant_external_ids_json`, `final_relevant_source_ids_json` | JSON lists of active document external IDs or registry source IDs judged relevant |
| `final_prohibited_external_ids_json`, `final_prohibited_source_ids_json` | JSON lists of active distractor/inappropriate evidence IDs; do not use inactive IDs as substitutes |
| `final_gold_evidence_excerpts_json` | JSON list of exact excerpts copied from active frozen chunks, with source/chunk provenance retained in review notes |
| `final_prohibited_claims_json` | JSON list of unsupported claims adjudicated by humans |
| `final_must_abstain`, `final_expects_conflict` | Explicit `true` or `false`; blank means undecided |
| `adjudicated` | Set `true` only after the independent decisions and disagreements are resolved and recorded |

Write `[]` for a deliberately empty list. The compiler also treats a blank list
field as empty, so it cannot distinguish an omitted decision from an intentional
empty list. The custodian must check all list decisions explicitly. Source IDs
label entire registry sources; use document external IDs when narrower labeling
is appropriate. Excerpt validation normalizes whitespace and checks substring
membership in any active chunk; humans must verify the excerpt belongs to the
selected relevant document and supports the actual question.

Compare every reviewer decision and record agreements/disagreements, reasons,
source/version/chunk references, adjudicator identity, date, and the final decision
in a separate adjudication log. Keep both original independent sheets. Save a new
`development-adjudicated.csv` and custodian-held `test-adjudicated.csv`; copy final
resolved values into those sheets. Do not silently fill gaps with defaults.
Neither review completeness nor clinical correctness is established by a passing
compiler alone.

## Compile and validate without opening test results

The custodian runs compilation against the same frozen runtime. Replace
`/controlled-review` with the actual restricted review directory:

```bash
care-anxrag benchmark-compile \
  artifacts/benchmark-review-v1/development-adjudicated.csv \
  artifacts/benchmark-review-v1/anxiety-development.jsonl --project-root .

care-anxrag benchmark-compile \
  /controlled-review/test-adjudicated.csv \
  /controlled-review/anxiety-test.jsonl --project-root .

care-anxrag benchmark-validate-splits \
  artifacts/benchmark-review-v1/anxiety-development.jsonl \
  /controlled-review/anxiety-test.jsonl
```

Compilation rejects missing/distinct-reviewer failures, absent/false adjudication,
non-active external/source IDs, unsupported excerpts, invalid JSON lists, missing
explicit booleans, and duplicate IDs. The compiler does not itself enforce split
separation. The new split validator rejects empty files, incorrect split values,
missing item identity/question/stratum, unreviewed metadata, repeated IDs, and
questions duplicated after whitespace/case normalization, within or across
splits. It reports per-split and per-stratum counts without retrieval or evaluation.
It does not validate evidence, enforce sample-size/coverage targets, detect
paraphrase leakage, authenticate review identities, or declare a benchmark locked.

Do not use `evaluate` to validate test labels: it runs the system on test questions.
The unfilled development sheet must fail compilation; the header-only test sheet
must fail with `Annotation sheet contains no benchmark rows`. This is intentional.

## Development tuning, human lock, and final handoff

Run development experiments only after its annotation and compilation:

```bash
care-anxrag experiment-bundle \
  artifacts/benchmark-review-v1/anxiety-development.jsonl \
  artifacts/experiments/development-001 \
  --code-revision "$(git rev-parse HEAD)" --project-root .
```

Complete tuning before releasing test questions/gold to experiment operators. The
custodian records final split validation, human semantic-overlap review, coverage
and exclusions, reviewer/adjudication logs, and SHA-256 fingerprints of both CSVs,
both compiled JSONLs, the corpus audit, and the full configuration/model/code
state. Freeze test bytes in restricted read-only storage and retain the lock date
and approvals. Do not commit real reviewer identity mappings or test labels here.

On that fixed final revision, repeat `freeze-corpus` with
`--benchmark /controlled-review/anxiety-test.jsonl` and a fresh audit output. Then
handoff to `experiment-final` in the [experiment runbook](EXPERIMENT_RUNBOOK.md).
Split validation and the human lock are required procedural checks; the existing
final-experiment preflight does not enforce development/test separation itself.
Do not tune after inspecting test results. If a gold item is invalidated or corpus,
models, code, or configuration change, document it and begin a new reviewed phase.
Create `FINAL_RESULTS.md`, `ERROR_ANALYSIS.md`, and `VIVA_DEMO.md` only after measured,
provenance-bound experiments exist; this preparation package produces none.

## Engineering verification of this package

```bash
python -m pytest tests/test_benchmark_review.py tests/test_cli_benchmark_review.py -q
python -m pytest -q
./scripts/validate.sh
```

These deterministic checks use synthetic test fixtures to test tooling. Passing
software tests or offline acceptance checks is not research validation.
