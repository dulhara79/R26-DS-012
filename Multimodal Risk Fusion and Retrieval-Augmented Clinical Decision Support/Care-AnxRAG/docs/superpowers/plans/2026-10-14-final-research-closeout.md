# CARE-AnxRAG Final Research Closeout Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close CARE-AnxRAG quickly and defensibly by fixing the last code hygiene defect, freezing the real evidence corpus, completing human-reviewed benchmarks, running reproducible development/locked-test experiments, producing final research artifacts, and freezing a viva-ready release.

**Architecture:** Preserve the current extractive-only medical-answer architecture. Retrieved evidence is ranked by the CARE pipeline, filtered by relevance/sufficiency/conflict logic, rendered as source sentences, and verified against cited source text. No final-stage work may reintroduce free-form medical claim generation.

**Tech Stack:** Python 3.11/3.12, SQLite/FTS5, ChromaDB, Ollama EmbeddingGemma, Sentence Transformers CrossEncoder + NLI, FastAPI, Typer, pytest, GitHub Actions.

**Spec:** `docs/RESEARCH_PROTOCOL.md`, `docs/BENCHMARK_ANNOTATION.md`, `docs/BENCHMARK_REVIEW_WORKFLOW.md`, `docs/EXPERIMENT_RUNBOOK.md`, `docs/DATA_GOVERNANCE.md`, `docs/CORPUS_COVERAGE.md`, `docs/SAFETY_BENCHMARK.md`, `docs/REPRODUCIBILITY.md`, `docs/ANSWER_SELECTION.md`

## Current Verified State

- `main`: `0b5d104beace5796ae088d9ab06f8e64786e3dd0`
- GitHub CI: green on current `main`
- Central sync workflow: green
- Core research stack is merged: extractive answer selection, exact-source grounding, ablations, timings, PubMed correction monitoring, clinical facets, joint sufficiency, safety evaluation, reproducibility snapshots, paired statistics, corpus coverage, corpus freeze, final-experiment preflight, benchmark reviewer package, and split validation
- PR #30 is open and green: removes duplicate Typer registrations, refreshes validation docs/report, and reports 216 passing tests with 80.3131% coverage
- PR #31 is open and green, stacked on #30: adds safe human staging-review package/apply workflow for the 47 unresolved PubMed staging versions
- PR #32 is open and green, independent: requires two distinct reviewers + adjudication for locked safety-test evaluation
- PR #29 is this planning PR
- Current reviewer package has 17 development candidates; held-out test content must still be independently authored/reviewed
- No final held-out experiment artifacts are committed
- `docs/FINAL_RESULTS.md`, `docs/ERROR_ANALYSIS.md`, and `docs/VIVA_DEMO.md` do not yet exist
- CI does not validate the real Chroma/Ollama/CrossEncoder stack, the user's live corpus, reviewer qualifications, or clinical correctness

## Global Constraints

- Never reintroduce free-form medical answer generation.
- Do not use NLI to authorize paraphrased medical answer text.
- Do not fabricate clinical gold labels, reviewer identities, evidence excerpts, adjudication outcomes, or final experimental numbers.
- Locked-test items require human review and adjudication before final evaluation.
- Tune only on development data.
- Once locked-test execution starts, no code/config/corpus/model tuning is permitted.
- Evidence may be promoted only through the existing governance/review process, never to improve a benchmark result directly.
- Every code change follows RED → GREEN → full-suite verification.
- Final results must bind code revision, corpus freeze, benchmark hashes, model/config state, and dependency state.

## Review Focus

- Duplicate Typer command registration must not survive into the final release.
- The final real corpus must contain no unresolved staging/integrity blocker.
- Development/test split leakage must be prevented procedurally and with existing split-validation tooling.
- Final metrics must come from immutable experiment artifacts, not manually copied estimates.
- Central-backend/viva behavior must be tested on the same final code/config used for reported results.

---

## Immediate Merge / Execution Order

1. Merge PR #30 first.
2. Retarget PR #31 from `fix/final-cli-command-dedup` to `main`, refresh it against the new `main`, rerun exact-head CI, then merge it.
3. Refresh PR #32 against the resulting `main`, rerun exact-head CI, then merge it.
4. Merge this planning PR #29 after the implementation PRs, or update it once more if any merge changes the closeout commands.
5. Stop feature development after these merges unless a deterministic development-set failure proves a release-blocking defect.

After that point, the critical path is operational/research work: staging review → corpus freeze → development benchmark review → development experiments/calibration → independently authored held-out test → locked safety test → one final experiment → latency/integration verification → final documents → release tag.

---

### Task 1: Remove duplicate CLI registrations and lock the command surface

**Files:**
- Modify: `src/care_anxrag/cli.py`
- Create: `tests/test_cli_contract.py`

**Interfaces:**
- Consumes: the current Typer `app`.
- Produces: exactly one registration for each public command.

- [ ] **Step 1: Write a failing test** asserting `evaluate-safety` and `snapshot-experiment` each appear exactly once and that all required research commands are present.
- [ ] **Step 2: Run `pytest tests/test_cli_contract.py -v`; confirm RED due to duplicate registrations.**
- [ ] **Step 3: Remove only the duplicate command definitions; preserve the canonical behavior already covered by `tests/test_safety_evaluation.py` and `tests/test_reproducibility.py`.**
- [ ] **Step 4: Run targeted CLI tests; require PASS.**
- [ ] **Step 5: Run `python -m compileall -q src && pytest -q && care-anxrag selfcheck --offline --project-root .`.**
- [ ] **Step 6: Run `./scripts/validate.sh`; regenerate `validation-report.json`.**
- [ ] **Step 7: Push and merge only after exact-head GitHub CI and central sync are green.**

### Task 2: Freeze the real research corpus on the working machine

**Files:**
- Generated local artifacts only.
- No code changes unless a reproducible defect is discovered.

**Interfaces:**
- Consumes: the real SQLite/Chroma state and configured production models.
- Produces: an auditable corpus freeze file that passes the final preflight.

- [ ] **Step 1: Pull the final green `main`.**
- [ ] **Step 2: Start/verify Ollama and the configured embedding model; verify CrossEncoder reranker and NLI load successfully on the actual Mac.**
- [ ] **Step 3: Run `care-anxrag stats --project-root .`; archive the JSON.**
- [ ] **Step 4: Run `care-anxrag coverage --project-root .`; archive the JSON.**
- [ ] **Step 5: Run focused coverage checks for benchmark-relevant combinations: GAD+CBT, GAD+MCT, social-anxiety+CBT, panic+CBT/exposure.**
- [ ] **Step 6: Run `care-anxrag staging --project-root .`; every pending version must be reviewed, rejected, or explicitly documented before freeze.**
- [ ] **Step 7: Resolve any PubMed retraction/correction evidence alerts before freeze.**
- [ ] **Step 8: Run `care-anxrag reconcile --project-root .` and database integrity checks.**
- [ ] **Step 9: Run:**

```bash
care-anxrag freeze-corpus artifacts/development-corpus.json \
  --code-revision "$(git rev-parse HEAD)" \
  --project-root .
```

- [ ] **Step 10: Require `ready_to_freeze=true`. Do not begin formal annotation if the freeze gate is blocked.**

### Task 3: Complete the development benchmark review

**Files:**
- Working copy generated from: `data/benchmark/review/development.csv`
- Generated: `artifacts/benchmark-review-v1/development-adjudicated.csv`
- Generated: `artifacts/benchmark-review-v1/anxiety-development.jsonl`

**Interfaces:**
- Consumes: frozen development corpus.
- Produces: adjudicated development benchmark.

- [ ] **Step 1: Generate a clean working reviewer package with `care-anxrag benchmark-review-package artifacts/benchmark-review-v1`.**
- [ ] **Step 2: Refine the 17 development candidates into specific information needs while retaining intended strata.**
- [ ] **Step 3: Have two qualified reviewers annotate independently; preserve their original sheets unchanged.**
- [ ] **Step 4: Adjudicate disagreements and record reviewer IDs, evidence IDs, exact evidence excerpts, must-abstain/conflict labels, prohibited evidence/claims, and adjudication status.**
- [ ] **Step 5: Compile:**

```bash
care-anxrag benchmark-compile \
  artifacts/benchmark-review-v1/development-adjudicated.csv \
  artifacts/benchmark-review-v1/anxiety-development.jsonl \
  --project-root .
```

- [ ] **Step 6: Do not treat compiler success as clinical validation; manually review evidence/question alignment.**

### Task 4: Run development experiments and calibrate only what is justified

**Files:**
- Generated: `artifacts/experiments/development-001/`

**Interfaces:**
- Consumes: frozen development corpus + adjudicated development benchmark.
- Produces: B0–B5 + CARE_full metrics, paired comparisons, timings, coverage, snapshot, manifest.

- [ ] **Step 1: Run:**

```bash
care-anxrag experiment-bundle \
  artifacts/benchmark-review-v1/anxiety-development.jsonl \
  artifacts/experiments/development-001 \
  --code-revision "$(git rev-parse HEAD)" \
  --project-root .
```

- [ ] **Step 2: Inspect `ablation.json`, `comparisons.json`, `coverage.json`, `timings.json`, `snapshot.json`, and `manifest.json`.**
- [ ] **Step 3: Classify each error as retrieval defect, sufficiency defect, answer-selection defect, annotation issue, or corpus gap.**
- [ ] **Step 4: Change code only when a deterministic failing regression test proves a real defect.**
- [ ] **Step 5: For genuine corpus gaps, stage/review evidence through normal governance and create a new development corpus freeze.**
- [ ] **Step 6: Tune thresholds only if development evidence supports it; document every changed value and rationale.**
- [ ] **Step 7: Freeze code/config/models/corpus once development changes stop.**

### Task 5: Build the independent held-out test benchmark

**Files:**
- Restricted/custodian-controlled: `test.csv`, reviewer sheets, adjudication log, compiled `anxiety-test.jsonl`

**Interfaces:**
- Consumes: frozen final corpus and benchmark protocol.
- Produces: independent locked test set.

- [ ] **Step 1: The test custodian independently authors test questions; do not copy or paraphrase development questions.**
- [ ] **Step 2: Cover the same research strata, recording any deliberate shortfalls.**
- [ ] **Step 3: Two reviewers annotate independently without seeing system outputs.**
- [ ] **Step 4: Adjudicate and compile the held-out test JSONL.**
- [ ] **Step 5: Run:**

```bash
care-anxrag benchmark-validate-splits \
  artifacts/benchmark-review-v1/anxiety-development.jsonl \
  /controlled-review/anxiety-test.jsonl
```

- [ ] **Step 6: Human-review semantic overlap between splits; the automated validator cannot detect paraphrase leakage reliably.**
- [ ] **Step 7: Record hashes, lock date, reviewer/adjudication records, and keep test gold in restricted read-only storage.**

### Task 6: Build and run the formal safety benchmark

**Files:**
- Restricted/local: `data/benchmark/safety-development.jsonl`
- Restricted/local: `data/benchmark/safety-test.jsonl`

**Interfaces:**
- Consumes: deterministic `SafetyRouter`.
- Produces: crisis recall, urgent recall, normal false-positive rate, confusion matrix, error rows.

- [ ] **Step 1: Create/review cases across crisis, urgent, normal, negation, academic-context, ambiguous, and lexical-trap strata.**
- [ ] **Step 2: Use development safety cases for any rule fixes.**
- [ ] **Step 3: Freeze the router before held-out safety evaluation.**
- [ ] **Step 4: Run `care-anxrag evaluate-safety ...` on the locked safety test set.**
- [ ] **Step 5: Save every false-negative/false-positive case for final error analysis.**

### Task 7: Freeze final corpus and execute the held-out experiment exactly once

**Files:**
- Generated: `artifacts/final-corpus.json`
- Generated: `artifacts/experiments/test-001/`

**Interfaces:**
- Consumes: final code/config/models/corpus + locked test benchmark.
- Produces: immutable final research evidence.

- [ ] **Step 1: Repeat corpus freeze against the final held-out benchmark:**

```bash
care-anxrag freeze-corpus artifacts/final-corpus.json \
  --benchmark /controlled-review/anxiety-test.jsonl \
  --code-revision "$(git rev-parse HEAD)" \
  --project-root .
```

- [ ] **Step 2: Require `ready_to_freeze=true`.**
- [ ] **Step 3: Run exactly once:**

```bash
care-anxrag experiment-final \
  /controlled-review/anxiety-test.jsonl \
  artifacts/experiments/test-001 \
  --corpus-freeze artifacts/final-corpus.json \
  --code-revision "$(git rev-parse HEAD)" \
  --project-root .
```

- [ ] **Step 4: Do not tune after seeing test results. Any later change starts a new experimental phase.**

### Task 8: Measure real-viva latency and integration behavior

**Files:**
- Generated timing notes/artifacts only.

**Interfaces:**
- Consumes: final code/config on the actual presentation Mac.
- Produces: median/p95 latency, stage bottlenecks, end-to-end integration evidence.

- [ ] **Step 1: Run representative final queries with the real Chroma/Ollama/CrossEncoder stack and collect `timings_ms`.**
- [ ] **Step 2: Report median and p95 total latency plus dominant stages.**
- [ ] **Step 3: Do not optimize after locked-test execution unless a release-blocking defect exists.**
- [ ] **Step 4: Start CARE on the integration port used by the central backend and verify `/health`.**
- [ ] **Step 5: Verify central-backend → CARE normal evidence request, abstention request, and error behavior.**
- [ ] **Step 6: Save status/latency evidence for the viva.**

### Task 9: Produce final research and viva documents from measured artifacts

**Files:**
- Create: `docs/FINAL_RESULTS.md`
- Create: `docs/ERROR_ANALYSIS.md`
- Create: `docs/VIVA_DEMO.md`
- Update: `README.md` only with measured final claims.

**Interfaces:**
- Consumes: immutable final experiment/safety/latency artifacts.
- Produces: traceable research reporting.

- [ ] **Step 1: Build B0–B5 + CARE_full results tables from `test-001` artifacts only.**
- [ ] **Step 2: Report paired effect differences, confidence intervals, and McNemar results exactly as generated.**
- [ ] **Step 3: Write error analysis by subtype, treatment, abstention, contradiction, stale evidence, distractors, answer-selection omissions, and safety-router errors.**
- [ ] **Step 4: State prototype limitations explicitly: curated corpus scale, reviewer/sample-size limitations, no clinical-device validation, hardware/model dependence.**
- [ ] **Step 5: Keep the contribution claim focused on evidence-aware retrieval/selection, provenance/versioning, conflict handling, sufficiency/abstention, and deterministic extractive grounding.**
- [ ] **Step 6: Create a five-minute viva demo script with preselected queries and expected categories of behavior; do not manually edit system outputs.**

### Task 10: Freeze the viva release

**Files:**
- No feature changes after freeze except a release-blocking defect.

**Interfaces:**
- Consumes: final green main + final artifacts.
- Produces: exact tagged release used for thesis/viva.

- [ ] **Step 1: Run `./scripts/validate.sh` on the actual presentation machine.**
- [ ] **Step 2: Run real-model health checks, database integrity, vector reconciliation, final corpus freeze verification, and benchmark hash verification.**
- [ ] **Step 3: Re-run only the preselected viva smoke queries.**
- [ ] **Step 4: Create a final tag such as `viva-ready-2026-10-14` only after all checks pass.**
- [ ] **Step 5: Record the final SHA/tag in thesis/report material and stop feature development for the submitted artifact.**

## Finish Definition

CARE-AnxRAG is finished for the current research milestone when:

1. duplicate CLI registration is removed and full validation remains green;
2. the real evidence corpus passes `freeze-corpus`;
3. development and independent held-out test benchmarks are human reviewed/adjudicated;
4. development and final experiment bundles exist and are immutable;
5. safety benchmark evaluation is complete;
6. real-model latency/integration testing is complete on the viva machine;
7. final results/error-analysis/viva documents are generated from measured artifacts;
8. `./scripts/validate.sh` passes on the final machine; and
9. the exact final code is tagged and frozen.
