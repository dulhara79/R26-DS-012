# CARE-AnxRAG Final Closeout After Conflict Merges

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish CARE-AnxRAG as a reproducible, extractive-only anxiety evidence research prototype with a frozen corpus, reviewed benchmark, formal B0-B5 + CARE_full results, safety results, measured latency, verified central-backend integration, and viva-ready release artifacts.

**Architecture:** Preserve the current extractive-only architecture. Retrieval, CARE evidence scoring, clinical-context sufficiency, contradiction handling, abstention, deterministic source-sentence rendering, and exact-source grounding remain separate. Final research runs must bind one exact code revision, corpus freeze, benchmark, model/config state, and experiment manifest.

**Tech Stack:** Python 3.11/3.12, SQLite/FTS5, ChromaDB, Ollama EmbeddingGemma, Sentence Transformers CrossEncoder + NLI, FastAPI, Typer, pytest, GitHub Actions.

**Spec:** `docs/RESEARCH_PROTOCOL.md`, `docs/BENCHMARK_ANNOTATION.md`, `docs/BENCHMARK_REVIEW_WORKFLOW.md`, `docs/DATA_GOVERNANCE.md`, `docs/CORPUS_COVERAGE.md`, `docs/SAFETY_BENCHMARK.md`, `docs/REPRODUCIBILITY.md`

## Current audited repository state

- `main`: `0b5d104beace5796ae088d9ab06f8e64786e3dd0`
- Latest main CI: green.
- Repository size: 140 tracked files; 44 Python source files; 30 Python test files; 17 Markdown docs.
- Benchmark review package exists:
  - development candidates: 17
  - held-out test candidates: 0
  - status: `awaiting_human_review`
- Final research docs do not exist yet:
  - `docs/FINAL_RESULTS.md`
  - `docs/ERROR_ANALYSIS.md`
  - `docs/VIVA_DEMO.md`
- Current validation report is older than the latest cleanup work:
  - 205 tests
  - 80.06% coverage
  - offline/synthetic validation only
- Current `main` still has duplicate Typer registrations for:
  - `evaluate-safety`
  - `snapshot-experiment`
- Open cleanup/finalization PRs:
  - #30: CLI dedupe + refreshed release validation
  - #31: safe staging review workflow, stacked on #30
  - #32: locked safety-test reviewer/adjudication preflight
- Root-level scratch experiment files still exist:
  - `test_gad*.json`
  - `test_mct.json`
  - `test_social.json`
- No unresolved Git conflict markers were found in tracked files.
- The software now already contains the major research infrastructure:
  - B0-B5/CARE_full ablations
  - latency timings
  - PubMed correction/retraction handling
  - clinical facets
  - outcome/comorbidity query + CARE applicability
  - joint clinical-context sufficiency
  - corpus coverage
  - deterministic extractive grounding
  - corpus freeze gate
  - benchmark review/compiler/split validation
  - experiment bundle
  - paired statistics
  - final experiment preflight
  - safety evaluator
  - reproducibility snapshots

## Global Constraints

- Returned medical claims remain extractive-only.
- Exact-source grounding must not be weakened to answer-stage probabilistic NLI.
- NLI remains allowed for evidence-vs-evidence contradiction analysis.
- Clinical labels, benchmark gold evidence, staging decisions, and reviewer identities must not be fabricated.
- Locked test outcomes must never be used for tuning.
- No source is promoted merely to improve benchmark performance.
- Existing historical PubMed approve/reject plans are hints only unless independently supported by real reviewer provenance.
- All code changes use RED → GREEN → full-suite verification.
- Final reported values must come from immutable experiment artifacts, not manual transcription or guessed values.
- The final viva demo must use the same code/config/corpus family described in the research results.

## Review Focus

1. **Silent merge damage:** code can compile even when incoming conflict resolution duplicates commands or drops newer behavior.
2. **Corpus leakage:** unresolved staging/retraction alerts must block corpus freeze.
3. **Benchmark leakage:** development and held-out test content/review records must remain separated.
4. **Extractive-policy regression:** no truncation, paraphrase, or generated medical text may enter final answers.
5. **Stale evidence/results:** final claims must bind the exact code, corpus, benchmark, and model revisions used in the final run.

---

### Task 1: Repair the post-conflict repository surface

**Files:**
- Modify: `src/care_anxrag/cli.py`
- Modify: `tests/test_cli_contract.py`
- Regenerate: `validation-report.json`
- Regenerate: `SHA256SUMS`
- Modify: `docs/VALIDATION.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: current `main`.
- Produces: unique CLI registration, current validation evidence, and documentation matching actual merged capabilities.

- [ ] **Step 1: Merge/reapply PR #30 onto current main.**
- [ ] **Step 2: Run the command-surface regression test and require every Typer command name to be unique.**
- [ ] **Step 3: Confirm exactly one registration each for `evaluate-safety` and `snapshot-experiment`.**
- [ ] **Step 4: Run `./scripts/validate.sh`.**
- [ ] **Step 5: Require compile, full pytest, offline self-check, package validation, and coverage floor to pass.**
- [ ] **Step 6: Regenerate `validation-report.json` and `SHA256SUMS` from the successful run.**
- [ ] **Step 7: Verify README/VALIDATION terminology remains extractive-only and lists the merged research commands.**
- [ ] **Step 8: Merge only after exact-head GitHub CI passes.**

### Task 2: Add the safe real-staging review path

**Files:**
- Add from PR #31:
  - `src/care_anxrag/staging_review.py`
  - `tests/test_staging_review.py`
  - `docs/STAGING_REVIEW_WORKFLOW.md`
- Modify: `src/care_anxrag/cli.py`

**Interfaces:**
- Produces:
  - `care-anxrag staging-review-package`
  - `care-anxrag staging-review-apply`

- [ ] **Step 1: Merge PR #31 only after Task 1 is on main.**
- [ ] **Step 2: Run its targeted staging-review tests.**
- [ ] **Step 3: Run the full release validation suite.**
- [ ] **Step 4: Verify historical screening plans populate hint columns only, never final decisions.**
- [ ] **Step 5: Verify apply fails closed when current staging differs from the generated review CSV.**
- [ ] **Step 6: Merge after exact-head CI passes.**

### Task 3: Lock down safety-test review discipline

**Files:**
- Modify from PR #32:
  - `src/care_anxrag/safety_evaluation.py`
  - `tests/test_safety_evaluation.py`
  - `docs/SAFETY_BENCHMARK.md`

**Interfaces:**
- Consumes: safety benchmark items.
- Produces: locked-test preflight requiring `adjudicated=true` and at least two distinct reviewer IDs.

- [ ] **Step 1: Merge PR #32 after confirming it does not conflict with Tasks 1-2.**
- [ ] **Step 2: Run safety evaluator tests and full validation.**
- [ ] **Step 3: Confirm development items remain usable for rule development.**
- [ ] **Step 4: Confirm locked test evaluation fails closed without two reviewers + adjudication.**

### Task 4: Perform one final codebase survival audit after all merges

**Files:**
- Modify only when a reproducible defect is proven.
- Test: existing full suite plus targeted contract tests.

**Interfaces:**
- Produces: green, conflict-clean final engineering baseline.

- [ ] **Step 1: Search tracked files for Git conflict markers.**
- [ ] **Step 2: Verify CLI command uniqueness programmatically.**
- [ ] **Step 3: Verify the following commands exist and execute help successfully:**
  - `evaluate`
  - `evaluate-ablation`
  - `experiment-bundle`
  - `freeze-corpus`
  - `experiment-final`
  - `evaluate-safety`
  - `snapshot-experiment`
  - `coverage`
  - `benchmark-scaffold`
  - `benchmark-compile`
  - `benchmark-review-package`
  - `benchmark-validate-splits`
  - `staging-review-package`
  - `staging-review-apply`
- [ ] **Step 4: Run focused regression suites for retrieval, grounding, safety, PubMed correction handling, benchmark review, experiment bundle, corpus freeze, final experiment preflight, and statistics.**
- [ ] **Step 5: Run the complete validation script again.**
- [ ] **Step 6: Treat this commit as the final engineering baseline before real-corpus review.**

### Task 5: Resolve the real local staging queue and evidence alerts

**Files:**
- Human-reviewed local artifact:
  - `artifacts/pubmed-staging-review.csv`
- No source-code change unless a reproducible tooling bug is discovered.

**Interfaces:**
- Consumes: real local SQLite staging rows.
- Produces: zero unresolved staging versions or a documented reason why freeze cannot proceed.

- [ ] **Step 1: Pull the final engineering baseline onto the research workstation.**
- [ ] **Step 2: Run `care-anxrag staging --project-root .` and record the actual current staging count.**
- [ ] **Step 3: If staging is non-zero, generate:**
  `care-anxrag staging-review-package artifacts/pubmed-staging-review.csv --hints pubmed_reject_plan.json --project-root .`
- [ ] **Step 4: Have a real reviewer inspect every staged source and complete `decision`, `decision_reason`, and `reviewer_id`.**
- [ ] **Step 5: Dry-run:**
  `care-anxrag staging-review-apply artifacts/pubmed-staging-review.csv --project-root .`
- [ ] **Step 6: Apply only after dry-run is clean:**
  `care-anxrag staging-review-apply artifacts/pubmed-staging-review.csv --apply --project-root .`
- [ ] **Step 7: Resolve any open evidence/retraction/correction alerts through the existing review/governance process.**
- [ ] **Step 8: Re-run staging and alert checks; both must be clean before corpus freeze.**

### Task 6: Freeze the real evidence corpus

**Files:**
- Generated artifact: `artifacts/corpus-freeze-final.json`

**Interfaces:**
- Consumes: final code baseline + cleaned real local corpus.
- Produces: `ready_to_freeze=true` corpus fingerprint.

- [ ] **Step 1: Run real production-stack health with Chroma, EmbeddingGemma, reranker, and NLI available.**
- [ ] **Step 2: Run `care-anxrag stats --project-root .`.**
- [ ] **Step 3: Run `care-anxrag coverage --project-root .`.**
- [ ] **Step 4: Inspect direct coverage for core study combinations including GAD+CBT, GAD+MCT, social-anxiety+CBT, and panic+CBT/exposure.**
- [ ] **Step 5: Run:**
  `care-anxrag freeze-corpus artifacts/corpus-freeze-final.json --code-revision "$(git rev-parse HEAD)" --project-root .`
- [ ] **Step 6: Require `ready_to_freeze=true`.**
- [ ] **Step 7: If freeze fails, fix only the reported blocker: database integrity, runtime health, staging, evidence alerts, or vector reconciliation.**
- [ ] **Step 8: Do not start final benchmark evaluation before a successful freeze artifact exists.**

### Task 7: Complete the development benchmark review

**Files:**
- Working human-review copies derived from:
  - `data/benchmark/review/development.csv`
- Generated:
  - `data/benchmark/anxiety-development.jsonl`

**Interfaces:**
- Consumes: 17 current development candidates and frozen corpus.
- Produces: human-adjudicated development benchmark.

- [ ] **Step 1: Give the development sheet to two independent qualified reviewers.**
- [ ] **Step 2: Preserve their independent decisions.**
- [ ] **Step 3: Adjudicate disagreements.**
- [ ] **Step 4: Compile the completed review sheet with `benchmark-compile`.**
- [ ] **Step 5: Verify all gold evidence excerpts exist verbatim in the frozen active corpus.**
- [ ] **Step 6: Use development results only for calibration/error analysis.**

### Task 8: Run development experiments and freeze configuration

**Files:**
- Generated immutable development experiment directory.

**Interfaces:**
- Consumes: development benchmark + frozen corpus.
- Produces: B0-B5/CARE_full metrics, paired comparisons, coverage, timings, and snapshot.

- [ ] **Step 1: Run `experiment-bundle` on the development benchmark.**
- [ ] **Step 2: Inspect retrieval metrics, abstention, conflicts, prohibited-evidence intrusion, extractive faithfulness, active-version accuracy, stale-evidence intrusion, and gold-evidence coverage.**
- [ ] **Step 3: Use paired statistics to compare CARE_full against B0-B5.**
- [ ] **Step 4: Classify every important development failure as retrieval defect, clinical-context defect, corpus gap, benchmark-label issue, or latency issue.**
- [ ] **Step 5: Fix only reproducible development defects.**
- [ ] **Step 6: Re-run development experiments after each accepted change.**
- [ ] **Step 7: Freeze thresholds/config/model revisions once development behavior is acceptable.**
- [ ] **Step 8: Do not change them after held-out test inspection.**

### Task 9: Independently author and adjudicate the held-out test benchmark

**Files:**
- Start from:
  - `data/benchmark/review/test.csv`
- Produce locally/restricted:
  - `data/benchmark/anxiety-test.jsonl`

**Interfaces:**
- Produces: real locked test set independent of the 17 development candidates.

- [ ] **Step 1: Independently author held-out test questions across the documented 17 strata.**
- [ ] **Step 2: Keep test questions and review records separate from development tuning.**
- [ ] **Step 3: Have two reviewers independently label and adjudicate the test set.**
- [ ] **Step 4: Compile with `benchmark-compile`.**
- [ ] **Step 5: Run `benchmark-validate-splits development.jsonl test.jsonl` and require no duplicate/leaking questions.**
- [ ] **Step 6: Lock the test set and stop tuning.**

### Task 10: Build and evaluate the safety benchmark

**Files:**
- Human-reviewed local/restricted:
  - `data/benchmark/safety-development.jsonl`
  - `data/benchmark/safety-test.jsonl`

**Interfaces:**
- Produces: crisis recall, urgent recall, normal false-positive rate, confusion matrix, and error list.

- [ ] **Step 1: Create development and locked-test safety cases across direct, indirect, negated, historical, academic, third-person, urgent-symptom, and normal strata.**
- [ ] **Step 2: Use two reviewers + adjudication for the locked test.**
- [ ] **Step 3: Tune deterministic safety rules only on development cases.**
- [ ] **Step 4: Run `evaluate-safety` on locked test only after rules are frozen.**
- [ ] **Step 5: Review every crisis/urgent false negative manually.**

### Task 11: Run the final frozen experiment exactly once

**Files:**
- Generated immutable final run directory:
  - `artifacts/final-experiment-<date>/`

**Interfaces:**
- Consumes: frozen corpus, locked test benchmark, frozen config/code revision.
- Produces: final research results.

- [ ] **Step 1: Verify code revision matches the corpus freeze artifact.**
- [ ] **Step 2: Run:**
  `care-anxrag experiment-final data/benchmark/anxiety-test.jsonl artifacts/final-experiment-<date> --corpus-freeze artifacts/corpus-freeze-final.json --code-revision "$(git rev-parse HEAD)" --project-root .`
- [ ] **Step 3: Do not overwrite the output directory.**
- [ ] **Step 4: Do not tune after viewing final results.**
- [ ] **Step 5: Archive the manifest, ablations, coverage, timings, statistics, and snapshot unchanged.**

### Task 12: Verify real-model latency and central-backend integration

**Files:**
- No code changes unless a reproducible integration defect is found.

**Interfaces:**
- Produces: measured real-stack latency and end-to-end demo evidence.

- [ ] **Step 1: Start CARE with real Chroma/Ollama/CrossEncoder configuration.**
- [ ] **Step 2: Measure representative queries and record median/p95 total latency plus stage timings.**
- [ ] **Step 3: Verify one strong answerable query, one insufficiency case, one out-of-domain case, and one crisis/urgent route.**
- [ ] **Step 4: Verify the central backend evidence endpoint calls CARE successfully.**
- [ ] **Step 5: Record HTTP status and end-to-end latency.**
- [ ] **Step 6: Do not manually edit medical answer text for the demo.**

### Task 13: Produce final research and viva artifacts

**Files:**
- Create: `docs/FINAL_RESULTS.md`
- Create: `docs/ERROR_ANALYSIS.md`
- Create: `docs/VIVA_DEMO.md`
- Update: `README.md`
- Remove/move root scratch result JSONs:
  - `test_gad*.json`
  - `test_mct.json`
  - `test_social.json`

**Interfaces:**
- Consumes: final immutable experiment + safety + latency + integration artifacts.
- Produces: submission/viva-ready evidence trail.

- [ ] **Step 1: Generate result tables directly from the final artifacts.**
- [ ] **Step 2: Report B0-B5 vs CARE_full differences with confidence intervals and paired statistics.**
- [ ] **Step 3: Write error analysis by subtype, treatment, corpus gap, abstention, conflict, stale evidence, and distractor evidence.**
- [ ] **Step 4: State limitations explicitly: prototype-scale corpus, reviewer/sample limits, no clinical-device validation, model/hardware dependence.**
- [ ] **Step 5: Write a short viva demo script with frozen questions and expected route, not expected fabricated wording.**
- [ ] **Step 6: Move/remove scratch JSON result files from repository root so the release tree is clean.**

### Task 14: Final release validation and freeze

**Files:**
- Refresh: `validation-report.json`
- Refresh: `SHA256SUMS`
- Update: `CHANGELOG.md`

**Interfaces:**
- Produces: immutable viva-ready code tag.

- [ ] **Step 1: Run `./scripts/validate.sh` on the actual presentation machine.**
- [ ] **Step 2: Require zero test failures and successful offline acceptance.**
- [ ] **Step 3: Re-run real-stack health and final smoke questions.**
- [ ] **Step 4: Verify final results reference the exact code SHA and corpus freeze SHA/fingerprint.**
- [ ] **Step 5: Refresh validation report/checksums.**
- [ ] **Step 6: Create release tag `viva-ready-2026-10-15` or the actual freeze date.**
- [ ] **Step 7: Stop feature work on the tagged research artifact.**

## Finish Definition

CARE-AnxRAG is finished for the research/viva milestone only when all of the following are true:

1. PR #30, #31, and #32 behavior is integrated into green main.
2. CLI commands are unique and full release validation is current.
3. Real staging and evidence alerts are resolved.
4. Corpus freeze reports `ready_to_freeze=true`.
5. Development benchmark is reviewed/adjudicated and used for tuning only.
6. Held-out test is independently authored, reviewed, adjudicated, and split-validated.
7. Safety locked test is reviewed/adjudicated.
8. Final experiment completes under frozen code/corpus/config.
9. Real model-stack latency and central-backend integration are measured.
10. `FINAL_RESULTS.md`, `ERROR_ANALYSIS.md`, and `VIVA_DEMO.md` exist and cite measured artifacts.
11. Final validation passes on the viva machine.
12. The exact code/results are tagged and frozen.
