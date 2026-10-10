# CARE-AnxRAG Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish CARE-AnxRAG as a defensible research prototype by integrating the remaining feature stack, freezing the evidence/benchmark state, producing reproducible ablation/safety/performance results, closing only evidence-backed gaps, and freezing a viva-ready release.

**Architecture:** Keep the current extractive-only architecture. Retrieval/relevance, CARE evidence scoring, contradiction analysis, sufficiency/abstention, deterministic source-sentence rendering, and exact-source citation verification remain separate. Research evaluation runs against one frozen corpus snapshot and one locked benchmark; no ranking or threshold change is allowed after the locked test is opened.

**Tech Stack:** Python 3.11/3.12, SQLite/FTS5, ChromaDB, Ollama EmbeddingGemma, Sentence Transformers CrossEncoder + NLI, FastAPI, Typer, pytest, GitHub Actions.

**Spec:** `docs/RESEARCH_PROTOCOL.md`, `docs/BENCHMARK_ANNOTATION.md`, `docs/DATA_GOVERNANCE.md`, `docs/CORPUS_COVERAGE.md`

## Global Constraints

- Medical answer text remains extractive-only; no free-form medical generation.
- NLI may compare evidence items for contradiction but must not authorize paraphrased answer text.
- Treatment support must come from source text/title, not treatment topic tags alone.
- Locked test items require adjudication and at least two distinct annotators.
- Clinical gold labels must not be fabricated by the implementation agent.
- Tune thresholds/weights only on development data; freeze before locked-test evaluation.
- Frozen evaluation must use one corpus snapshot, benchmark version, code revision, model revisions, and configuration.
- Existing active evidence provenance/versioning rules remain authoritative.
- All code changes use RED → GREEN → full-suite verification before PR creation.
- No new corpus item is promoted solely to improve a benchmark score; promotion follows the existing review/governance process.

## Review Focus

- Stacked PR integration after `main` changed: preserve all merged #3/#14/#15 behavior and avoid silently dropping stack commits.
- Locked-test leakage: test labels must not influence CARE weights, thresholds, candidate counts, or corpus promotion.
- Corpus/benchmark mismatch: a question with no direct active evidence must be marked insufficient/abstain rather than fixed by a ranking hack.
- Version contamination: superseded/withdrawn evidence must never appear in active experiment results.
- Reproducibility drift: every reported result must bind code, corpus, benchmark, models, config, and dependency state.

---

### Task 1: Integrate the remaining research feature stack onto current `main`

**Files:**
- Modify only conflict-resolved files already touched by PRs #4, #5, #6, #7, #9, #10, #11, #12, #13.
- Test: repository full test suite and offline self-check after each integration unit.

**Interfaces:**
- Consumes: current `main` at/after `3bc3ee713fdae79aaaa5212e036893ff744a92fc`.
- Produces: a single green `main` containing ablations, timings, PubMed correction monitoring, clinical facets, joint sufficiency, safety evaluation, and reproducibility snapshots.

- [ ] **Step 1: Re-read each open PR diff against current `main` and classify it as stacked or independent.**
- [ ] **Step 2: Integrate PR #4 (ablation modes) onto current `main`; resolve conflicts with merged research evaluation without removing version-safety metrics or locked-test validation.**
- [ ] **Step 3: Run `python -m compileall -q src && pytest -q && care-anxrag selfcheck --offline --project-root .`; require zero failures.**
- [ ] **Step 4: Integrate PR #5 (latency instrumentation), rerun the same full gate, and confirm skipped ablation stages report `0.0` timings.**
- [ ] **Step 5: Integrate PR #6 → #7 → #9 → #10 → #11 in dependency order, running the full gate after each merge/integration.**
- [ ] **Step 6: Integrate independent PR #12 (safety benchmark evaluation) and #13 (experiment snapshots), resolving CLI conflicts against current commands including `coverage`.**
- [ ] **Step 7: Verify final `main` has commands `evaluate`, `evaluate-ablation`, `evaluate-safety`, `snapshot-experiment`, `coverage`, `stats`, `retrieve`, and `ask`.**
- [ ] **Step 8: Commit/merge only when exact-head GitHub CI and central sync are green.**

### Task 2: Add a one-command experiment bundle

**Files:**
- Create: `src/care_anxrag/experiment_bundle.py`
- Modify: `src/care_anxrag/cli.py`
- Create: `tests/test_experiment_bundle.py`
- Create: `docs/EXPERIMENT_RUNBOOK.md`

**Interfaces:**
- Consumes: `evaluate_ablation(...)`, safety evaluator, coverage audit, reproducibility snapshot builder, retrieval/answer timings.
- Produces: `run_experiment_bundle(benchmark: Path, output_dir: Path, runtime: Runtime, *, code_revision: str) -> dict[str, Any]` and CLI command `care-anxrag experiment-bundle BENCHMARK OUTPUT_DIR ...`.

- [ ] **Step 1: Write failing tests requiring one command to create an immutable run directory with ablation results, coverage report, timings summary, and reproducibility snapshot.**
- [ ] **Step 2: Run targeted tests and confirm RED because the experiment bundle does not exist.**
- [ ] **Step 3: Implement the minimal bundle writer; do not recompute or invent labels.**
- [ ] **Step 4: Ensure filenames are deterministic: `ablation.json`, `coverage.json`, `timings.json`, `snapshot.json`, `manifest.json`.**
- [ ] **Step 5: Run targeted tests, then full suite and offline self-check.**
- [ ] **Step 6: Document the exact production experiment command and required environment/model state.**
- [ ] **Step 7: Commit and open a focused PR.**

### Task 3: Add publication-ready comparison statistics

**Files:**
- Create: `src/care_anxrag/statistics.py`
- Modify: `src/care_anxrag/experiment_bundle.py`
- Create: `tests/test_statistics.py`
- Modify: `docs/RESEARCH_PROTOCOL.md`

**Interfaces:**
- Consumes: per-item reports from B0–B5 and CARE_full.
- Produces: paired bootstrap 95% confidence intervals for continuous/ranking metrics and exact McNemar counts/test result for paired binary outcomes.

- [ ] **Step 1: Write failing tests for deterministic seeded paired bootstrap intervals using fixed synthetic per-item scores.**
- [ ] **Step 2: Write failing tests for McNemar discordant counts `b` and `c`, including the zero-discordance case.**
- [ ] **Step 3: Run targeted tests and confirm RED.**
- [ ] **Step 4: Implement `paired_bootstrap_difference(..., seed: int = 20260927, samples: int = 10000)` and `mcnemar_exact(...)` without adding heavy dependencies.**
- [ ] **Step 5: Add comparisons of each baseline against `CARE_full` for retrieval metrics and abstention correctness where labels exist.**
- [ ] **Step 6: Export `comparisons.json` and a flat `comparisons.csv` in the experiment bundle.**
- [ ] **Step 7: Run targeted + full verification and commit.**

### Task 4: Freeze and audit the real local evidence corpus

**Files:**
- No source-code change unless the audit reveals a reproducible bug.
- Generated local artifacts: experiment output directory outside tracked clinical content.
- Optional tracked template: `docs/CORPUS_FREEZE_CHECKLIST.md`.

**Interfaces:**
- Consumes: the user's real local SQLite/Chroma evidence state.
- Produces: a frozen corpus fingerprint and a coverage-gap report.

- [ ] **Step 1: On the research workstation, pull the final `main` and run `care-anxrag selfcheck --offline --project-root .`.**
- [ ] **Step 2: Run `care-anxrag stats --project-root .` and save output.**
- [ ] **Step 3: Run `care-anxrag coverage --project-root .` and save output.**
- [ ] **Step 4: Run focused coverage checks for the benchmark's explicit subtype × treatment pairs, especially GAD+CBT, GAD+MCT, social-anxiety+CBT, panic+CBT/exposure.**
- [ ] **Step 5: Review staging/retraction/correction alerts; resolve or explicitly document unresolved items before freeze.**
- [ ] **Step 6: Run vector reconciliation and database integrity checks.**
- [ ] **Step 7: Generate an experiment snapshot bound to the current code revision and record active version/content hashes.**
- [ ] **Step 8: Declare this snapshot the development corpus freeze; do not promote evidence during evaluation except through a documented new experimental phase.**

### Task 5: Build and adjudicate the benchmark without fabricating clinical gold

**Files:**
- Create locally: `data/benchmark/anxiety-development.jsonl`
- Create locally after independent review: `data/benchmark/anxiety-test.jsonl`
- Optional generated reviewer sheet: `artifacts/benchmark_annotation.csv`

**Interfaces:**
- Consumes: active corpus IDs/excerpts and the benchmark schema in `evaluation.py`.
- Produces: development and locked-test JSONL with valid source IDs, gold excerpts, abstention/conflict labels, two annotator IDs, and adjudication status.

- [ ] **Step 1: Generate a question/annotation scaffold across the protocol strata; do not auto-fill clinical relevance or correctness labels.**
- [ ] **Step 2: Link candidate evidence IDs/excerpts from the frozen corpus to make human review fast while preserving reviewer control.**
- [ ] **Step 3: Have two qualified reviewers annotate the development set independently and adjudicate disagreements.**
- [ ] **Step 4: Tune only on development data; record every changed threshold/weight/config value and rationale.**
- [ ] **Step 5: Freeze those values.**
- [ ] **Step 6: Have the two reviewers annotate/adjudicate the locked test set without exposing test outcomes to tuning.**
- [ ] **Step 7: Validate JSONL with `care-anxrag evaluate`; locked-test preflight must reject any item lacking two distinct annotators or adjudication.**

### Task 6: Run the formal ablation experiment

**Files:**
- Generated: `artifacts/experiments/<run-id>/...`
- No code modifications during the locked test.

**Interfaces:**
- Consumes: frozen corpus, frozen config, adjudicated test benchmark.
- Produces: B0, B1, B2, B3, B4, B5, CARE_full reports plus paired comparisons and reproducibility metadata.

- [ ] **Step 1: Warm required models and verify production stack health.**
- [ ] **Step 2: Run the experiment bundle on the development benchmark first; confirm all modes complete and artifacts are internally consistent.**
- [ ] **Step 3: Run the locked test exactly once under the frozen configuration.**
- [ ] **Step 4: Verify active-version accuracy, stale-evidence intrusion, prohibited-evidence intrusion, extractive faithfulness, citation validity, abstention accuracy, MRR, Recall@5, Precision@5, and nDCG@5.**
- [ ] **Step 5: Generate paired baseline-vs-CARE comparisons with 95% CIs and McNemar results.**
- [ ] **Step 6: Archive the complete run manifest and do not overwrite it.**

### Task 7: Run safety-router and adversarial validation

**Files:**
- Local benchmark: `data/benchmark/safety-test.jsonl`
- Generated results in the same experiment bundle or a sibling safety run.

**Interfaces:**
- Consumes: merged PR #12 safety evaluator and adjudicated safety labels.
- Produces: crisis recall, urgent recall, normal false-positive rate, confusion matrix, per-stratum errors.

- [ ] **Step 1: Build/adjudicate crisis, urgent, normal, ambiguous, and lexical-trap safety cases with two reviewers.**
- [ ] **Step 2: Run `care-anxrag evaluate-safety ...`.**
- [ ] **Step 3: Inspect every false negative in crisis/urgent strata and every clinically disruptive normal false positive.**
- [ ] **Step 4: If a deterministic rule change is needed, make it on development cases only, rerun tests, then freeze before final safety test.**
- [ ] **Step 5: Save the final safety report and error analysis.**

### Task 8: Measure and optimize latency without sacrificing evidence quality

**Files:**
- Generated: timing summaries and before/after experiment bundles.
- Code changes only if profiling identifies a verified bottleneck.

**Interfaces:**
- Consumes: PR #5 stage timings.
- Produces: median/p95 total latency and stage-level timing distribution for the production stack.

- [ ] **Step 1: Run representative development questions on the actual viva hardware and collect timings.**
- [ ] **Step 2: Identify the dominant stage from measurements; do not guess.**
- [ ] **Step 3: If optimization is required, change one variable at a time (candidate counts, rerank depth, NLI pair count, model warm state) and rerun the same development benchmark.**
- [ ] **Step 4: Reject any speed change that materially degrades frozen development retrieval/safety metrics.**
- [ ] **Step 5: Freeze the final performance configuration and report median/p95 latency with hardware/model details.**

### Task 9: Close only evidence-backed corpus gaps

**Files:**
- Existing ingestion/source-review workflow only.
- No benchmark-specific hardcoded ranking rules.

**Interfaces:**
- Consumes: coverage audit + development error analysis.
- Produces: reviewed evidence additions or explicit documented limitations.

- [ ] **Step 1: Classify each failure as retrieval bug, sufficiency bug, benchmark-label issue, or genuine corpus gap.**
- [ ] **Step 2: For genuine gaps, locate only authorized/high-quality evidence through the existing source connectors and stage it.**
- [ ] **Step 3: Human-review the staged evidence under current governance rules.**
- [ ] **Step 4: Promote only approved evidence, regenerate embeddings/indexes, rerun coverage, and create a new corpus snapshot.**
- [ ] **Step 5: If the corpus changes, invalidate previous final results and repeat the freeze/evaluation cycle; never patch the locked test outcome directly.**

### Task 10: Validate API/central-backend integration and viva demo

**Files:**
- CARE-AnxRAG plus the existing central backend integration configuration.
- No API contract changes unless a failing integration test proves one is required.

**Interfaces:**
- Consumes: final CARE service on port/configuration used by the central backend.
- Produces: verified health, normal evidence answer, abstention, safety route, and central-backend call path.

- [ ] **Step 1: Start CARE service and verify `/health`.**
- [ ] **Step 2: Test one strong-coverage query: CBT for social anxiety disorder.**
- [ ] **Step 3: Test one explicit corpus-gap/sufficiency case and verify abstention rather than fabricated synthesis.**
- [ ] **Step 4: Test an out-of-domain question and verify abstention.**
- [ ] **Step 5: Test crisis/urgent routing separately from normal RAG.**
- [ ] **Step 6: Test the central backend evidence endpoint against CARE and record end-to-end latency/status.**
- [ ] **Step 7: Save three concise viva examples with citations/confidence/abstention reason and no hidden manual edits.**

### Task 11: Produce the final research artifacts

**Files:**
- Create: `docs/FINAL_RESULTS.md`
- Create: `docs/ERROR_ANALYSIS.md`
- Create: `docs/VIVA_DEMO.md`
- Update: `README.md` only with measured/frozen results.

**Interfaces:**
- Consumes: immutable experiment, safety, coverage, and performance artifacts.
- Produces: claims traceable to measured results.

- [ ] **Step 1: Build a results table for B0–B5 and CARE_full without inventing missing values.**
- [ ] **Step 2: Report effect sizes/CIs and McNemar results for defined primary comparisons.**
- [ ] **Step 3: Write error analysis by subtype, treatment, abstention, contradiction, stale evidence, and distractor evidence.**
- [ ] **Step 4: State limitations explicitly: prototype-scale curated corpus, expert annotation limits, no clinical-device validation, model/hardware dependence.**
- [ ] **Step 5: Document the contribution as evidence-aware retrieval/selection, contradiction handling, sufficiency, calibrated abstention, provenance/versioning, and deterministic extractive grounding—not as novelty in Chroma/BM25/RRF/CrossEncoder.**
- [ ] **Step 6: Create a five-minute demo script and a panel-defense evidence trail from the frozen run artifacts.**

### Task 12: Freeze the viva release

**Files:**
- No feature changes after freeze except release-blocking defects.

**Interfaces:**
- Consumes: green final `main`, frozen artifacts, final docs.
- Produces: immutable viva-ready Git tag/release candidate.

- [ ] **Step 1: Run `./scripts/validate.sh` on the actual presentation machine.**
- [ ] **Step 2: Run production health/self-check with the exact embedding/reranker/NLI models used for the demo.**
- [ ] **Step 3: Verify database integrity, vector reconciliation, corpus snapshot, benchmark hashes, and experiment manifest.**
- [ ] **Step 4: Re-run only the preselected viva smoke queries; do not tune after this point.**
- [ ] **Step 5: Create the final tag `viva-ready-2026-09-27` (or the actual freeze date if later) only after all checks pass.**
- [ ] **Step 6: Record the tag SHA in the report and stop feature development for the submitted/viva artifact.**

## Finish Definition

CARE-AnxRAG is considered finished for the current research milestone only when:

1. all intended PR #4–#13 capabilities are integrated into green `main`;
2. the real corpus is frozen and fingerprinted;
3. development and locked-test benchmarks are independently annotated/adjudicated;
4. B0–B5 and CARE_full have immutable result artifacts;
5. safety-router evaluation is complete;
6. latency is measured on the actual demo hardware;
7. all reported claims are sourced from those artifacts;
8. end-to-end CARE + central backend demo paths are verified;
9. the final validation suite passes on the viva machine; and
10. a release tag pins the exact code used for results and demonstration.
