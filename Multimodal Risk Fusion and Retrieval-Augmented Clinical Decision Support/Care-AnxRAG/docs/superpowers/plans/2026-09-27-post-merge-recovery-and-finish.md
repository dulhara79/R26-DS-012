# CARE-AnxRAG Post-Merge Recovery and Finish Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the two post-merge regressions, verify that every merged research capability is still wired correctly, then finish CARE-AnxRAG with reproducible experiments, reviewed benchmarks, measured performance, final research artifacts, and a frozen viva-ready release.

**Architecture:** Preserve the current extractive-only medical-answer architecture. Retrieval, CARE scoring, contradiction handling, clinical-context sufficiency, deterministic evidence presentation, and exact-source grounding remain separate. Final research runs use one frozen code revision, corpus snapshot, benchmark, configuration, and model set.

**Tech Stack:** Python 3.11/3.12, SQLite/FTS5, ChromaDB, Ollama EmbeddingGemma, Sentence Transformers CrossEncoder + NLI, FastAPI, Typer, pytest, GitHub Actions.

**Spec:** `docs/RESEARCH_PROTOCOL.md`, `docs/BENCHMARK_ANNOTATION.md`, `docs/DATA_GOVERNANCE.md`, `docs/CORPUS_COVERAGE.md`, `docs/SAFETY_BENCHMARK.md`, `docs/REPRODUCIBILITY.md`

## Global Constraints

- Medical claims returned to users must remain extractive-only.
- Exact-source grounding must not be weakened to probabilistic answer-stage NLI.
- NLI remains allowed for evidence-vs-evidence contradiction analysis.
- Treatment support must come from evidence text/title, not topic tags alone.
- Locked test items require at least two distinct annotators and adjudication.
- Clinical gold labels are never generated or inferred by the implementation agent.
- Development data may be used for tuning; locked-test outcomes must not influence configuration.
- No evidence item is promoted solely to improve benchmark performance.
- Every code change uses RED → GREEN → full-suite validation before PR creation.
- Final reported numbers must bind code revision, corpus snapshot, benchmark hash, model/config state, and dependency state.

## Review Focus

- **Conflict-loss in CLI wiring:** merged feature modules can exist while commands disappear; a command-surface contract test must prevent recurrence.
- **Extractive policy regression:** answer formatting must never modify or paraphrase medical source wording.
- **Benchmark leakage:** locked-test labels must never affect thresholds, candidate counts, weights, or corpus promotion.
- **Version contamination:** superseded/withdrawn evidence must never count as active final evidence.
- **Stale validation claims:** README, validation report, and viva material must reflect the actual final code and measured runs only.

---

### Task 1: Repair post-merge CLI regressions and protect the command surface

**Files:**
- Modify: `src/care_anxrag/cli.py`
- Test: `tests/test_safety_evaluation.py`
- Test: `tests/test_reproducibility.py`
- Create: `tests/test_cli_contract.py`

**Interfaces:**
- Consumes: `run_safety_evaluation(router, items)`, `load_safety_benchmark(path)`, `build_experiment_snapshot(runtime, *, code_revision, benchmark_path)`.
- Produces: CLI commands `evaluate-safety` and `snapshot-experiment`, plus a stable command-surface contract.

- [ ] **Step 1: Add a failing command-surface test asserting the registered commands include:**
  `init`, `sync`, `ask`, `retrieve`, `stats`, `coverage`, `sources`, `staging`, `approve`, `reject`, `withdraw`, `reconcile`, `evaluate`, `evaluate-ablation`, `evaluate-safety`, `snapshot-experiment`, `selfcheck`, `serve`, `scheduler`.
- [ ] **Step 2: Run `pytest tests/test_cli_contract.py tests/test_safety_evaluation.py::test_evaluate_safety_cli_outputs_report tests/test_reproducibility.py::test_snapshot_experiment_cli_writes_json -v`; confirm RED.**
- [ ] **Step 3: Implement `evaluate_safety_command(benchmark: Path) -> None` in `cli.py` using `SafetyRouter`, `load_safety_benchmark`, and `run_safety_evaluation`.**
- [ ] **Step 4: Implement `snapshot_experiment_command(output: Path, code_revision: str | None, benchmark: Path | None, project_root: Path | None) -> None` in `cli.py`; write deterministic JSON to the requested output path.**
- [ ] **Step 5: Rerun the targeted tests; require PASS.**
- [ ] **Step 6: Run `python -m compileall -q src && pytest -q && care-anxrag selfcheck --offline --project-root .`; require zero failures.**
- [ ] **Step 7: Push the repair branch and require exact-head GitHub CI + central sync to pass before merge.**

### Task 2: Perform a post-merge feature-survival audit and regenerate validation evidence

**Files:**
- Modify only if a missing wiring defect is found.
- Regenerate: `validation-report.json`
- Modify: `docs/VALIDATION.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: merged modules/tests for ablation, timings, PubMed integrity events, clinical facets, safety evaluation, reproducibility, coverage, exact grounding.
- Produces: a green post-merge validation report and documentation that matches the actual code.

- [ ] **Step 1: Run `./scripts/validate.sh` from a clean checkout after Task 1.**
- [ ] **Step 2: If any failure appears, diagnose root cause before changing code; do not patch expected outputs to match broken behavior.**
- [ ] **Step 3: Verify the following feature contracts through existing tests/commands: ablation modes, timings, PubMed CommentsCorrections relationships, evidence alerts, clinical outcome/comorbidity facets, joint-context sufficiency, safety evaluation, experiment snapshot, corpus coverage, extractive grounding.**
- [ ] **Step 4: Regenerate `validation-report.json`; it must reference the current code and current test count, not the August 2026 report.**
- [ ] **Step 5: Update `docs/VALIDATION.md` terminology from the legacy “rule-based generator” wording to the deterministic extractive renderer where appropriate.**
- [ ] **Step 6: Update README evaluation metrics/commands to include the merged ablation, safety, coverage, version-safety, extractive-faithfulness, and reproducibility capabilities.**
- [ ] **Step 7: Commit only after the full validation script succeeds.**

### Task 3: Add one-command research experiment bundling

**Files:**
- Create: `src/care_anxrag/experiment_bundle.py`
- Modify: `src/care_anxrag/cli.py`
- Create: `tests/test_experiment_bundle.py`
- Create: `docs/EXPERIMENT_RUNBOOK.md`

**Interfaces:**
- Consumes: `evaluate_ablation(...)`, `audit_corpus_coverage(...)`, `build_experiment_snapshot(...)`, retrieval/answer `timings_ms`.
- Produces: `run_experiment_bundle(runtime, benchmark_path: Path, output_dir: Path, *, code_revision: str) -> dict[str, Any]` and CLI `care-anxrag experiment-bundle BENCHMARK OUTPUT_DIR ...`.

- [ ] **Step 1: Write a failing test requiring one run to create `ablation.json`, `coverage.json`, `timings.json`, `snapshot.json`, and `manifest.json`.**
- [ ] **Step 2: Confirm RED because the bundle API/CLI does not exist.**
- [ ] **Step 3: Implement the minimal bundle writer using existing evaluators; do not duplicate their scoring logic.**
- [ ] **Step 4: Make the output directory immutable-by-default: refuse to overwrite a non-empty run directory unless an explicit future override is deliberately added.**
- [ ] **Step 5: Add CLI `experiment-bundle` and document the exact command.**
- [ ] **Step 6: Run targeted tests, then full suite/self-check, then open a focused PR.**

### Task 4: Add publication-ready paired statistical comparisons

**Files:**
- Create: `src/care_anxrag/statistics.py`
- Modify: `src/care_anxrag/experiment_bundle.py`
- Create: `tests/test_statistics.py`
- Modify: `docs/RESEARCH_PROTOCOL.md`

**Interfaces:**
- Consumes: per-item B0–B5/CARE_full evaluation rows.
- Produces: paired bootstrap confidence intervals and exact McNemar comparison outputs.

- [ ] **Step 1: Write deterministic failing tests for `paired_bootstrap_difference(left, right, *, samples=10000, seed=20260927)`.**
- [ ] **Step 2: Write failing tests for `mcnemar_exact(left_correct, right_correct)`, including zero-discordance behavior.**
- [ ] **Step 3: Implement both functions without adding a heavy statistics dependency.**
- [ ] **Step 4: Compare each baseline against `CARE_full` for applicable retrieval/evidence metrics and abstention correctness.**
- [ ] **Step 5: Export `comparisons.json` and flat `comparisons.csv` inside the experiment bundle.**
- [ ] **Step 6: Update the protocol to distinguish prespecified CARE weights from development-calibrated thresholds unless weights are actually tuned.**
- [ ] **Step 7: Run full validation and open a focused PR.**

### Task 5: Freeze and audit the real local corpus

**Files:**
- No source-code changes unless a reproducible defect is discovered.
- Generated local research artifacts only.

**Interfaces:**
- Consumes: the user's real SQLite/Chroma corpus.
- Produces: frozen corpus coverage, integrity state, and reproducibility fingerprint.

- [ ] **Step 1: Pull the final green `main` on the research workstation.**
- [ ] **Step 2: Run `care-anxrag selfcheck --offline --project-root .` and production health checks with the configured real models.**
- [ ] **Step 3: Run `care-anxrag stats --project-root .` and save the JSON.**
- [ ] **Step 4: Run `care-anxrag coverage --project-root .` and save the JSON.**
- [ ] **Step 5: Run focused coverage checks for benchmark-relevant combinations, including GAD+CBT, GAD+MCT, social-anxiety+CBT, and panic+CBT/exposure.**
- [ ] **Step 6: Review staging plus evidence alerts/correction events; resolve or document every unresolved item.**
- [ ] **Step 7: Run reconciliation/database integrity checks.**
- [ ] **Step 8: Generate `snapshot-experiment` against the intended benchmark scaffold and freeze this as the development corpus snapshot.**

### Task 6: Build development and locked-test benchmarks with human adjudication

**Files:**
- Create locally: `data/benchmark/anxiety-development.jsonl`
- Create locally: `data/benchmark/anxiety-test.jsonl`
- Create locally: `data/benchmark/safety-test.jsonl`

**Interfaces:**
- Consumes: frozen active evidence IDs/excerpts and existing benchmark schemas.
- Produces: development + locked-test question sets with valid clinical/evidence labels.

- [ ] **Step 1: Create a balanced question scaffold across protocol strata; leave clinical relevance/gold labels for human review.**
- [ ] **Step 2: Attach candidate retrieved source IDs/excerpts to accelerate reviewer annotation without treating them as gold.**
- [ ] **Step 3: Have two qualified reviewers independently annotate development items and adjudicate disagreements.**
- [ ] **Step 4: Calibrate thresholds only on development data; keep any prespecified CARE weights unchanged unless the protocol explicitly records a weight-tuning experiment.**
- [ ] **Step 5: Freeze the configuration.**
- [ ] **Step 6: Have two reviewers annotate/adjudicate the locked test set without using results for tuning.**
- [ ] **Step 7: Validate benchmark files through `care-anxrag evaluate`; locked-test preflight must reject missing reviewer/adjudication metadata.**

### Task 7: Run formal ablation, safety, coverage, and latency experiments

**Files:**
- Generated: `artifacts/experiments/<run-id>/...`
- No code changes during locked-test execution.

**Interfaces:**
- Consumes: frozen code, corpus, config, benchmark, and models.
- Produces: immutable research results.

- [ ] **Step 1: Run the experiment bundle on development data and verify all B0–B5 + CARE_full modes complete.**
- [ ] **Step 2: Run the safety evaluator on the reviewed development safety set; fix rules only from development errors.**
- [ ] **Step 3: Measure latency on the actual viva hardware using stage timings; record median and p95.**
- [ ] **Step 4: Freeze config/model state.**
- [ ] **Step 5: Run the locked test once and save the immutable experiment directory.**
- [ ] **Step 6: Verify key outputs: Recall@5, Precision@5, MRR, nDCG@5, abstention accuracy, conflict accuracy, citation validity, extractive faithfulness, gold-evidence coverage, prohibited-evidence intrusion, active-version accuracy, stale-evidence intrusion, crisis recall, urgent recall, normal false-positive rate, and paired comparisons.**

### Task 8: Close only genuine development-set gaps

**Files:**
- Existing source/ingestion/review workflow only unless a reproducible code defect is identified.

**Interfaces:**
- Consumes: development error analysis and coverage audit.
- Produces: reviewed evidence additions or explicit limitations.

- [ ] **Step 1: Classify each development failure as retrieval defect, sufficiency defect, benchmark-label issue, or genuine corpus gap.**
- [ ] **Step 2: Fix code only when a deterministic failing test proves a defect.**
- [ ] **Step 3: For corpus gaps, stage authorized/high-quality evidence and route it through the existing human review process.**
- [ ] **Step 4: If the corpus changes, create a new corpus snapshot and rerun development evaluation before touching the locked test.**
- [ ] **Step 5: Stop tuning before locked-test execution.**

### Task 9: Validate end-to-end API/backend/viva behavior

**Files:**
- No contract changes unless a failing integration test proves they are required.
- Create: `docs/VIVA_DEMO.md`

**Interfaces:**
- Consumes: final CARE service and central backend configuration.
- Produces: reproducible smoke/demo cases.

- [ ] **Step 1: Verify `/health` on the final CARE service.**
- [ ] **Step 2: Verify one strong-coverage answer, one evidence-comparison case, one insufficiency abstention, one OOD abstention, and one crisis route.**
- [ ] **Step 3: Verify the central backend evidence endpoint against CARE and record end-to-end latency/status.**
- [ ] **Step 4: Save the exact demo questions, expected route, evidence IDs/citations, and fallback behavior in `docs/VIVA_DEMO.md`.**
- [ ] **Step 5: Do not manually edit medical answer text for the demo.**

### Task 10: Produce final results/docs and freeze the release

**Files:**
- Create: `docs/FINAL_RESULTS.md`
- Create: `docs/ERROR_ANALYSIS.md`
- Update: `README.md`
- Regenerate: `validation-report.json`

**Interfaces:**
- Consumes: immutable experiment/safety/performance artifacts.
- Produces: final research claims and viva-ready release.

- [ ] **Step 1: Build the B0–B5 + CARE_full results table from saved artifacts only.**
- [ ] **Step 2: Report paired differences, confidence intervals, and McNemar results where applicable.**
- [ ] **Step 3: Write error analysis by subtype, treatment, context sufficiency, conflict, stale evidence, distractors, and safety route.**
- [ ] **Step 4: State limitations explicitly: curated prototype corpus, reviewer sample size, no clinical-device validation, model/hardware dependence.**
- [ ] **Step 5: Update README with only measured/frozen final capabilities and results.**
- [ ] **Step 6: Run `./scripts/validate.sh` on the actual viva machine and regenerate `validation-report.json`.**
- [ ] **Step 7: Re-run final smoke queries without tuning.**
- [ ] **Step 8: Create the viva release tag only after all checks pass; record the tag SHA in `FINAL_RESULTS.md`.**

## Finish Definition

CARE-AnxRAG is finished for the current research milestone when:

1. `main` CI is green after post-merge recovery;
2. the current validation report is regenerated from the final code;
3. experiment bundle + paired statistics are implemented and tested;
4. the real corpus is frozen and fingerprinted;
5. development and locked-test benchmarks are independently reviewed/adjudicated;
6. B0–B5 and CARE_full results are immutable and reproducible;
7. formal safety and latency results exist;
8. central-backend and viva demo paths are verified;
9. final research/error-analysis/demo documents are generated from saved artifacts; and
10. a release tag pins the exact code used for reported results and demonstration.
