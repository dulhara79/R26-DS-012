# ClinAnx P5B Timeline & Clinical Notes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add honest clinician Timeline and Clinical Notes deep views without creating any second assessment authority or inventing backend contracts.

**Architecture:** `PatientOverviewController` remains the owner of current `AssessmentSummary`. P5B adds an independent timeline history read model/repository/controller over the verified clinician timeline endpoint, plus a dedicated clinical-notes controller that preserves existing local `ClinicalNote` records and submits through `POST /v1/clinical-notes`. A successful note submission invokes Patient Overview's canonical refresh; neither P5B screen computes fusion/risk locally.

**Tech Stack:** Flutter/Dart, Provider-era project patterns, existing `ApiClient`, `RecordStore`, `CentralBackendGateway`, flutter_test, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-16-clinanx-p5b-timeline-clinical-notes-design.md`

## Global Constraints

- Base commit is `1be42a9c2260601a449b06ff1dca777a3b00a24f`.
- Work only on `integration/clinanx-p5b-timeline-clinical-notes`.
- `AssessmentSummary` remains authoritative for current multimodal assessment.
- Never call TC-WPN `/predict` from P5B.
- Never derive current risk/tier/fusion/contribution locally.
- Never fabricate missing Timeline values or interpolate gaps.
- Do not invent a clinical-note history GET endpoint.
- Use canonical `subjectId` for backend calls and a separate local record key for `RecordStore`.
- C3 must be presented as `Clinical NLP / TC-WPN signal`, never overall patient risk.
- Successful note submission refreshes Patient Overview's canonical assessment.
- Production behavior changes require a failing test first.
- Do not merge the P5B PR without explicit user approval.

---

### Task 1: Timeline contract and repository boundary

**Files:**
- Create: `lib/domain/contracts/timeline_entry.dart`
- Create: `lib/domain/repositories/timeline_repository.dart`
- Create: `test/contracts/timeline_entry_test.dart`
- Create: `test/repositories/p5b_repositories_test.dart`

**Interfaces:**
- Produces `TimelineEntry.fromJson(Map<String,dynamic>)`.
- Produces `TimelineRepository.history(String subjectId, {int limit = 200})`.

- [ ] **Step 1: Write failing contract tests**

Tests must prove exact parsing of `composite`, `tier`, `band`, `assessment_status`, `missing_modalities`, `computed_at`, and `trigger`, plus null preservation for missing composite/time.

- [ ] **Step 2: Run CI and verify RED**

Expected: focused P5B tests fail because `TimelineEntry` and `TimelineRepository` do not exist.

- [ ] **Step 3: Add minimal domain contract/repository interface**

`TimelineEntry` must not derive tier/band from composite and must use nullable values for missing fields.

- [ ] **Step 4: Verify GREEN for contract tests**

Expected: parsing tests pass with missing values preserved.

- [ ] **Step 5: Commit**

Commit message: `feat: add P5B timeline history contract`

---

### Task 2: Verified Central Backend timeline adapter

**Files:**
- Create: `lib/data/repositories/p5b_repositories.dart`
- Extend: `test/repositories/p5b_repositories_test.dart`

**Interfaces:**
- Consumes `ApiClient` and `TimelineRepository`.
- Produces `CentralBackendTimelineRepository`.
- Calls only `GET /v1/doctor/patients/{subjectId}/timeline?limit=N`.
- Parses only the response `trend` array.

- [ ] **Step 1: Write failing repository tests**

Use an injected HTTP client/`ApiClient` to assert the exact route, query parameter, Authorization behavior already used by the verified Central Backend compatibility path, and exact trend parsing.

- [ ] **Step 2: Verify RED**

Expected: tests fail because production adapter is absent.

- [ ] **Step 3: Implement minimal adapter**

Return an empty list when `trend` is absent/empty. Map 404 to empty history only if the existing API failure vocabulary reports not-found; rethrow other failures.

- [ ] **Step 4: Verify GREEN**

Expected: exact-route and parser tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: read verified clinician timeline history`

---

### Task 3: Timeline controller and 24h/7d filtering

**Files:**
- Create: `lib/state/timeline_controller.dart`
- Create: `test/timeline_controller_test.dart`

**Interfaces:**
- Consumes `TimelineRepository` and canonical `subjectId`.
- Produces loading/data/empty/offline/error state plus a `TimelineWindow` enum (`hours24`, `days7`).
- Filtering uses `DateTime now` injected for deterministic tests.

- [ ] **Step 1: Write failing controller tests**

Tests must prove:
- load requests canonical subject id;
- 24h includes only returned rows with timestamps inside the last 24h;
- 7d includes only returned rows inside seven days;
- missing timestamp is retained as unknown/unwindowed rather than rewritten;
- no synthetic points are created;
- offline/error states contain no fabricated data.

- [ ] **Step 2: Verify RED**

Expected: controller tests fail because controller/window types do not exist.

- [ ] **Step 3: Implement minimal controller**

Use immutable returned rows and explicit state. Do not derive risk values.

- [ ] **Step 4: Verify GREEN**

Expected: controller tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: add P5B timeline controller`

---

### Task 4: Timeline clinician UI

**Files:**
- Create: `lib/features/patients/timeline_screen.dart`
- Create: `test/features/patients/timeline_screen_test.dart`

**Interfaces:**
- Consumes `TimelineController` or production `subjectId`.
- Shows `24 hours` and `7 days` selectors.

- [ ] **Step 1: Write failing widget tests**

Tests must prove:
- exact server composite/tier/band/status/timestamp/trigger render without local derivation;
- null composite renders `—`, not `0`/`Low`;
- missing time renders `Time not reported`;
- empty window says no assessment records were returned;
- explanatory copy states gaps are not low-risk values;
- no forecast-history or attention-event marker text appears without backend data.

- [ ] **Step 2: Verify RED**

Expected: widget tests fail because screen is absent.

- [ ] **Step 3: Implement minimal list-based Timeline UI**

Do not draw connecting line charts because current history is not guaranteed regularly sampled.

- [ ] **Step 4: Verify GREEN**

Expected: Timeline widget tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: add clinician timeline deep view`

---

### Task 5: Clinical Notes repository/controller boundary

**Files:**
- Create: `lib/domain/repositories/clinical_notes_repository.dart`
- Extend: `lib/data/repositories/p5b_repositories.dart`
- Create: `lib/state/clinical_notes_controller.dart`
- Create: `test/clinical_notes_controller_test.dart`
- Extend: `test/repositories/p5b_repositories_test.dart`

**Interfaces:**
- `ClinicalNotesRepository.loadLocalNotes(localRecordId)`.
- `ClinicalNotesRepository.saveLocalNotes(localRecordId, notes)`.
- `ClinicalNotesRepository.effectiveSupportSet(localRecordId)`.
- `ClinicalNotesRepository.submitNote(...) -> ClinicalNoteIngestResult`.
- `ClinicalNotesController` receives `subjectId`, `localRecordId`, repository, clinician id, and `Future<void> Function() refreshCanonicalAssessment`.

- [ ] **Step 1: Write failing controller/repository tests**

Tests must prove:
- local storage uses `localRecordId`, while network submit uses canonical `subjectId`;
- saving a draft never requires network;
- editing a draft updates it rather than duplicating it;
- failed submit preserves note text and marks analysis failed;
- backend result without component detail does not fabricate a C3 score;
- successful submit stores the returned C3 result and invokes canonical refresh exactly once;
- direct TC-WPN prediction is never called.

- [ ] **Step 2: Verify RED**

Expected: tests fail because P5B notes repository/controller do not exist.

- [ ] **Step 3: Implement minimal repository/controller**

Production repository may reuse `RecordStore` and `CentralBackendGateway.submitNote`, but the controller must not depend on legacy `ChartController` or `FusionResult`.

- [ ] **Step 4: Verify GREEN**

Expected: note lifecycle and refresh tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: add P5B clinical notes workflow`

---

### Task 6: Clinical Notes clinician UI

**Files:**
- Create: `lib/features/patients/clinical_notes_screen.dart`
- Create: `test/features/patients/clinical_notes_screen_test.dart`

**Interfaces:**
- Consumes `ClinicalNotesController` or production identities/callback.

- [ ] **Step 1: Write failing widget tests**

Tests must prove:
- page explicitly labels displayed history as device-local;
- clinician can create/save a draft;
- failed analysis retains visible note content and retry action;
- analysed C3 is labelled `Clinical NLP / TC-WPN signal`;
- `TC-WPN Risk`, `Overall risk`, fixed `largest weight` wording, and locally derived risk bands are absent;
- missing C3 result renders `Not reported`/no-result state rather than zero.

- [ ] **Step 2: Verify RED**

Expected: widget tests fail because screen is absent.

- [ ] **Step 3: Implement minimal notes UI**

Use existing design-system components where practical. Keep note creation/editing focused; no support-set management is added in P5B.

- [ ] **Step 4: Verify GREEN**

Expected: widget tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: add clinician clinical-notes deep view`

---

### Task 7: Patient Overview navigation and identity plumbing

**Files:**
- Modify: `lib/features/patients/patient_overview_screen.dart`
- Modify: `lib/features/patients/patients_screen.dart`
- Extend: `test/features/patients/patient_overview_screen_test.dart`
- Create/extend relevant Patients navigation test if required.

**Interfaces:**
- Patient Overview gains `localRecordId` separately from canonical `subjectId`.
- Adds Timeline and Clinical Notes navigation callbacks for injection tests.
- Production Clinical Notes navigation receives canonical subject id + local record id + canonical refresh callback.

- [ ] **Step 1: Write failing navigation tests**

Tests must prove:
- Timeline and Clinical Notes links appear only after a loaded assessment context is available;
- Timeline receives canonical subject id;
- Clinical Notes receives both identities without conflating them;
- successful note refresh callback calls `PatientOverviewController.load(showLoading: false)`;
- opening P5B views does not change the currently rendered `AssessmentSummary` itself.

- [ ] **Step 2: Verify RED**

Expected: tests fail because P5B navigation does not exist.

- [ ] **Step 3: Add minimal production navigation**

`PatientsScreen` passes `patient.mrn` as `localRecordId` while continuing to pass resolved canonical `subjectId` for backend authority.

- [ ] **Step 4: Verify GREEN**

Expected: navigation tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: link P5B views from patient overview`

---

### Task 8: Authority regression checks and CI coverage

**Files:**
- Create: `test/p5b_timeline_clinical_notes_authority_test.dart`
- Modify: `.github/workflows/p1_p2_dashboard_ci.yml`

**Interfaces:**
- Static/structural test scans P5B implementation surface.

- [ ] **Step 1: Write failing authority test**

Assert P5B production files contain no direct `/predict`, no guessed clinical-note history route, no `AlertBandX.fromScore`, no local fusion arithmetic, no fixed `largest weight` claim, and no `FusionResult` current-authority use.

- [ ] **Step 2: Verify RED if any violation exists; otherwise verify test meaning by temporarily targeting a known forbidden fixture/string before final form**

The test must be demonstrated capable of failing for the intended rule.

- [ ] **Step 3: Update workflow focused test/analyze lists and forbidden-pattern shell checks for P5B files**

Keep full tests/analyze as visibility steps and do not hide unrelated baseline failures.

- [ ] **Step 4: Run focused CI**

Expected: all P0-P5B focused tests pass and focused analysis reports no issues.

- [ ] **Step 5: Commit**

Commit message: `test: enforce P5B clinician authority boundaries`

---

### Task 9: Final verification, documentation and PR

**Files:**
- Create: `docs/superpowers/verification/2026-09-16-clinanx-p5b-timeline-clinical-notes-verification.md`

**Interfaces:**
- No production interfaces.

- [ ] **Step 1: Run fresh focused verification**

Run the exact focused tests and analyzer from CI plus authority checks. Record counts and results.

- [ ] **Step 2: Run full test/analyze visibility**

Record every failure truthfully. If the known legacy `test/widget_test.dart` ConsentGate expectation remains the only full-suite failure, document it as pre-existing; do not call the full suite green.

- [ ] **Step 3: Compare base to head**

Review every changed file and ensure no teammate repository was modified.

- [ ] **Step 4: Write verification document**

Include base/head SHAs, commands/checks, focused results, full-suite results, CURRENT vs TARGET gaps, and known limitations.

- [ ] **Step 5: Open PR to `main`**

PR title: `feat: add ClinAnx P5B timeline and clinical notes`

Do not merge. Wait for explicit user approval.
