# ClinAnx P3 Patient Overview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 3 Patient Overview as a fusion-first clinician screen showing a separate near-term forecast, current multimodal assessment, C1-C4 signal state, provenance, freshness, and honest unavailable/offline states without inventing backend routes.

**Architecture:** Reuse the P0 `AssessmentSummary` contract and `AssessmentRepository` boundary. Add a dedicated `PatientOverviewController` and screen. While Uvindu's latest-assessment clinician contract remains unverified, production repository calls remain contract-gated (`ApiFailure.notConfigured`); widget/controller tests use fixture/fake repositories. Existing legacy chart/notes functionality remains reachable as a deeper view and is not made authoritative for the P3 overview.

**Tech Stack:** Flutter/Dart, Provider-compatible `ChangeNotifier`, existing design system, existing P0 contracts, GitHub Actions.

**Spec:** `ClinAnx_Clinician_Mobile_App_Technical_Architecture_and_UX_Specification(1).pdf` — Phase 3 / Patient Overview; `R26-DS-012_System_Integration_Implementation_Handbook(1).pdf` — ClinAnx workflow and authority constraints.

## Global Constraints

- Forecast and current multimodal assessment remain separate objects and separate UI sections.
- Fusion is primary; TC-WPN is only the C3 Clinical NLP signal.
- C2 remains explicitly experimental/excluded when `notValidated` and not included in fusion.
- Missing/unavailable/stale data must never be converted to Low or numeric zero.
- No local fusion calculation or local authoritative escalation derivation.
- Do not call C3 `/predict` directly.
- Do not live-wire `/v1/patients/{id}/assessment/latest` or other target clinician routes until the backend contract is implemented and verified.
- Preserve 401/403/409 semantics from P1/P2.
- P4 attention-event actions and P5 deep clinical views are out of scope; P3 may show non-authoritative navigation entry points only where existing functionality already exists.

---

### Task 1: P3 verification gate and controller contract

**Files:**
- Create: `lib/state/patient_overview_controller.dart`
- Create: `test/patient_overview_controller_test.dart`
- Modify: `.github/workflows/p1_p2_dashboard_ci.yml`

**Interfaces:**
- Consumes: `AssessmentRepository.latestAssessment(String subjectId)` and `AsyncDataState<AssessmentSummary>`.
- Produces: `PatientOverviewController.load()` and explicit loading/data/empty/unavailable/offline/sessionExpired/forbidden/conflict/error states.

- [ ] **Step 1: Write failing controller tests** for successful assessment loading, null assessment -> unavailable, `notConfigured` -> unavailable, 401 -> sessionExpired, 403 -> forbidden, 409 -> conflict, and network/timeout -> offline.
- [ ] **Step 2: Add the P3 branch and P3 focused tests/analyzer paths to CI, then push tests only.**
- [ ] **Step 3: Verify RED** — CI must fail because `PatientOverviewController` does not yet exist.
- [ ] **Step 4: Implement minimal controller** using `AssessmentRepository`; no backend route literals.
- [ ] **Step 5: Verify GREEN** with focused controller tests and analysis.

### Task 2: Fusion-first Patient Overview UI

**Files:**
- Create: `lib/features/patients/patient_overview_screen.dart`
- Create: `test/features/patients/patient_overview_screen_test.dart`

**Interfaces:**
- Consumes: `PatientOverviewController`, `AssessmentSummary`, optional display label.
- Produces: `PatientOverviewScreen(subjectId, displayId, controller)`.

- [ ] **Step 1: Write failing widget tests** proving forecast renders before current assessment, physiological scope is labelled, current score/tier are server values, confidence and uncertainty are separate, C1-C4 are visible, C2 shows `Experimental — not included in fusion`, C3 shows `Clinical NLP / TC-WPN`, stale C1 stays stale, unavailable C3 never renders `0`, and unavailable assessment never renders Low.
- [ ] **Step 2: Verify RED** — tests fail because the screen does not yet exist.
- [ ] **Step 3: Implement the minimal screen** using existing design primitives and conservative labels.
- [ ] **Step 4: Verify GREEN** with widget tests and focused analysis.

### Task 3: Navigation without inventing live backend behavior

**Files:**
- Modify: `lib/features/dashboard/server_dashboard_screen.dart`
- Modify: `lib/features/patients/patients_screen.dart`
- Create/modify tests under `test/features/dashboard/` and `test/features/patients/`.

**Interfaces:**
- Dashboard card uses canonical `PatientSummary.subjectId` and `displayId`.
- Legacy Patients roster resolves Aura participant ID through the already verified `/v1/subjects/resolve?app_user_id=...` compatibility path before opening P3; if resolution is unavailable it keeps the existing legacy chart path rather than fabricating a canonical subject ID.

- [ ] **Step 1: Write failing navigation tests** for Dashboard -> Patient Overview and safe legacy Patients behavior.
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Add navigation callbacks/helpers** without wiring a guessed latest-assessment endpoint.
- [ ] **Step 4: Verify GREEN.**

### Task 4: Authority guards and full verification

**Files:**
- Modify: `.github/workflows/p1_p2_dashboard_ci.yml`
- Create: `test/p3_patient_overview_authority_test.dart`
- Create: `docs/superpowers/verification/2026-09-16-clinanx-p3-patient-overview-verification.md`

- [ ] **Step 1: Add source guards**: no `AlertBandX.fromScore` in P3 overview, no local fusion math, no direct C3 `/predict`, no unverified target latest-assessment route literal in production adapters.
- [ ] **Step 2: Run focused P0/P1/P2/P3 tests and focused analysis.**
- [ ] **Step 3: Run whole-repository tests and analysis as visibility gates and record exact results.**
- [ ] **Step 4: Compare branch to `main`, inspect diff for scope creep, and record backend-contract status.**
- [ ] **Step 5: Open a PR to `main`; do not merge without user approval.**
