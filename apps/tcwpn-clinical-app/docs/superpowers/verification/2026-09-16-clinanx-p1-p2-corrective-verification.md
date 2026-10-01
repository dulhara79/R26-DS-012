# ClinAnx P1/P2 Corrective Verification

**Date:** 2026-09-16  
**Repository:** `dulhara79/tcwpn_mobile_app`  
**Base:** `main` at `1ee399aee47473a906de7d160d924d12ef8df56a`  
**Branch:** `fix/clinanx-p1-p2-verification-gate`

## Purpose

This corrective slice closes the verification gaps discovered after PR #19 was merged. It does not start P3 and it does not change teammate repositories.

## Backend contract gate

The Central Backend repository `UVINDUSEN/component4final` was rechecked before this corrective implementation. Its visible `main` remains at `a1ceef8daba268dac24d81aedd76c89e7c9ccc6a`, and the required target clinician contracts are still not verified there: clinician-principal/JWT auth, assignment-scoped clinician dashboard/patient reads, canonical latest-assessment aggregate, persisted `ForecastResult`, and persistent server `AttentionEvent` lifecycle.

Therefore the existing verification ruling remains in force:

**LIVE TARGET WIRING REMAINS BLOCKED.**

ClinAnx keeps the P0/P1/P2 domain contracts, repository interfaces, state/controller and fixture-backed Dashboard UI, but production target adapters must not call guessed documentation example routes. The target auth/dashboard/latest-assessment adapters now fail explicitly with `ApiFailure.notConfigured` until the backend owner supplies an implemented/OpenAPI-verified contract.

## Corrections applied

- Restored the backend verification gate by removing unverified live target route literals from `CentralBackendAuthRepository`, `CentralBackendDashboardRepository`, and `CentralBackendAssessmentRepository`.
- Added contract-gate repository tests proving those target adapters do not attempt network traffic while the backend contract is unverified.
- Mapped `ApiFailure.notConfigured` to an explicit Dashboard `unavailable` state rather than a generic error or fabricated data.
- Repaired the legacy Alerts navigation regression after `openChart()` moved to `patients_screen.dart`. Legacy alerts remain readable until the planned P4 migration; no new authoritative local escalation generation is restored.
- Corrected the obsolete gateway contract assertion so `403` means `forbidden`, distinct from `401` expired/invalid identity.
- Added CI authority checks preventing `/v1/subjects/attach`, `_raiseIfEscalated`, removed KPI dashboard authority code, local Dashboard risk derivation, and unverified target endpoint literals from returning.
- Removed P1/P2 analyzer noise introduced by the previous slice and removed duplicate/unused imports in the existing secure HTTP file without changing its TLS-pinning behavior.

## Verification evidence

GitHub Actions run `35053033279` on commit `187ff2c7c6c5bc4b74a47b7d106ba83dcad585d7` produced:

- Focused P0/P1/P2 tests: **87 passed**.
- Focused static analysis: **No issues found**.
- Authority / forbidden-pattern checks: **passed**.
- Whole-repository static analysis: **No issues found**.
- Whole-repository tests: **113 passed, 1 failed**.

The one remaining whole-suite failure is pre-existing test expectation drift in `test/widget_test.dart`: the test named `boots to sign-in when there is no session` constructs `ClinAnxApp(consented: false, signedIn: false)` but expects the Login screen. Current app boot order intentionally places the consent gate before sign-in, so `consented: false` renders `ConsentGateScreen`. This corrective P1/P2 slice does not alter consent/auth boot behavior; the failure is recorded rather than hidden because the approved P1/P2 plan permits known baseline failures to remain visible when they are distinguished from branch regressions.

## Teammate repositories

No changes were made to:

- `UVINDUSEN/component4final`
- `UVINDUSEN/Care-AnxRAG`
- `DewduSendanayake/anxiety_mobile_app`

Live target wiring must be revisited only after Uvindu provides the implemented endpoint/OpenAPI contract and representative responses required by the existing backend-contract gate document.

## Next planned feature

After this corrective PR is reviewed and merged, the next planned ClinAnx slice remains **P3 Patient Overview**. P4 Attention Events, P5 Deep Clinical Views, P6 Notifications, P7 Hardening and P8 Research Release remain later slices in the approved sequence.
