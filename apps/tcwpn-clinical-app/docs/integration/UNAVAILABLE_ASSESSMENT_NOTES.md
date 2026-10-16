# ClinAnx: unavailable assessment and clinical-note navigation

## Source-level root cause

The Central Backend `central_backend/clinician_api.py::_assessment_wire` intentionally returns a prior `fusion_result_id` when the last fusion is no longer valid for **current** risk. The same response sets `assessment_status="unavailable"` and `current_assessment=null`. For a newly enrolled patient with no prior fusion, `fusion_result_id` is null as well.

Previously, ClinAnx's `CentralBackendAssessmentRepository.latestAssessment` rejected **any** non-null `fusion_result_id` for an unavailable assessment as a malformed server response. The patient overview then displayed an error screen with no navigation to the separate note-editor workflow.

## Corrected ClinAnx behavior

- Accept `assessment_status="unavailable"`, `current_assessment=null` and a valid **historical** `fusion_result_id`. Keep this ID only as provenance; do not claim it is a current assessment or synthesize a score.
- Continue to reject contradictory unavailable payloads containing a current score, tier or band, non-positive fusion identifiers, or mismatched subject identity.
- On errors where the assessment is unavailable, offline or malformed, show links for **View clinical notes** and **View timeline** independently. Clinical note drafts remain *device-local* until submitted for analysis; a backend session and clinician-to-subject assignment are needed for server actions.
- Do **not** expose these navigation actions from the patient overview when the server explicitly rejects authorization (401/403) or reports a record conflict (409). Patient-specific backend operations still enforce authorization.
- Do not surface supporting evidence or clinical review that depends on a valid current `AssessmentSummary` until that data is available.

## Verification

```sh
flutter test test/repositories/assessment_unavailable_contract_test.dart
flutter test test/features/patients/patient_overview_recovery_navigation_test.dart
flutter test test/features/patients/patient_overview_screen_test.dart
flutter test test/features/patients/clinical_notes_screen_test.dart
flutter test test/contracts/p0_invariants_test.dart
flutter analyze
```

On a test device, use an authorized clinician assigned to a synthetic subject; verify that a newly linked patient without fusion history opens as "Assessment unavailable" **with** the clinical-note action. Verify the same after a previously valid fusion snapshot has become stale. Confirm no historical score is displayed as current and server note submission continues to enforce assignment permissions. A physical-device test using the configured backend is required before claiming integration verified.

The September handbook is a system design baseline, not a claim that the October live deployment was tested. No Central Backend or Aura changes are included in this PR.
