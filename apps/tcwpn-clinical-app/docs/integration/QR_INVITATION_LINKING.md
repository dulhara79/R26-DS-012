# ClinAnx QR invitation linking (clinician-side)

## What this change supports
- Patients issue an authenticated one-use clinician assignment invitation via the Central Backend.
- ClinAnx **Patients > Link patient > Scan invitation QR** scans that invitation; the scanned token fills the existing form.
- The clinician taps **Accept invitation** to redeem through the *existing* `POST /v1/clinicians/me/assignments` route, using the authenticated clinician session.
- The Patients screen refreshes the server-owned, assignment-scoped roster. It does not add a local patient record from the QR.

## Compatibility: current Aura QR is NOT a consent token
At the reviewed Aura patient-app revision, **Connect to Doctor** shows a QR containing the static `P_<16 hex>` Participant ID. This identifier does not grant medical record access. Scanning it in ClinAnx shows an explanation and **never** submits the identifier as an invitation. In the current patient app, the patient must tap **Give clinician a code** and provide the displayed invite for manual entry.

The patient-app team may later render the **existing backend-issued invite token** as a QR (raw 43-character base64url token preferred; `clinanx://invite/<token>` is also understood). Do not encode the patient's raw ID or a long-lived credential as an invitation QR. Updating the Aura app is outside this clinician-only PR.

The scanner strictly accepts only 43-character backend-issued token shapes and the app-specific invite URI wrapper. Manual code entry continues to accept the pre-existing backend contract. The backend, not Flutter, validates issuer/expiry/single-use/access. Scanner success alone never grants access.

## Repro / verification
1. In Aura, sign in to the correct backend and request **Give clinician a code**.
2. In ClinAnx, log in as a clinician against the **same backend deployment**, go to Patients > Link patient, and enter the invitation. This manual path works without a patient-app update.
3. Once Aura supports an invitation QR, scan it in ClinAnx. Confirm the form contains the decoded invite, then tap **Accept invitation**.
4. Verify the assigned roster refreshes from the backend. Confirm a second redemption is rejected, an expired code is rejected, and unassigned clinicians cannot retrieve the patient.
5. Scan the original Participant ID QR and verify it does **not** create an assignment. Deny camera permission and verify manual entry remains available.
6. Run `flutter test test/features/patients/patient_invitation_qr_test.dart test/features/patients/link_patient_qr_test.dart test/features/patients/assignment_invite_flow_test.dart`, `flutter analyze`, and test on an actual Android/iOS device.

## Deployment notes
- Keep `BACKEND_BASE` configured to the same HTTPS backend for Aura and ClinAnx.
- Camera access is needed for scanning; Android manifest and iOS usage description are declared.
- Do not log invite tokens or patient records in scanner diagnostics.
- No new server endpoints, schema changes, or privileged app credentials are required for this QR front-end addition.
