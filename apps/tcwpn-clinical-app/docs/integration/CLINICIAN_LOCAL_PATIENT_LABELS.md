# Clinician-entered patient display labels (Option A)

ClinAnx clinicians can set a **device-local study-safe label** on an already
assigned patient in the Patients roster (pencil button). The label replaces
the pseudonymous roster title and is passed to the detail-screen header.

- This is **not** a backend name, consented Aura profile attribute, identity
  proof, demographic input, or a modification of canonical `subject_id`.
- The central backend remains responsible for authentication, assignments,
  assessment data, and note analysis. No migration, encryption key, or new
  endpoint is needed.
- The clinician must have an active session and a fresh, assigned-patient roster.
  No label is loaded/displayed from an offline dashboard snapshot.
- Labels are stored via `flutter_secure_storage`, scoped to the authenticated
  clinician and canonical subject ID. The existing global secure-storage sign
  out calls `deleteAll()` and therefore **erases labels on sign out**.
  Labels are not synchronized across devices, and should be re-entered if
  needed after signing out.
- Only enter a research-approved nickname, not a legal name, unless local
  identifiable information is explicitly permitted and the device is secured.
- Use the **Remove label** button to restore the backend pseudonym.

## Verification
Run `flutter test test/data/local/patient_label_store_test.dart`,
`flutter test`, and `flutter analyze`. Verify in a device: clinician A's
labels do not appear in clinician B's session, only assigned patients are
editable, sign out wipes labels, and all backend requests still use canonical
subject IDs. Testing was not performed in this execution environment.

The older patient-name-sharing backend/Aura/ClinAnx PRs are superseded.
