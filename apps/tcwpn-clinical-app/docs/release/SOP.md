# ClinAnx Research Release SOP

This SOP describes the repeatable mobile-side procedure for an R26-DS-012 ClinAnx research release.

> **Research prototype — not a diagnostic device.** Use this SOP for engineering/research reproducibility only.

## 1. Purpose

The goal is to produce a traceable ClinAnx build whose source revision, dependencies, environment, Firebase routing and verification evidence can be reconstructed later.

## 2. Roles

- **Release owner:** selects the candidate commit, records release metadata and stops the release when a blocker appears.
- **Mobile verifier:** runs automated/mobile smoke checks and records results.
- **Backend/integration owner:** verifies server-side assignment, event persistence, cross-client identity and end-to-end behavior outside this repository.

One person may hold multiple roles, but evidence must still identify what was actually checked.

## 3. Preconditions

Before starting:

1. Phase 8 mobile hardening changes are merged.
2. Phase 9 documentation/tests are present.
3. No new feature work is mixed into the release candidate.
4. The intended research environment and Firebase slot are known.
5. Required credentials/configuration are available outside source control.
6. Any release-blocking known limitation has an owner or explicit research-demo exception.

## 4. Freeze the candidate

Run:

```bash
git checkout main
git pull --ff-only
git status --short
git rev-parse HEAD
flutter --version
```

Record the exact commit and Flutter version. The working tree must be clean.

For the Phase 9 release process, use Flutter `3.47.4` unless a later release deliberately changes the pinned SDK and updates the release documentation/CI together.

## 5. Restore dependencies reproducibly

Use the committed lockfile:

```bash
flutter pub get
```

Do not run dependency upgrades during release preparation. Dependency upgrades are separate engineering changes and require their own verification.

## 6. Automated verification

Run locally when a matching environment is available:

```bash
flutter test
flutter analyze
```

Also require the PR-to-`main` GitHub Actions gates to be green. Record the exact run IDs in `TEST_EVIDENCE.md`.

If a check fails:

- stop the release;
- record the failure;
- fix the root cause on a branch;
- rerun the complete gate;
- do not convert a failed check into a manual pass without a documented reason.

## 7. Build configuration review

Confirm:

- remote `BACKEND_BASE` and `AUTH_BASE` use HTTPS;
- `APP_VERSION` matches `pubspec.yaml`;
- `BUILD_REVISION` equals the exact source commit;
- `BUILD_ENVIRONMENT` identifies the intended environment;
- `DEMO_DATA=false` for participant/research use;
- Firebase Primary/Secondary values match the selected build slot;
- no service/server credential is embedded in the mobile build.

## 8. Build the artifact

Use the commands in `APK_BUILD.md`.

Produce either:

- Primary research APK; and/or
- Secondary Firebase disaster-recovery APK.

Do not describe the Secondary artifact as seamless runtime failover. It is a separately configured build.

## 9. Record artifact identity

After build:

1. calculate SHA-256;
2. record APK filename;
3. record app version;
4. record `BUILD_REVISION`;
5. record environment;
6. record backend host/environment identifier;
7. record Firebase slot/project ID;
8. record build date/tester.

## 10. Manual smoke and usability verification

Use synthetic/demo participants unless the approved research environment and governance process allow otherwise.

At minimum verify:

- launch → consent → sign-in;
- Dashboard loads or fails explicitly;
- Patient Overview separates current assessment from forecast;
- unavailable/stale states remain explicit;
- Activity shows server AttentionEvent lifecycle state;
- notification permission denial leaves polling/Activity usable;
- configured push opens by event identity when real FCM setup is available;
- receiving notification does not acknowledge/resolve the event;
- Settings shows correct non-secret release identity;
- sign-out clears session state.

Execute `docs/qa/phase8_usability_checklist.md` for the formal manual hardening pass.

## 11. Cross-system dress rehearsal

The mobile release owner coordinates with the backend/patient-app owners for the handbook end-to-end scenario. ClinAnx alone cannot prove:

- assignment authorization;
- same `fusion_result_id` across audiences;
- server event persistence/concurrency;
- patient + clinician delivery of the same event;
- backend audit durability.

Record these results separately rather than claiming them from mobile CI.

## 12. Release or stop

Release only when required automated and manual checks are complete for the intended use.

Stop the release when:

- a safety-critical automated test is red;
- the wrong environment/Firebase project is configured;
- `BUILD_REVISION` does not match source;
- participant data would traverse plaintext HTTP;
- unavailable data is presented as reassuring risk;
- notification receipt changes event lifecycle locally;
- an unresolved limitation is incompatible with the planned demo/study.

## 13. Rollback

If the candidate fails after installation:

1. stop further distribution;
2. capture build revision and failure evidence;
3. revert to the last accepted APK/build revision;
4. ensure backend schema/contract compatibility before reopening the older client;
5. file the defect and create a new candidate rather than silently replacing the artifact under the same release identity.
