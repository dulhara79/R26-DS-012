# ClinAnx Deployment Runbook

This runbook covers the mobile-side deployment and operational handoff for the R26-DS-012 ClinAnx research prototype.

> **Research prototype — not a diagnostic device.** This runbook does not establish clinical deployment readiness.

## 1. Prepare the environment

Install the pinned Flutter SDK used by the release candidate (`3.47.4` for Phase 9), Android tooling and a compatible Java 17 environment.

Verify:

```bash
flutter --version
flutter doctor
git rev-parse HEAD
git status --short
```

Use a clean source tree and the committed `pubspec.lock`.

## 2. Configure the build outside source control

Required research/study values include:

- `BACKEND_BASE`
- `AUTH_BASE`
- `APP_VERSION`
- `BUILD_ENVIRONMENT`
- `BUILD_REVISION`
- `DEMO_DATA=false`

For push, select either:

- Primary Firebase client configuration; or
- Secondary Firebase client configuration for the disaster-recovery build.

Do not store passwords, service private keys, server credentials or signing secrets in the repository.

## 3. Validate endpoint safety

For any non-loopback environment:

- backend/auth URLs must be HTTPS;
- Settings must identify the expected backend host/environment;
- no secret is displayed;
- local demo authentication must not be used with real participant data.

If the app rejects a plaintext remote endpoint, fix the environment configuration rather than weakening the client.

## 4. Build

Follow `APK_BUILD.md` exactly for Primary or Secondary builds.

After build, calculate and record the APK SHA-256.

## 5. Install

Example Android installation:

```bash
adb install -r build/app/outputs/flutter-apk/app-release.apk
```

If replacing a build with a different application identity/signature, uninstall/reinstall may be required. Preserve any approved study data-migration procedure before doing so.

## 6. First-launch smoke test

Verify in order:

1. App launches without a crash.
2. Consent gate appears when required.
3. Clinician can authenticate using the configured auth mode.
4. Settings shows the expected app version, environment, source revision, backend host and Firebase slot/project.
5. Dashboard either loads server data or presents an explicit unavailable/error state.
6. Patients/Patient Overview do not fabricate low risk when data is missing.
7. Current multimodal assessment and forecast are clearly separate.
8. C2 is visibly experimental/excluded when present.
9. CARE evidence failure/abstention is explicit.
10. Sign-out clears the session.

## 7. AttentionEvent and notification smoke test

When a matching backend is available:

1. Ensure an OPEN server AttentionEvent exists for an assigned clinician.
2. Confirm Activity can discover it through the server contract.
3. If FCM is configured, confirm the notification contains generic routing information only.
4. Open the notification and confirm ClinAnx fetches the canonical event detail.
5. Confirm notification receipt/open does not automatically ACK or RESOLVE the event.
6. Acknowledge the event and confirm the server response replaces local state.
7. Resolve it and confirm the server response remains canonical.
8. Restart the app and confirm the server lifecycle state is still represented correctly.

If push is unavailable, keep the app resumed long enough to verify the 30-second polling fallback discovers the OPEN event.

## 8. Primary/Secondary Firebase procedure

### Primary build

Use `PUSH_FIREBASE_SLOT=primary` with the Primary Firebase project values.

### Secondary DR build

Use `PUSH_FIREBASE_SLOT=secondary` with the Secondary Firebase project values.

The two builds must display different slot/project identity in Settings. Do not attempt to recover by swapping only an API key at runtime; registration tokens belong to the project that issued them.

## 9. Common failures

| Symptom | Mobile-side action |
|---|---|
| `401` | Re-authenticate; do not show cached clinical state as current authority |
| `403` | Treat as forbidden/assignment denial; do not retry as another patient locally |
| `409` on event mutation | Refresh the canonical event and present the server state |
| timeout/offline | Keep safe local work and show explicit offline/unavailable state |
| stale assessment | Show stale/last-updated provenance; never present as fresh |
| no push permission | Continue Activity/polling fallback |
| FCM unavailable/unconfigured | Keep core app usable; rely on persistent event polling while resumed |
| malformed push | Ignore; do not navigate to a fabricated event |
| RAG unavailable | Show supporting evidence unavailable; do not invent guidance |
| Firebase slot mismatch | Rebuild with the correct project configuration and verify Settings identity |

## 10. Rollback

Keep the last accepted artifact with:

- commit SHA;
- APK hash;
- environment;
- Firebase slot/project;
- app version;
- known backend contract compatibility.

If a new build fails:

1. stop distribution;
2. collect logs without PHI/secrets;
3. reinstall the last accepted artifact if contract-compatible;
4. record the failure in the release evidence/defect log;
5. produce a new versioned candidate after the fix.

## 11. Handoff information

Provide the next operator/researcher with:

- `PHASE9_RESEARCH_RELEASE.md`;
- `SOP.md`;
- this runbook;
- `TEST_EVIDENCE.md`;
- `KNOWN_LIMITATIONS.md`;
- `APK_BUILD.md`;
- the exact artifact + SHA-256;
- the exact source revision;
- the intended backend environment and Firebase slot/project.

Do not include passwords, private keys, signing secrets or participant data in the handoff document.
