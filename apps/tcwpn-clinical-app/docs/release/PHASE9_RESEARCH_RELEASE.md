# Phase 9 — ClinAnx Research Release

This document is the mobile-side release manifest for **R26-DS-012 System Integration Handbook Phase 9**.

> **Research prototype — not a diagnostic device.** This release package documents engineering reproducibility and verification. It is not clinical deployment approval.

## 1. Handbook deliverables

Phase 9 requires:

- pinned versions;
- SOP;
- test evidence;
- deployment/runbook;
- known limitations.

ClinAnx satisfies those mobile-side documentation requirements through:

- `README.md`
- `ARCHITECTURE.md`
- `APK_BUILD.md`
- `docs/release/SOP.md`
- `docs/release/DEPLOYMENT_RUNBOOK.md`
- `docs/release/TEST_EVIDENCE.md`
- `docs/release/KNOWN_LIMITATIONS.md`

## 2. Release identity

| Field | Phase 9 value / rule |
|---|---|
| Application | ClinAnx clinician mobile app |
| App version | `1.0.0+1` |
| Flutter SDK | `3.47.4` for Phase 9 CI/release verification |
| Dependency graph | committed `pubspec.lock` |
| Phase 9 base | Phase 8 merged `main` commit `4e9e5510acfb9d13e11c13cd03f59672bcbb831d` |
| Release source revision | must equal the exact final build commit injected as `BUILD_REVISION` |
| Research environment | supplied per artifact, e.g. `study` / `study-dr` |
| Firebase routing | Primary or Secondary build slot; project ID recorded per artifact |
| APK hash | record SHA-256 after artifact is actually built |

Do not pre-fill backend/service/model versions from memory. Record the versions reported by the integrated research environment at release time.

## 3. Mobile release invariants

- ClinAnx does not compute authoritative multimodal fusion.
- Current assessment and forecast are separate.
- Missing/stale/unavailable data is never converted to a reassuring risk state.
- C2 remains explicitly experimental/excluded while that research rule is active.
- TC-WPN remains a Clinical NLP signal rather than overall multimodal authority.
- CARE-AnxRAG failure and abstention are displayed honestly.
- Clinician session identity is used for mobile clinical calls.
- Remote participant/clinical traffic requires HTTPS.
- Push payloads carry routing identity only.
- Notification receipt never changes AttentionEvent clinical lifecycle state.
- FCM failure does not remove the server-backed Activity/polling recovery path.

## 4. Release candidate checklist

### Source and configuration

These stay open until the actual release artifact is selected and built.

- [ ] Final release commit/tag selected.
- [ ] `BUILD_REVISION` equals that exact commit.
- [ ] `git status --short` is clean before build.
- [ ] `pubspec.lock` is committed and unchanged during artifact build.
- [ ] `APP_VERSION` matches `pubspec.yaml` for the artifact.
- [ ] Correct `BUILD_ENVIRONMENT` recorded.
- [ ] Correct HTTPS backend/auth endpoints selected.
- [ ] `DEMO_DATA=false` for participant/research build.
- [ ] Correct Firebase slot and project selected.

### Automated verification

The Phase 9 mobile engineering gate has been executed on the PR. Recorded evidence is in `TEST_EVIDENCE.md`.

- [x] Phase 9 PR focused tests green.
- [x] Authority/privacy guards green.
- [x] Full `flutter test` green.
- [x] Full `flutter analyze` green.
- [x] P0 contract workflow green.

Latest verified documentation/evidence head before this checklist update: `ccaaf44053e92e42a46b74f325efbf4925c31910`.

- integration and hardening checks — run `35238965194` — success
- P0 contract verification — run `35238965209` — success

Any later commit must receive its own successful PR workflows before merge.

### Manual mobile verification

These remain intentionally unchecked until a human performs them.

- [ ] Phase 8 usability/hardening checklist executed by a human.
- [ ] Primary build installed and smoke-tested on a physical/emulated device as applicable.
- [ ] Secondary Firebase DR build identity checked if the DR artifact is part of the release.
- [ ] Session expiry checked.
- [ ] Notification permission denial checked.
- [ ] Push event open checked with a real configured Firebase project where available.
- [ ] Polling fallback checked.
- [ ] Offline/stale display checked.
- [ ] No secrets/PHI observed in diagnostic UI/logs.

### Cross-system evidence — external to this repository

These must not be marked complete based only on ClinAnx CI:

- [ ] Central Backend assignment authorization verified.
- [ ] Patient and clinician projections share the same `fusion_result_id` for the same assessment.
- [ ] Server AttentionEvent persists across app restarts.
- [ ] Multi-clinician ACK/RESOLVE concurrency reconciles to one server state.
- [ ] Patient and assigned clinician notifications reference the same event.
- [ ] End-to-end dress rehearsal completed.

## 5. Artifact record

Fill this only after a real artifact is built and tested.

| Field | Recorded value |
|---|---|
| Release commit | Capture from `git rev-parse HEAD` |
| Build revision shown in app | Capture from Settings |
| APK filename | Capture actual artifact name |
| APK SHA-256 | Capture after build |
| Environment | Capture actual environment |
| Backend host | Capture actual configured host |
| Firebase slot | Capture `primary` or `secondary` |
| Firebase project ID | Capture actual project ID |
| Automated CI run IDs | Record final green Phase 9 runs |
| Manual tester/date | Record only after execution |

## 6. Release decision rule

Do **not** describe Phase 9 as a fully validated research release until:

1. automated Phase 9 verification is green;
2. the required human usability/smoke checks are performed;
3. the artifact identity/hash is recorded;
4. cross-system release blockers relevant to the intended demonstration/study are verified or explicitly accepted as documented limitations.

A CI-green mobile branch proves mobile engineering checks; it does not by itself prove whole-system clinical or research deployment readiness.
