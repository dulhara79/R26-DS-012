# ClinAnx Phase 9 Test Evidence

This file records mobile-side engineering evidence for the R26-DS-012 Handbook Phase 9 research-release package.

> Evidence is recorded only when it actually exists. Automated CI does not substitute for human usability or whole-system verification.

## 1. Phase 8 baseline

Phase 8 hardening was merged to `main` as commit:

`4e9e5510acfb9d13e11c13cd03f59672bcbb831d`

The Phase 8 PR head used commit:

`404bfb403253edee039aa512a748c248aaabf45d`

Recorded final Phase 8 workflow evidence:

- ClinAnx integration and hardening checks — run `35234426458` — success
- ClinAnx P0 contract verification — run `35234426420` — success

Those runs covered focused tests, authority/privacy guards, full `flutter test`, and analysis.

## 2. Phase 9 TDD RED evidence

Phase 9 documentation acceptance test was introduced first at commit:

`849551e3ee6541c6bc0906d04e69b389d96b8c0e`

PR workflow run:

- ClinAnx integration and hardening checks — run `35236252072` — expected failure

Expected RED result:

- 328 existing tests passed
- 4 Phase 9 documentation assertions failed

The failures were intentional and specifically showed that:

1. the Phase 9 `docs/release/` set did not yet exist;
2. `README.md` did not link to the release package;
3. active release documentation still contained the obsolete shared mobile backend-token instruction;
4. the public README did not yet state the explicit research-prototype / non-diagnostic-device boundary.

This demonstrates that the new Phase 9 test could detect the documentation drift before the implementation was written.

## 3. Phase 9 GREEN evidence

The Phase 9 release-documentation implementation commit was:

`78596e3abde56dc4e8aa1a391c9790d09b1f4ae5`

Recorded PR workflow evidence for that commit:

- ClinAnx integration and hardening checks — run `35237214844` — **success**
- ClinAnx P0 contract verification — run `35237214921` — **success**

The integration/hardening workflow recorded:

- focused suite: **301 tests passed**;
- focused `flutter analyze`: **No issues found**;
- authority/privacy/forbidden-pattern checks: **success**;
- full regression suite: **332 tests passed**;
- full `flutter analyze`: **No issues found**.

The four Phase 9 release-documentation tests passed in both the focused and full suites. They verify that the release-documentation set exists, the README links to the release evidence/runbook, active release docs do not reintroduce the obsolete shared mobile backend credential, and the public research-use boundary is explicit.

A later documentation-only evidence update may produce a newer PR head commit/run. The run IDs above remain the recorded GREEN evidence for the Phase 9 implementation commit; final merge readiness must still be checked against the latest PR head before merge.

## 4. Dependency/toolchain evidence

For the Phase 9 release process:

- app version: `1.0.0+1`
- Flutter SDK pinned in CI/release documentation: `3.47.4`
- dependency graph: committed `pubspec.lock`
- source identity: `BUILD_REVISION` must equal the actual build commit

A real artifact release must also capture `flutter --version`, the Git SHA and APK SHA-256 at build time.

## 5. Automated behavior already covered

The existing regression suite includes evidence for, among other things:

- no local authoritative fusion on server-backed clinician views;
- stale/missing signals do not become zero/Low;
- current assessment and forecast remain separate;
- C2 remains explicit experimental/excluded evidence;
- AttentionEvent lifecycle uses canonical server responses;
- notification delivery does not manufacture ACK/RESOLVE authority;
- push payload is routing-only;
- push/poll duplicate delivery is deduplicated;
- clinician-scoped local cache/notification state;
- session-expiry/forbidden/conflict error semantics;
- failure-injection and accessibility guards;
- no reusable privileged service credential in active mobile Dart;
- HTTPS requirement for non-loopback clinical endpoints.

## 6. Manual evidence — not yet fabricated

The following remains human work until actually executed:

- `docs/qa/phase8_usability_checklist.md`
- physical/emulated release APK smoke test
- real configured Primary Firebase push delivery
- Secondary Firebase DR artifact check, if part of the release
- notification permission-denied scenario on target OS/device
- offline/stale usability review
- large-text/manual accessibility review

Record tester/date/build/commit when these are run.

## 7. Cross-system evidence — outside mobile CI

This repository alone cannot prove:

- server-side assignment enforcement;
- same patient/clinician `fusion_result_id` from a live integrated backend;
- event persistence across backend restarts;
- multi-clinician lifecycle concurrency;
- patient and clinician notification delivery of the same server event;
- backend audit durability;
- full end-to-end dress rehearsal.

These must be supplied by the integration/backend test process and referenced in the final project release evidence.

## 8. Evidence policy

A release claim must distinguish:

- **automated mobile evidence** — reproducible GitHub/local test results;
- **manual mobile evidence** — human/device checklist results;
- **cross-system evidence** — backend/patient/integration verification;
- **known limitation** — a gap that is documented rather than falsely marked passed.
