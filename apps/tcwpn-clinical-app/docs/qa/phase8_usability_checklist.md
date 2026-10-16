# ClinAnx Handbook Phase 8 Usability / Hardening Checklist

**Purpose:** Manual evidence for the System Integration Handbook Phase 8 hardening scope: offline/stale behavior, security review, failure injection, usability, and reproducible configuration.

**Important:** This is research-prototype evidence, not clinical validation. Keep every item unchecked until a human actually performs it.

## Build under test

Record before testing:

- APK/app version:
- Git commit / build revision:
- build environment:
- BACKEND_BASE host:
- auth mode:
- Firebase slot: primary / secondary:
- Firebase project ID:
- DEMO_DATA state:
- test date:
- tester:

Use synthetic/demo participants only.

## Manual scenarios

| # | Scenario | Expected safe behavior | Pass |
|---|---|---|---|
| 1 | Sign in as clinician A, use cached clinical views, sign out, then sign in as clinician B. | B cannot see A's clinician-scoped local clinical cache or pending notification state. | [ ] |
| 2 | Load an assessment, then disconnect the network. | Previously cached information is explicitly marked Offline/stale/last known; old data is not presented as current. | [ ] |
| 3 | Open a partial assessment with stale C1 and unavailable C3/C4. | Missing/stale signals remain unavailable/partial; they never become Low, Green, or zero. | [ ] |
| 4 | Expire the clinician session token. | App presents session-expired/sign-in recovery; no successful/low-risk state is fabricated. | [ ] |
| 5 | Use a forbidden/unassigned participant response fixture. | 403/forbidden is shown distinctly; no other participant's cached details are exposed. | [ ] |
| 6 | Simulate 409 while acknowledging/resolving an AttentionEvent. | App refreshes/reconciles canonical server state rather than inventing a local winner. | [ ] |
| 7 | Simulate backend timeout / 5xx. | Existing local work remains intact; failure is explicit and not converted into a clinical result. | [ ] |
| 8 | Simulate RAG timeout/unavailable. | Supporting evidence is shown unavailable; no fallback clinical guidance is invented. | [ ] |
| 9 | Simulate RAG abstention. | Abstention is shown separately from service failure and no unsupported answer is generated. | [ ] |
| 10 | Deny OS notification permission. | ClinAnx remains usable and server-backed polling/Activity remains available; permission denial does not remove the fallback path. | [ ] |
| 11 | Make FCM unavailable or leave Firebase unconfigured in a test build. | Consent/sign-in/clinical navigation still work; push failure does not block the app and polling remains the recovery path. | [ ] |
| 12 | Deliver the same AttentionEvent through push and polling. | Event identity is deduplicated; notification delivery does not ACK/RESOLVE the event. | [ ] |
| 13 | Deliver a malformed/wrong-type push. | It is ignored safely and cannot navigate to a fabricated event. | [ ] |
| 14 | Attempt a test push containing patient name, MRN, note text, or score fields. | Payload is rejected by the client privacy guard; clinical detail is fetched only after authenticated event open. | [ ] |
| 15 | Review Settings research-build information. | App version/build environment/backend host/auth mode/Firebase slot+project/demo state are understandable; no token, password, API key, salt, note text, or other secret/PHI is displayed. | [ ] |
| 16 | Compare a primary-slot build with a secondary-slot DR build. | Each build clearly reports its active Firebase slot/project; no claim of seamless runtime project switching is made. | [ ] |
| 17 | Attempt a non-local HTTP clinical endpoint in a controlled test build. | Request is refused as insecure; normal remote clinical traffic requires HTTPS. | [ ] |
| 18 | Repeat Dashboard -> Patient -> Activity -> Event Detail -> Settings using enlarged system text. | Critical state labels and actions remain readable/operable and meaning is not conveyed by color alone. | [ ] |
| 19 | Review Current Assessment and Forecast with a tester unfamiliar with the app. | Tester can distinguish current multimodal assessment from forecast and understands TC-WPN is a signal, not the overall risk authority. | [ ] |
| 20 | Review an OPEN event notification then open the event. | Tester understands receiving a notification is not the same as acknowledging or resolving the event. | [ ] |

## Observer questions

Ask the tester to explain in their own words:

1. Which state is the current multimodal assessment?
2. Which information is a forecast?
3. What does stale/unavailable mean?
4. Is TC-WPN the overall multimodal risk result?
5. What happens if push delivery fails?
6. Does receiving a notification acknowledge an event?
7. Which Firebase project is this build using?
8. What information is intentionally absent from a push notification?

Record misunderstandings as usability defects even if automated tests pass.

## Defect log

| ID | Build/commit | Scenario | Problem | Severity | Owner | Resolution / retest |
|---|---|---|---|---|---|---|
| P8-U-001 | | | | | | |

## Completion rule

Do not mark the manual usability portion of Handbook Phase 8 complete until every safety-critical scenario above has been run by a human and either passes or has a documented limitation/owner. Automated fixture tests and CI are engineering evidence; they are not a substitute for this human usability pass.
