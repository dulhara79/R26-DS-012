# ClinAnx Phase 7 Usability Checklist

**Purpose:** Repeatable research-prototype usability pass for ClinAnx hardening. This is not a clinical-validation instrument and does not replace backend/API acceptance testing.

**Build under test:** record APK/app version, Git commit, BACKEND_BASE environment, auth mode, and date before starting.

## Test setup

- Use synthetic/demo participants only.
- Use two clinician test identities when checking local privacy isolation.
- Prepare fixtures for complete, partial, unavailable, stale-C1, unavailable-C3, RAG-abstained, RAG-unavailable, OPEN/ACKNOWLEDGED/RESOLVED AttentionEvent, 401, 403 and 409 states.
- Run once at normal system text size and once with enlarged system text.

## Tasks and expected behavior

| # | Clinician task | Expected safe behavior | Pass |
|---|---|---|---|
| 1 | Sign in as clinician A, open a cached patient, sign out, then sign in as clinician B. | B cannot see A's local roster, dashboard cache, pending notification open, notes/support/fusion cache or legacy alerts. | [ ] |
| 2 | Open Dashboard and identify a participant requiring attention. | Needs Attention is based on server-provided event/assessment state; no client-side risk recomputation is presented as authoritative. | [ ] |
| 3 | Open Patient Overview and explain Current assessment versus Forecast. | Current multimodal assessment and forecast are visually/textually separate; TC-WPN is a contributing Clinical NLP signal. | [ ] |
| 4 | Open a partial assessment with stale C1 and unavailable C3. | Stale/unavailable labels are explicit. Missing data is never rendered as Low, Green or zero. | [ ] |
| 5 | Lose network access after a successful server read. | Cached server data, if shown, is clearly marked Offline/stale/last known. No old score is presented as current. | [ ] |
| 6 | Open an OPEN AttentionEvent and acknowledge it. | UI sends the server mutation, then displays the canonical returned event state. Notification delivery itself is not treated as acknowledgement. | [ ] |
| 7 | Simulate another clinician changing the same event before the action completes. | 409/conflict causes refresh/reconciliation with server state; the app does not fabricate a local winner. | [ ] |
| 8 | Resolve an acknowledged event. | RESOLVED state/actor/time come from the server response. | [ ] |
| 9 | Ask CARE when RAG abstains. | UI displays an abstained/no-sufficiently-grounded-evidence state and does not manufacture fallback guidance. | [ ] |
| 10 | Ask CARE when RAG times out/unavailable. | UI displays Supporting evidence unavailable; timeout is not converted into an answer. | [ ] |
| 11 | Expire the clinician token. | Controlled session-expired/sign-in recovery appears rather than Low risk or generic assessment success. | [ ] |
| 12 | Attempt to open an unassigned/forbidden patient fixture. | Access is denied with 403 semantics; no cached data for another clinician/patient is exposed. | [ ] |
| 13 | Repeat Dashboard → Patient → Activity → Event Detail → Settings with enlarged system text. | Critical labels/actions remain readable and operable; meaning is not conveyed by color alone. | [ ] |
| 14 | Review Settings security/governance copy. | Note routing says Central Backend → Clinical NLP; pinning status distinguishes explicitly pinned hosts from platform TLS; no unverifiable security claim is shown. | [ ] |

## Observer prompts

Ask the tester to explain, in their own words:

1. Which number/state is the current multimodal assessment?
2. Which information is a forecast rather than the current assessment?
3. What does an unavailable/stale signal mean?
4. Is Clinical NLP / TC-WPN the overall patient-risk engine?
5. Does receiving a phone notification mean an event has been acknowledged?
6. What should they do when supporting evidence is unavailable or abstained?

A failure to distinguish these concepts should be logged as a usability defect even if the software technically works.

## Defect log

| ID | Build/commit | Task | Observed problem | Severity | Owner | Resolution / retest |
|---|---|---|---|---|---|---|
| P7-U-001 | | | | | | |

## Completion rule

Phase 7 usability evidence is complete only when every P0 safety task above passes or the remaining limitation is explicitly documented as an external/backend dependency. Do not mark live end-to-end behavior as verified from fixture-only testing.