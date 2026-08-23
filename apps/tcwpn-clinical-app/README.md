# R26-DS-012 — Unified Clinical Console

One Flutter app that hosts **both** research components behind a single launcher:

| Module | Component | Owner | What it does |
|---|---|---|---|
| **TC-WPN** | Temporal-Confidence Weighted Prototypical Networks | Kaushalya D. | Few-shot anxiety detection from clinical notes; patient roster; support-set (few-shot) management; PDF/share export. Talks to the Hugging Face Space `/predict`. |
| **C3** | Personalised Intervention Framework | Seneviratne K.A.U.A. | Calibrated XGBoost risk tiering with conformal prediction, GAD-7 tracking, XAI-driven activity recommendation, and the continuous-learning reward loop. Talks to the FastAPI `/v3/*` backend. |

## Architecture — how the two apps became one

The two components were originally **separate Flutter apps** with heavy overlap in
class names (`HomeScreen`, `LoginScreen`, `NotificationService`, `RiskBadge`,
`SectionHeader`, `StatCard`, `models.dart`) and two different risk models
(`PredictionResult`/`RiskLevel enum` vs `RiskResult`/`int tier`). A naive
file-merge would collide on every one of those.

**Chosen strategy: module isolation under a shared launcher.**

```
lib/
├── main.dart                 # root app → PortalLauncher
├── shell/
│   └── portal_launcher.dart  # clinician console: pick a module
├── tcwpn/                     # Component 1 — self-contained
│   ├── tcwpn_entry.dart       # exposes TcwpnEntry (own MaterialApp + Provider)
│   ├── theme/  models/  services/  widgets/  screens/
└── c3/                        # Component 2 — self-contained
    ├── c3_entry.dart          # exposes C3Entry (own MaterialApp + splash)
    ├── config/ models/ services/ widgets/ screens/
```

Each module is launched as **its own `MaterialApp`** (isolated theme + navigator +
state). Because the two trees never import each other, the duplicate class names
never clash — `tcwpn/…/HomeScreen` and `c3/…/home_screen.dart#HomeScreen` live in
separate libraries. This preserves **both teams' code essentially unchanged** and
keeps the diff reviewable.

## Changes applied during the merge

New files: `main.dart`, `shell/portal_launcher.dart`, `tcwpn/tcwpn_entry.dart`,
`c3/c3_entry.dart`, and `tcwpn/theme/app_theme.dart` (the original was never
supplied, so it was reconstructed to satisfy every `AppColors.*` token and
`TextTheme` style the TC-WPN screens use).

Four correctness/safety fixes (each flagged during review):

1. **TC-WPN `api_service.dart`** — guarded a `substring(0, 80)` that threw a
   `RangeError` whenever the encoded request body was shorter than 80 chars.
2. **C3 `risk_result_screen.dart`** — the tier-0 explainer hardcoded
   *"The model is 100% certain"* regardless of the real value; now shows the
   actual `confidence`.
3. **C3 `storage_service.dart`** — `clear()` called `prefs.clear()` (wiped
   **all** SharedPreferences); now removes only the `c3_*` keys, so a co-hosted
   module's data can never be destroyed.
4. **TC-WPN `patient_provider.dart`** — `NotificationService.showRiskNotification`
   existed but was never called; now fires a real device notification on
   high / very-high risk.

## Setup

This bundle contains `lib/` + `pubspec.yaml`. Generate the platform folders and run:

```bash
flutter create .          # scaffolds android/ ios/ etc. into this folder
flutter pub get
flutter run
```

### Backends to point at

- **TC-WPN**: `lib/tcwpn/services/api_service.dart` → `_baseUrl`
  (default: the HF Space `https://dulharakaushalya-tc-wpn-demo.hf.space`).
- **C3**: `lib/c3/config/api_config.dart` → `base`
  (default `http://10.0.2.2:8000`, i.e. localhost from the Android emulator;
  change to your deployed URL for a physical device).

### Android cleartext note

The C3 default backend is plain HTTP. For a physical-device demo over HTTP you
must allow cleartext in `android/app/src/main/AndroidManifest.xml`:

```xml
<application android:usesCleartextTraffic="true" ... >
```

For anything beyond a demo, both backends should move to HTTPS and the stored
auth token / PHI should be moved from `SharedPreferences` to secure storage.

## Known follow-ups (not blocking a build)

- Both modules store PHI / auth tokens in plaintext `SharedPreferences`.
- C3 encodes NHANES features (`genderEnc`, `maritalEnc`, `incomeEnc`) client-side;
  this belongs server-side with the calibrated pipeline.
- C3 `home_screen` recomputes risk tier from the GAD-7 score alone for its summary
  card, which can disagree with the full classifier result shown on the risk
  screen. Align to a single source of truth before submission.
- TC-WPN AUROC / threshold figures are scattered across screens (login vs
  settings); centralise them.
- The C3 module is written in patient voice. If the console is to be strictly
  clinician-operated, the C3 flow should be re-framed as "run assessment for the
  selected patient" rather than self-signup — a follow-up task.

---

## Shared patient identity (v1.1)

Both modules now operate on **one canonical patient**, so a TC-WPN note analysis
and a C3 GAD-7 belong to the same record.

**Flow:** `PatientPickerScreen` (pick/create patient) → `PortalLauncher(patient)`
→ TC-WPN / C3 / **Unified record**, all scoped to the same MRN.

**How the join works:**

- `lib/shared/patient_identity.dart` — `PatientIdentity` (keyed by **MRN**) plus a
  `PatientRegistry` (persisted list + active patient), provided at the app root.
  This layer imports nothing from `tcwpn/` or `c3/`, so it stays cycle-free.
- **TC-WPN** (`tcwpn_entry.dart`): when handed a patient, `ensurePatient()` upserts
  that MRN into the roster and opens straight to their detail screen, so notes
  attach to that MRN.
- **C3** (`c3_entry.dart`): when handed a patient, it sets
  `StorageService.activePatientId = MRN`, synthesises a C3 user from the shared
  identity (`userId == MRN`), and skips self-signup. Every GAD-7 / session /
  reward value is stored under that MRN.
- **C3 storage is now namespaced per MRN** (`storage_service.dart`): keys become
  `c3_*__<MRN>`. Standalone C3 (no active patient) falls back to the original
  global keys, so the module still runs on its own.
- `lib/shared/patient_record_screen.dart` — reads the TC-WPN roster
  (`patients_v1`) **and** the C3 store for that MRN and renders both on one screen.

**Caveat for clinician-operated C3:** there is no C3 auth token in this mode, so
the `/v3/*` backend calls are best-effort and fall back to C3's built-in local
computation (the GAD-7 screen already classifies locally on any error). To sync
clinician-operated sessions to the C3 backend, thread a real token through
`PatientIdentity` / the synthesised `UserModel`. The local unified record works
regardless.

---

## v1.2 — Single-app restructure (supersedes the launcher model)

Earlier builds embedded each component as its *own* MaterialApp behind a
launcher, which made the product feel like two apps. That is gone. Now:

- **One MaterialApp, one light theme.** A single `AppPalette` is the source of
  truth; both components' colour tokens (`AppColors`, `C3Theme`) are redefined to
  point at it, so every existing screen re-themes without being individually
  edited. No more teal-vs-green split, no per-module splash screen.
- **One navigation shell** (`shared/app_shell.dart`): a single bottom nav —
  **Patients · Dashboard · Settings**. There is no second bottom nav anywhere.
- **Components live inside the patient chart.** Opening a patient shows
  `PatientHomeScreen`: one screen with a segmented control —
  **Overview / Clinical notes / Intervention**. TC-WPN note analysis and the C3
  GAD-7 / risk / intervention flows are sections here, reached as ordinary detail
  routes under the one theme. The doctor never "switches apps".
- **Providers hoisted to the root** (`PatientProvider` + `PatientRegistry`) so
  state is shared app-wide.
- Removed: `portal_launcher.dart`, `patient_picker_screen.dart`,
  `tcwpn_entry.dart`, `c3_entry.dart`, `patient_record_screen.dart` (folded into
  the patient chart's Overview).
- The old standalone TC-WPN and C3 `HomeScreen`s (each with its own 4-tab bottom
  nav) and the C3 signup/login screens are no longer routed to in the doctor
  flow; they remain in the tree but are unused.

**Remaining follow-up (optional):** a few deep C3 screens still address the
patient in second person ("How do you feel right now?", "Your results"). If the
console must read strictly as clinician-to-clinician everywhere, those strings
need a wording pass — structurally they're already inside the single app.
