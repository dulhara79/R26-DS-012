# Anxiety Digital Biomarker Console — rebuild

Clinician-facing app for **R26-DS-012**. One product, one theme, one navigation
model. The four research components appear as sections of a patient chart, never
as separate apps.

---

## Structure

```
lib/
  main.dart                          entry, session gate
  core/
    config/env.dart                  all service endpoints, build-time injected
    design/tokens.dart               colour, spacing, radius, alert bands
    design/theme.dart                the single ThemeData
    design/components.dart           shared vocabulary + FusionBar (signature)
  domain/models.dart                 all four components + fusion, defensive parsing
  data/
    api/api_client.dart              one HTTP client, typed failures
    api/gateways.dart                TcwpnGateway, C3Gateway, ModalityGateway, FusionGateway
    local/stores.dart                SecureStore (keychain) + RecordStore (per-MRN)
  state/controllers.dart             RosterController, ChartController
  features/
    auth/login_screen.dart
    shell.dart                       caseload, patients, add-patient
    chart/patient_chart_screen.dart  Risk | Clinical notes | Intervention
    tcwpn/note_analysis_screen.dart
    tcwpn/tcwpn_result_screen.dart
    tcwpn/support_set_screen.dart
    fusion/fusion_detail_screen.dart
    alerts/alerts_screen.dart
    settings/settings_screen.dart
```

## Running

```bash
flutter run \
  --dart-define=TCWPN_BASE=https://dulharakaushalya-tc-wpn-demo.hf.space \
  --dart-define=FUSION_BASE=https://<org>-anxiety-fusion.hf.space \
  --dart-define=C3_BASE=https://<org>-c3-intervention.hf.space \
  --dart-define=C1_BASE=https://<org>-wearable.hf.space \
  --dart-define=C2_BASE=https://<org>-tbg.hf.space \
  --dart-define=HF_TOKEN=hf_xxx \
  --dart-define=DEMO_DATA=false
```

A component with no address is excluded from the composite and the remaining
fusion weights are rescaled. Nothing is hard-coded to `10.0.2.2`.

## Dependencies

```yaml
dependencies:
  flutter: {sdk: flutter}
  provider: ^6.1.2
  http: ^1.2.2
  shared_preferences: ^2.3.2
  flutter_secure_storage: ^9.2.2
  google_fonts: ^6.2.1
  intl: ^0.19.0
  uuid: ^4.5.1
  fl_chart: ^0.69.0
  pdf: ^3.11.1
  printing: ^5.13.2
  share_plus: ^10.0.2
```

---

## Design direction

**Clinical paper.** Cool near-white ground (`#F7F9FA`), deep slate-teal ink
(`#0F5B6E`), hairline structure instead of drop shadows.

Three type roles, deliberately paired:

| Role | Face | Used for |
|---|---|---|
| Display | Inter Tight, tight tracking | Screen titles, hero numbers |
| Body | Inter | Everything read as prose |
| Data | IBM Plex Mono, tabular figures | Risk scores, thresholds, weights, MRNs, latency |

The mono register is the point. A score of `0.7134` is instrument output; setting
it in the same face as body copy makes it read as an opinion.

**Signature element — `FusionBar`.** The proposal's §5.1 late-fusion equation,
drawn at full size: one stacked bar where each segment's width is that modality's
weighted contribution, the remainder is risk not accounted for, and modalities
with no reading appear as gaps rather than silently vanishing. It appears compact
on every patient row and full-size at the top of every chart, so the framework's
central idea is the one thing a clinician sees everywhere.

Component colours are fixed and never reused for anything else: sand-blue =
wearable, sage = behavioural, clay = intervention, brand teal = clinical notes.
Learn it once, and it holds across every screen.

---

## What each component gets

### Component 4 — TC-WPN (this project)

The deepest surface in the app, because it carries `w₄ = 0.40`.

- **`note_analysis_screen`** — support-set readiness stated *before* analysis:
  K = 0 means meta-trained prototypes with no site adaptation, and the clinician
  is told so. Note is saved before the network call, so a sleeping Space never
  costs anyone their typing.
- **`tcwpn_result_screen`** —
  - calibrated probability against the locked decision threshold, drawn in place
    on the 0–1 scale so `0.41` vs `0.4036` reads differently from `0.95`
  - confidence, entropy, K, raw score, ECE
  - prototype distances for both classes
  - **attention attribution rendered only from weights the service returned.**
    When the service sends phrases without weights, the section is retitled "key
    phrases", the intensity ramp is dropped, and the difference is stated. The
    old build derived prominence from list index and labelled it *"ClinicalBERT
    attention weights"* — a claim the model never made.
  - **support-set influence**: which labelled examples shaped the prototype, with
    the recency and confidence weights broken out. This is TC-WPN's actual novelty
    made visible, and no screen in the old build showed it.
  - **HITL verdict capture**: agree / disagree, stored on the note, plus one-tap
    promotion of a reviewed note into the support set. Few-shot adaptation now has
    a closed loop.
- **`support_set_screen`** — site and per-patient scopes, class balance warning,
  and a required note date (recency weighting discounts older notes, so defaulting
  to "today" would silently overweight a three-year-old example).

### Fusion layer

`FusionGateway.fuse` → composite, band, per-modality contributions. On failure it
returns `FusionResult.local`, which renormalises across reporting modalities and
is labelled **provisional** everywhere it appears. Bands per §5.1: GREEN < 0.25,
AMBER < 0.50, RED < 0.75, DARK RED ≥ 0.75.

### Components 1, 2, 3

Consumed through `ModalityGateway` and `C3Gateway`. Missing readings degrade the
composite honestly rather than being imputed. The `Intervention` chart section
renders C3's tier, APS conformal set, SHAP, DiCE and FAISS cases as soon as
`C3_BASE` is configured — models are in `domain/models.dart` and parsing is done.

---

## Wire contracts

**`POST {TCWPN_BASE}/predict`**
```jsonc
{ "patient_id": "P001", "note_text": "...", "note_type": "Psychiatry note",
  "note_date": "2026-07-31T09:00:00Z", "visit_count": 3,
  "support_set": [{"id":"…","text":"…","label":"anxiety","note_date":"…"}],
  "return_attention": true, "return_support_contributions": true }
```
Response — everything optional except `risk_score` / `confidence` / `threshold`:
```jsonc
{ "prediction":"ANXIETY", "risk_score":0.71, "calibrated_probability":0.68,
  "confidence":0.83, "entropy":0.41, "threshold":0.4036, "support_k":12,
  "ece":0.061, "prototype_distance_anxiety":0.42, "prototype_distance_control":0.88,
  "attention_spans":[{"text":"difficulty controlling the worry","weight":0.19}],
  "support_contributions":[{"note_id":"…","label":"anxiety","excerpt":"…",
     "temporal_weight":0.87,"confidence_weight":0.92,"combined_weight":0.80,
     "note_date":"2025-11-02"}],
  "temporal_context":"Visit 3 of 5", "model_version":"TC-WPN v1.0", "latency_ms":840 }
```

**`POST {FUSION_BASE}/fuse`**
```jsonc
{ "patient_id":"P001", "renormalise_on_missing": true,
  "components": { "c1_physiological": {"score":0.62,"available":true,"captured_at":"…"},
                  "c2_behavioral": {"score":null,"available":false},
                  "c3_intervention": {"score":0.55,"available":true},
                  "c4_clinical_nlp": {"score":0.68,"available":true} } }
```
```jsonc
{ "composite_score":0.638, "alert_level":"RED", "renormalised":true,
  "modalities_available":3, "confidence":0.81,
  "weights":{"c1_physiological":0.3125,"c3_intervention":0.1875,"c4_clinical_nlp":0.50},
  "scores":{"c1_physiological":0.62,"c2_behavioral":null,"c3_intervention":0.55,"c4_clinical_nlp":0.68},
  "computed_at":"2026-07-31T09:02:11Z" }
```

---

## Defects from the old build, and where each is fixed

| Defect | Fix |
|---|---|
| `StorageService.activePatientId` static → cross-patient contamination | `ChartController.mrn` is `final`; every `RecordStore` call takes the MRN as a required argument |
| C3 seeded with `token: ''` → every GAD-7 silently used the local heuristic | Auth removed from the C3 path; `C3Gateway` uses the shared client with real credentials |
| No authentication in front of patient records | `LoginScreen._authenticate` is a real integration point that returns null until wired |
| "Biometric login: Enabled" switch with empty callback | Removed. Settings reports only true state |
| Four different AUROC figures across three screens | No performance literals in the UI; Settings renders the service's own `/health` |
| Attention explanation derived from list index | `AttentionSpan.hasWeight`; UI retitles and drops the ramp when weights are absent |
| Hardcoded dashboard metrics (0.342, 94%, 4.6k) | Caseload computes every figure from the roster |
| `removePatient` left notes and assessments orphaned | `RecordStore.purgePatient` clears every namespace |
| Support-set manager and notifications unreachable | Support set is reachable from Caseload and from every chart; Alerts is a root destination |
| Unguarded `fromJson` → `StateError` / type crashes | Every field goes through `_d`/`_i`/`_s`/`_b` with defaults |
| `Patient.initials` `RangeError` on double-spaced names | Splits on `RegExp(r'\s+')` and filters empties |
| `Navigator..pop()..pop()` ate the patient chart | Single pop / `pushReplacement` |
| Sign-out was a no-op inside the shell | Clears the keychain and resets the navigator stack |
| Two divergent patient registries | One `RosterController` over one `RecordStore` roster |
| Three gender vocabularies, "Other" recoded as Female | One list in `AddPatientSheet`; `c3Demographics()` encodes the third category explicitly |
| Token in `SharedPreferences` | `flutter_secure_storage` |

---

## Remaining work

1. **`_authenticate`** in `login_screen.dart` — wire to the hospital IdP. Nothing
   else changes.
2. **PDF export** — port `pdf_service.dart`, adding the TC-WPN attribution table,
   the fusion breakdown, and the clinician verdict to the report.
3. **Intervention section** — the C3 widgets (conformal set, SHAP bars, DiCE,
   FAISS cases). Models and parsing are already in `domain/models.dart`.
4. **Offline queue** — retry drafts and failed fusion calls on reconnect;
   `connectivity_plus` listener plus a pending-work store.
5. **Accessibility pass** — `Semantics` on `FusionBar` segments and `BandChip`, so
   risk is not colour-only. Text scaling is already clamped to 1.3×.
6. **Tests** — `FusionResult.local` renormalisation, `AttentionSpan.hasWeight`
   fallback, `RecordStore.purgePatient` completeness, and every `fromJson` against
   an empty map.

---

## One research note

The blinded evaluation you ran — TC-WPN at AUROC 0.8989 vs ProtoNet 0.9291 once
anxiety terms are masked — is why the attention-attribution rule in this build is
strict. If the model is partly keying on lexical shortcuts, then a UI that
fabricates plausible-looking attribution is actively concealing the finding from
the person best placed to notice it. Showing real weights, or honestly showing
none, keeps the app aligned with what the evaluation actually supports.
