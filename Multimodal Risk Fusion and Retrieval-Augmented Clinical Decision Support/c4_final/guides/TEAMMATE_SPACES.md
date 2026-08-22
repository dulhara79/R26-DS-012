# Teammate Hugging Face Spaces — what's in the repo

I cloned `github.com/dulhara79/R26-DS-012` and read the actual code. Here is what
is really there, versus what you still have to ask for. **Read the "status"
column carefully — only two of the three are confirmed live, and neither returns
a plug-and-play score for your fusion.**

---

## Summary

| Component | Space / repo | Confirmed? | Ready to fuse as-is? |
|---|---|---|---|
| **C1 physiological** | `Dewdu/physiological-anxiety-escalation` (Space)<br>`Dewdu/physiological-anxiety-weights` (dataset) | ✅ in code | ⚠️ needs an adapter — see C1 below |
| **C2 behavioural** | — | ✅ withheld (null result) | N/A — weight 0, don't call it |
| **C3 clinical notes** | `dulharakaushalya-tc-wpn-demo.hf.space` | ✅ live URL in the app | ⚠️ needs note text as input — see C3 below |

The exact base URLs:

- **C1:** `https://Dewdu-physiological-anxiety-escalation.hf.space`
  *(derived from the Space repo id `Dewdu/physiological-anxiety-escalation`; confirm by opening it — Spaces sleep, so the first hit is slow)*
- **C3:** `https://dulharakaushalya-tc-wpn-demo.hf.space`
  *(hard-coded in `apps/tcwpn-clinical-app/lib/services/api_service.dart` line 8)*

---

## C1 — Physiological (Dewdu)

**Where I found it:** `Self-supervised Physiological Biosensors/HF-Anxiety/main.py`, a FastAPI Space. Repo ids are in `scripts/personalize.py`:

```python
HF_SPACE_REPO   = "Dewdu/physiological-anxiety-escalation"
HF_WEIGHTS_REPO = "Dewdu/physiological-anxiety-weights"
```

**This is the important part — C1 is NOT a simple "send patient, get score" service.**
It is a per-user forecasting service backed by a time-series database (InfluxDB). It has three endpoints:

| Endpoint | What it does |
|---|---|
| `POST /ingest` | the patient's chest strap streams feature packets in here |
| `POST /set_norm_params/{user_id}` | stores the 3-minute calibration baseline |
| `GET /predict/{user_id}` | returns the current risk + a 10-step forecast for that user |

So you don't POST a patient's data to get a score. The strap has already been feeding `/ingest` for that `user_id`, and you **GET** the latest forecast by user id.

**The response from `GET /predict/{user_id}`** (from `main.py` lines 632–645):

```json
{
  "status": "success",
  "message": "Personalized physiological forecast ready.",
  "forecast": [ ... ],
  "adjusted_error_forecast": [ ... ],
  "risk_forecast": [0.42, 0.44, 0.47, ...],
  "current_reconstruction_error": 0.13,
  "current_risk_index": 41.8,
  "reconstruction_error_threshold": 0.11
}
```

**The field you want is `current_risk_index`.** Two things about it:

1. It is on a **0–100 scale**, not 0–1. Your harmoniser will percentile-rank it so the scale stops mattering — but know it going in.
2. There is **no `confidence` and no `coverage`** in the response. This is exactly Ask 1 from before. Either Dewdu adds them, or your client fills defaults (0.5 / 1.0), which weakens your reliability weighting for C1.

**For your reference distribution (Part 7):** ask Dewdu for the `current_risk_index` values across their held-out evaluation, OR compute percentiles from `risk_forecast` history. Don't use the raw reconstruction error — use the risk index, since that's what you're fusing.

---

## C2 — Behavioural

Confirmed **withheld**. The parent paper reports it did not beat its permutation
null (AUROC 0.5205 vs 0.4991). Your `fusion.py` already gives it weight 0 and
leaves `C2_URL` blank. Nothing to connect. This is correct, not a gap.

---

## C3 — Clinical notes / TC-WPN (Dulhara)

**Live Space:** `https://dulharakaushalya-tc-wpn-demo.hf.space` — confirmed, hard-coded in the clinician app.

**This one takes a clinical note as input, not a patient id.** From `api_service.dart`, the call is:

```
POST https://dulharakaushalya-tc-wpn-demo.hf.space/predict
Content-Type: application/json

{
  "note_text":       "<the clinical note the doctor typed>",
  "note_type":       "<e.g. progress / admission>",
  "anxiety_support": [],
  "control_support": []
}
```

**The response** (from the app's `PredictionResult.fromJson`, `models.dart` lines 148–153):

```json
{
  "risk_level":  "high",     // low | moderate | high | very_high
  "risk_score":  0.68,
  "confidence":  0.83
}
```

Good news: the score field is `risk_score` (your `clients.py` already recognises
that) and it **already includes `confidence`**. So C3 is the closest to
plug-and-play — but it needs the doctor's note text, which means the fusion
service can only include C3 *after* a note exists for that patient. Before the
first note, C3 is simply an unavailable modality (which your masked softmax
already handles).

**Health check:** `GET /health` exists and the app calls it on startup to wake the Space. Do the same before your first fuse.

---

## What this means for your fusion service — an important correction

Your current `clients.py` assumes every component takes the **same** request
(`{patient_id, mrn}`) and returns a **score directly**. That was the right
starting assumption, but the real Spaces don't work that way:

- **C1** wants `GET /predict/{user_id}` (no body), returns `current_risk_index` on 0–100
- **C3** wants `POST /predict` with `note_text`, returns `risk_score` on 0–1
- **C4** (yours) wants `POST /fusion_component` with demographics, returns `score` on 0–1

They need **three different adapters**, not one generic caller. I've written those
into `clients_real.py` (next file). Each adapter knows how to call its own Space
and pull the right field, then hands back the same `Reading` object so the rest
of your fusion is unchanged.

---

## The identity problem you must solve first

C1 keys on `user_id`. C3 keys on note text tied to a patient in the doctor's app.
C4 keys on the demographics submitted at enrolment. The doctor's app keys on `mrn`.
The patient app keys on `user_id`.

**Nothing joins these automatically.** Before the fusion can assemble one patient's
four scores, you need a mapping:

```
mrn  ⇄  user_id  ⇄  patient record in the clinician app
```

This is the Week-1 integration item. Decide with your teammates who owns the
mapping table and where it lives. Concretely: when the doctor registers a patient,
that's where `mrn` and the patient-app `user_id` get linked — capture both in one
place at that moment.

---

## Your action list

1. **Open both Spaces in a browser to confirm they're live** (they sleep, so wait for the wake-up):
   - `https://Dewdu-physiological-anxiety-escalation.hf.space/`
   - `https://dulharakaushalya-tc-wpn-demo.hf.space/health`
2. **Run the probe** on each to see the live response with your own eyes:
   ```
   python probe_space.py https://dulharakaushalya-tc-wpn-demo.hf.space
   ```
   (For C3 the probe's generic bodies won't include `note_text`, so it may not
   pull a score — that's expected. Use `/health` and the shapes above.)
3. **Use `clients_real.py`** instead of the generic `clients.py`.
4. **Ask Dewdu (C1)** to add `confidence` + `coverage` to `/predict`, and for held-out `current_risk_index` values.
5. **Ask Dulhara (C3)** for held-out `risk_score` values (they already send confidence — good).
6. **Settle the `mrn ⇄ user_id` mapping** with the app owners.
