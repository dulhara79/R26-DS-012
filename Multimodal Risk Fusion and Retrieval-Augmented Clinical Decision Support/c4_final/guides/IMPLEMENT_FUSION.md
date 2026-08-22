# Implementing the fusion model — step by step

**Component 4 · R26-DS-012**

You already have `fusion.py`, which is the *maths*. What was missing is everything around it: something to call your teammates' services, something to put the four scores on the same scale, and something your Flutter app can actually talk to.

That's what this part builds. When you finish, you'll have a web service that takes a patient ID and returns one final risk tier.

**Time:** about 45 minutes. No dataset needed. No teammates needed — you'll test against fake ones.

---

## What you're building

```
   Doctor's app asks: "what's the risk for patient NHSL-0142?"
                              │
                              ▼
              ┌───────────────────────────────┐
              │   FUSION SERVICE (app.py)     │
              └───────────────────────────────┘
                              │
        ┌──────────┬──────────┼──────────┬──────────┐
        ▼          ▼          ▼          ▼          │  ① ask all four at once
      C1         C2          C3         C4          │     (clients.py)
    physio    behaviour    notes      yours         │     slow one = "missing"
        │          │          │          │          │
        └──────────┴──────────┴──────────┘          │
                              │                     │
                              ▼                     │  ② put on same scale
                   percentile mapping               │     (harmonise.py)
                              │                     │
                              ▼                     │  ③ weight & combine
                  ω × recency × reliability         │     (fusion.py)
                              │                     │
                              ▼                     │
                   composite + tier ────────────────┘
```

Five files do this:

| File | What it does |
|---|---|
| `fusion.py` | the maths — weights, combining, banding *(you already have this)* |
| `harmonise.py` | **new** — puts four different score scales onto one axis |
| `clients.py` | **new** — calls the four services in parallel, handles failures |
| `app.py` | **new** — the web service the doctor's app talks to |
| `mock_components.py` | **new** — fake teammates, so you can test today |

Plus `test_fusion_service.py`, which proves it all works.

---

## Part 1 · Set up the folder

Inside your `component4` folder, make a new folder called `fusion_service` and put these files in it:

```
component4/
└── fusion_service/
    ├── app.py
    ├── clients.py
    ├── harmonise.py
    ├── fusion.py              ← copy of the one from fusion/
    ├── mock_components.py
    ├── test_fusion_service.py
    ├── requirements.txt
    ├── Dockerfile
    ├── .env.example
    └── reference/
        ├── c1_physiological.json
        ├── c3_clinical_nlp.json
        └── c4_demographic.json
```

Then install what it needs. Activate your virtual environment first (look for `(.venv)` in your prompt), then:

```bash
cd fusion_service
pip install -r requirements.txt
```

---

## Part 2 · Understand harmonisation, because it's the part people skip

This is the most important idea in the whole component, and it's easy to miss why it matters.

Your four teammates send numbers that **look** comparable but aren't:

| Component | What its number actually is | Typical range |
|---|---|---|
| C1 physiological | reconstruction error from an autoencoder | 0.02 – 0.35 |
| C2 behavioural | a GATv2 logit | unbounded |
| C3 clinical notes | prototype distance, threshold locked at 0.4036 | 0.3 – 0.9 |
| C4 yours | probability of GAD-7 ≥ 10, base rate 2% | 0.00 – 0.30 |

Now look at what happens if you just average them:

> C3 sends **0.62**. C4 sends **0.62**. Are these the same amount of risk?
>
> **No.** For C3, 0.62 is a fairly ordinary score — plenty of patients score higher. For your model, 0.62 is enormous — almost nobody scores that high, because your base rate is 2%. Averaging them treats a routine C3 reading and an extreme C4 reading as equal.

So before anything is combined, every score is converted to its **percentile** — "how high is this compared to everyone else this component has scored?" Now 0.62 from C3 might become 0.55 (a bit above average) while 0.62 from your model becomes 0.99 (off the charts). *Those* numbers are comparable, and those are what get weighted.

You'll see this working in Part 5.

⚠️ **The `reference/` files I gave you are placeholders — randomly generated so the pipeline runs.** They are good enough to develop and demo against. They are **not** valid for your paper. Part 7 explains how to replace them with real ones.

---

## Part 3 · Start the fake teammates

Your teammates' services don't exist yet. `mock_components.py` pretends to be them — and deliberately pretends *badly*, because that's what you'll actually get:

- **C1** sends a raw unbounded score with full metadata
- **C2** answers confidently but should be ignored entirely (zero weight)
- **C3** uses a different field name (`risk_score` not `score`), gives no `coverage`, and its timestamps are days or months old
- there's also a `/slow` endpoint that takes 20 seconds and a `/broken` one that returns an error

Open a terminal, go to `fusion_service`, and run:

```bash
uvicorn mock_components:app --port 7900
```

Leave it running. This is **terminal 1**.

---

## Part 4 · Start the fusion service

Open a **second** terminal. Activate the venv again (new terminal = must reactivate), go to `fusion_service`, and tell it where the components live:

**Mac / Linux:**
```bash
export C1_URL=http://127.0.0.1:7900/c1/predict
export C2_URL=
export C3_URL=http://127.0.0.1:7900/c3/predict
export C4_URL=http://127.0.0.1:7900/c1/predict
uvicorn app:app --reload --port 7861
```

**Windows PowerShell:**
```powershell
$env:C1_URL="http://127.0.0.1:7900/c1/predict"
$env:C2_URL=""
$env:C3_URL="http://127.0.0.1:7900/c3/predict"
$env:C4_URL="http://127.0.0.1:7900/c1/predict"
uvicorn app:app --reload --port 7861
```

> **Why is C2 blank?** Because the behavioural component gets zero weight. Leaving its URL empty is how you say "this one is permanently unavailable" — and it's more honest than calling a service whose answer you throw away.

> **Why does C4 point at the C1 mock?** Just so you have something answering while your real Space is being set up. Replace it in Part 6.

Now open **http://127.0.0.1:7861/health** in your browser. You should see your base weights:

```json
{
  "base_weights": {
    "c1_physiological": 0.2303,
    "c2_behavioral": 0.0,
    "c3_clinical_nlp": 0.4603,
    "c4_demographic": 0.3094
  }
}
```

Behavioural is 0.0. That's the pre-registered exclusion rule holding.

---

## Part 5 · Run the tests

Open a **third** terminal, activate the venv, go to `fusion_service`:

```bash
python test_fusion_service.py
```

Seven tests. Here's what each one is checking and why it matters — this is what you'll be asked about in a review, so read it rather than just checking it says PASS.

### Test 1 — service is configured
Base weights are right, reference distributions loaded.

### Test 2 — the maths is sane
Composite lands between the lowest and highest input, weights sum to 1, clinical notes carry the most weight (because they have the highest AUROC).

### Test 3 — the day-one guard
Only demographics available → **no tier at all**, with a reason. Not "Low risk".

> This is your most important safety behaviour. On day one you have demographics and nothing else. Saying "low risk" about a newly admitted psychiatric patient based only on their age and education would be actively dangerous. Be ready to explain this one.

### Test 4 — the zero-weight rule
Behavioural sends 0.99 — as high as possible. Composite doesn't move by a single decimal place:

```
without behavioural  : 0.6292
with behavioural 0.99: 0.6292
```

### Test 5 — recency
Same physiological score, once fresh and once months old:

```
physio weight, reading 1 minute old : 0.2156
physio weight, reading months old   : 0.0
```

Nobody typed those numbers. They fall out of the 30-minute half-life.

### Test 6 — harmonisation, visible
```
c1_physiological     raw 0.1269  ->  percentile 0.6125
c3_clinical_nlp      raw 0.7517  ->  percentile 0.8883
c4_demographic       raw 0.2123  ->  percentile 0.9042
```

Look at the last line. Your model sent **0.21** — which sounds low — but against your own distribution that's the **90th percentile**. Without harmonisation, that patient's demographic risk would have been badly understated in the composite. **This is the single best line to put on a slide.**

### Test 7 — live smoothing
Fourteen minute-by-minute readings. The composite climbs gradually rather than jumping, so one anxious minute doesn't trigger an alarm.

**All seven passing means your fusion layer works.** Screenshot that output — it's evidence for your report.

---

## Part 6 · Connect your real model

Once your DCAR Space is live (Part 9 of the setup guide), point at it. Stop the fusion service (`Ctrl+C`) and restart with:

```bash
export C4_URL=https://YOURNAME-dcar-demographic-risk.hf.space/fusion_component
export C4_TOKEN=r26ds012-dcar-8f3k2n9x
uvicorn app:app --reload --port 7861
```

Then in your browser go to **http://127.0.0.1:7861/docs**, find **POST /v1/fuse**, click "Try it out", and send:

```json
{ "mrn": "NHSL-0142" }
```

Your own model is now feeding the fusion over the internet.

### Instead of environment variables, use a `.env` file

Retyping those exports gets old. Rename `.env.example` to `.env`, fill in your real URLs, and add this at the very top of `app.py`:

```python
from dotenv import load_dotenv
load_dotenv()
```

⚠️ **Never commit `.env` to GitHub** — it has your tokens in it. Add a file called `.gitignore` containing the single line `.env`.

---

## Part 7 · Replace the placeholder references

The `reference/` files are random numbers. Before any result goes in your paper, replace them with real score distributions.

**For your own component** — you already have this. Your notebook saved `dcar_reference_scores.npy`:

```python
import numpy as np
from harmonise import save_reference

scores = np.load("../notebooks/artefacts/dcar_reference_scores.npy")
save_reference("c4_demographic", scores,
               source="DCAR test split, Zenodo cohort",
               model_version="dcar-v1.0")
```

**For your teammates' components** — ask each of them for one thing:

> "Can you send me a CSV with the risk scores your model produced on your **held-out evaluation set**? Just the numbers, one per line. I need them to put our scores on the same scale before fusing."

Then:

```python
import numpy as np
from harmonise import save_reference

scores = np.loadtxt("c3_heldout_scores.csv")
save_reference("c3_clinical_nlp", scores,
               source="TC-WPN held-out cohort, n=2278",
               model_version="tcwpn-v2.1")
```

> ⚠️ **Held-out scores, never training scores.** If they send you scores from the data their model trained on, those scores are optimistic — the model has seen those patients. Your reference distribution would inherit their overfitting and every percentile you compute would be wrong.

---

## Part 8 · What the doctor's app receives

The response is already shaped for the Flutter app's `FusionResult.fromJson`:

```json
{
  "composite_score": 0.8322,
  "tier": "High",
  "alert_level": "RED",
  "weights":  { "c1_physiological": 0.2206, "c2_behavioral": 0.0,
                "c3_clinical_nlp": 0.4834, "c4_demographic": 0.2960 },
  "scores":   { "c1_physiological": 0.6125, "c2_behavioral": null, ... },
  "contributions": { "c3_clinical_nlp": 0.4294, ... },
  "modalities_available": 3,
  "renormalised": false,
  "confidence": 0.7968,
  "harmonisation": { "c4_demographic": { "raw": 0.2123, "harmonised": 0.9042 } }
}
```

- **`weights`** already sum to 1 over available streams — that's what `FusionBar` draws
- **`contributions`** = weight × score, so the doctor sees *which* stream drove the tier
- **`harmonisation`** is your audit trail — raw in, percentile out, for every component
- **`tier: null`** means the system refused to judge; the app must show that state, not fall back to Low

---

## Part 9 · The endpoints

| Endpoint | When it's used |
|---|---|
| `POST /v1/fuse` | Doctor opens a patient. Calls all four services, returns the tier. |
| `POST /v1/fuse/manual` | You supply scores directly. **Use this for your paper** — reproducible, no network. |
| `POST /v1/physio/tick` | C1 posts here every minute. Smoothed, other streams reused from cache. |
| `GET /v1/patients/{mrn}/state` | Last computed state without re-calling anything. |
| `GET /health` | Config, weights, references. The app's Settings screen should render this. |

---

## Part 10 · When things go wrong

| What you see | Meaning | Fix |
|---|---|---|
| `Cannot reach the fusion service` | Not running, or wrong port | Terminal 2: `uvicorn app:app --port 7861` |
| `no endpoint configured` | That component's URL is blank | Expected for C2. For others, set the env var |
| `timeout after 8s` | Space asleep or slow | Normal on first call. Correct behaviour — it becomes "missing", not zero |
| `no numeric score field in [...]` | Teammate uses a field name I didn't anticipate | Add it to the list in `clients.py` → `to_reading()` |
| `no reference distribution` warning | Missing a `reference/*.json` | Part 7. Service still runs, comparison isn't valid |
| `drift: true` | Scores no longer match the reference | Teammate probably redeployed. Ask, then rebuild that reference |
| `ModuleNotFoundError` | venv not active in this terminal | Reactivate. Happens constantly with three terminals open |

---

## Part 11 · What's still missing — say this openly in a review

Known gaps are fine. Hidden ones aren't.

1. **State is in memory.** Restart the service and every cached reading is gone. Fine for the pilot, must become Postgres before NHSL.
2. **The reference distributions are placeholders** until Part 7 is done.
3. **The weights are specified, not learned.** Justified from published AUROCs, but not yet validated against real outcomes. Once you have ~150 labelled NHSL cases, fit learned weights and report this version as the baseline it beat.
4. **No conformal prediction yet.** The design calls for a tier *set* at 90% coverage; right now you get a point tier.
5. **No RAG layer.** Separate build.
6. **Physio ticks reuse cached readings** for the other streams rather than re-fetching. Deliberate — 1,440 fan-outs per patient per day would hammer your teammates' Spaces — but it means notes updated between full fuses aren't picked up until the next `/v1/fuse`.

---

## Do this now

1. Run the seven tests, screenshot the output.
2. Message your teammates today with two asks: **(a)** add `confidence`, `coverage`, `captured_at`, `model_version` to their responses; **(b)** send their held-out score vectors so you can build real reference distributions. Both are small for them and blocking for you.
3. Sort out the `user_id` ↔ `mrn` mapping between the two apps. Everything joins on that.
