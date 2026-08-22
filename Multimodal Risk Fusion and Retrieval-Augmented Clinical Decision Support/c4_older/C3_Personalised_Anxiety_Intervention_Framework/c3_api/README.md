# C3 Phase 3 — FastAPI Backend

**Project:** R26-DS-012 — Continuously Learning Personalised Anxiety Intervention Framework
**Student:** Seneviratne K.A.U.A. (IT22093950)

REST API wrapping the XGBoost + SHAP + FAISS pipeline built in Phases 2A–2C. 9 endpoints, JWT auth, FAISS graceful fallback, Railway-ready.

---

## Quick start (VS Code, local)

### 1. Clone and set up virtual env

```bash
cd c3_api
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Drop artifacts into `artifacts/`

Unzip `C3_FIXED_ARTIFACTS.zip` (and optionally `C3_VectorDB_Artifacts.zip`) into the `artifacts/` folder. Final layout:

```
artifacts/
├── xgboost_smotenc.pkl              ← REQUIRED
├── probability_calibrator.pkl       ← REQUIRED
├── conformal_predictor.pkl          ← REQUIRED
├── seed_case_base.csv               ← REQUIRED
├── xai_shap_explainer.pkl           ← optional (SHAP explain endpoint)
├── xai_counterfactual_engine.pkl    ← optional
├── xai_nl_template.json             ← optional
├── xai_lime_results.json            ← optional
├── feature_cols.json                ← optional (falls back to hardcoded order)
├── faiss_rawspace.index             ← optional (Phase 2C — falls back to euclidean)
└── faiss_metadata.json              ← optional (paired with faiss_rawspace.index)
```

If a required file is missing, startup fails fast with a clear message. Optional files trigger a warning and a fallback path.

### 3. Set environment variables

```bash
cp .env.example .env
# Edit .env and paste in a generated secret:
python -c "import secrets; print(secrets.token_hex(32))"
```

Or export directly:

```bash
export C3_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
```

### 4. Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

### 5. Smoke test (in another terminal)

```bash
python smoke_test.py
```

Expected: `Passed: 11/11`.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health`                  | Reports which artifacts loaded |
| POST | `/v3/register`             | Create user + return JWT |
| POST | `/v3/gad7/submit`          | Build 13-feature vector from GAD-7 + C1/C2/C4 |
| POST | `/v3/risk/classify`        | XGBoost → calibrated proba + conformal set |
| POST | `/v3/risk/explain`         | SHAP values + NL summary + counterfactual |
| POST | `/v3/recommend`            | FAISS retrieval of k similar cases |
| POST | `/v3/session/complete`     | Record session outcome |
| POST | `/v3/reward/compute`       | Composite reward R ∈ [−1, 1] + F12 update |
| POST | `/v3/clinician/review`     | Approve / modify / reject intervention |
| POST | `/v3/intervention/assign`  | Assign approved intervention to profile |

Full request/response schemas are at `/docs`.

---

## Testing from Flutter

- **Local device on same WiFi:** point Flutter to `http://<your-laptop-ip>:8000` (find with `ipconfig` / `ifconfig`).
- **Deployed:** point Flutter to `https://<your-project>.railway.app`.

CORS is enabled permissively for development. Tighten before production.

---

## Deploying to Railway

### One-time setup

1. Install the Railway CLI: `npm i -g @railway/cli`
2. `railway login`
3. From the `c3_api/` directory: `railway init`

### Environment variables

In the Railway dashboard → Variables, add:

| Key | Value |
|---|---|
| `C3_SECRET_KEY` | output of `python -c "import secrets; print(secrets.token_hex(32))"` |
| `FIREBASE_CREDENTIALS_JSON` | (leave empty for Phase 3) |

### Deploy

```bash
railway up
```

Railway uses `Procfile` + `railway.json` automatically. Health check hits `/health` to verify all artifacts loaded.

### Important — artifact size

Railway's free tier has build-time and image-size limits. The `artifacts/` folder can be hundreds of MB. Two options:

- **Option A (simplest):** commit artifacts to the repo and let Railway build them in. Works for < 500 MB total.
- **Option B (recommended for large artifacts):** host artifacts on S3 / Google Cloud Storage / Hugging Face, and download at startup inside `model_loader.load_all()`.

For Phase 3 supervisor demo, Option A is fine.

---

## Architecture notes

### The leakage fix (critical for dissertation viva)

`risk_tier_enc` (F10) is the previous session's predicted tier. Leaving it in place leaks label information. Protection at **two layers**:

1. **`FeatureVector.risk_tier_enc`** — Pydantic `@field_validator(mode="before")` forces 0.0 on every input.
2. **`inference._vector_to_row`** — even if the validator is bypassed, the inference helper rewrites `risk_tier_enc = 0.0` before building the numpy row.

If an examiner asks "how do you prevent leakage at inference?", the answer is: belt-and-braces, two independent layers, one in the schema and one in the inference code.

### ModelRegistry singleton

All artifacts load **once** at startup via the FastAPI `lifespan` context manager. Subsequent requests hit warm, in-memory objects. Memory cost ~200 MB; cold-start latency ~8 s on Railway free tier.

### FAISS graceful fallback

If `faiss_rawspace.index` is missing (Phase 2C not yet run), `/v3/recommend` falls back to SHAP-weighted euclidean L2 on `seed_case_base.csv`. Returns `retriever_used: "euclidean_fallback"` in the response so the caller knows.

### Reward function (ties to Phase 6)

```
R = 0.35·completion + 0.30·rating_norm + 0.25·gad7_delta − 0.10·escalation_penalty
clipped to [−1, 1]
```

The returned `updated_last_reward_norm = (R + 1) / 2` maps back to [0, 1] so Flutter can feed it in as F12 on the next session. This is the mechanism that makes the system "continuously learning".

---

## What's deferred to Phase 7

- Firebase persistence (current: in-memory dicts in `routers/user.py`, `routers/session.py`)
- Password hashing (current: passwords accepted but not stored)
- Tightened CORS origins
- Rate limiting
- Email verification

The in-memory stores are sufficient for supervisor demo and Phase 6 MRT simulation.

---

## Troubleshooting

**`RuntimeError: Required artifacts missing`** — the 4 REQUIRED PKLs/CSV aren't in `artifacts/`. Unzip `C3_FIXED_ARTIFACTS.zip` into that folder.

**`ModuleNotFoundError: faiss`** — either install `faiss-cpu` (in `requirements.txt`) or let the fallback kick in. Index file missing is fine; the import must still succeed.

**Cold start timeout on Railway** — increase `healthcheckTimeout` in `railway.json` from 100 to 180 seconds.

**SHAP values all zeros** — the SHAP explainer artifact failed to load; check startup logs. Fallback uses `SHAP_WEIGHTS` from `config.py` as proxy weights.
