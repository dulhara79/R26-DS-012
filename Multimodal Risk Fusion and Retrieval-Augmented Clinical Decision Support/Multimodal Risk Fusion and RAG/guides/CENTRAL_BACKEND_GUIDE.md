# Central Backend — how to run it, and how it maps to your diagram

**Component 4 · R26-DS-012** · validated: **80 tests, 0 failures** (updated after merging the service-contracts document — see `TODAY_PLAN.md` for the changelog)

---

## First: a naming correction you must apply to the diagram

Your diagram's columns are labelled `C3 Contextual` and `C4 TC-WPN`. Your paper
defines Component 3 as the few-shot clinical **text** model (TC-WPN) and Component 4
as **your** fusion + contextual work. Those are opposite.

**Fix the diagram, not the code.** Relabel:

| Diagram column now | Should read |
|---|---|
| `C3 Contextual` | **C4 Contextual (DCAR)** — demographics + GAD-7, yours |
| `C4 TC-WPN` | **C3 Clinical NLP (TC-WPN)** — clinical notes, Dulhara's |

The *arrows* in your diagram are all correct — demographics enter from the patient
app, notes from the clinician app. Only the two column headers are swapped. Left
unfixed, your paper and your codebase disagree, which is a bad thing to discover
at submission.

The code uses `c3_clinical_nlp` and `c4_demographic` throughout, matching the paper.

---

## What got built

Seven files. The backend owns ingestion and identity; the fusion service only fuses.

| File | Role |
|---|---|
| `main.py` | FastAPI app — every step of the diagram |
| `db_models.py` | PostgreSQL/SQLite schema (subjects, aliases, readings, results, audit) |
| `identity.py` | MRN hashing, pairing codes — the patient-separation machinery |
| `gate.py` | Step 28: the fusion gate |
| `modality_clients.py` | Adapters to the four component Spaces |
| `fusion_client.py` | Step 29: calls the Fusion Service (or runs it in-process) |
| `test_backend.py` | 80 assertions covering the whole flow |

---

## Diagram → endpoint map

| Steps | Diagram | Endpoint |
|---|---|---|
| 1–4 | enrol patient, enter MRN → subject_id + pairing_code | `POST /v1/subjects` |
| 5–9 | patient enters pairing code → alias linked | `POST /v1/subjects/pair` |
| 10–13 | chest-strap window (60s) → C1 → store | `POST /v1/ingest/physiological` |
| 14–17 | behavioural aggregates → excluded | `POST /v1/ingest/behavioural` |
| 18–21 | contextual + GAD-7 → **your DCAR** → store | `POST /v1/ingest/contextual` |
| 22–26 | clinician writes note → TC-WPN → store | `POST /v1/clinical-notes` |
| 27–31 | select latest per modality → gate → fuse → persist | `POST /v1/fusion/run` |
| 32–33 | patient view: composite + band | `GET /v1/patients/{id}/risk` |
| 34–35 | clinician view: everything | `GET /v1/doctor/patients/{id}/timeline` |

---

## Run it in 5 minutes

```bash
cd central_backend
pip install -r requirements.txt

# 1. Set the pepper — the backend REFUSES to hash MRNs without it
python -c "import secrets; print('MRN_PEPPER=' + secrets.token_urlsafe(48))"
# copy that line into a file called .env  (rename env.example.txt to .env first)

# 2. Run the validation suite — needs no network, no database setup
python test_backend.py

# 3. Start it
uvicorn main:app --reload --port 8000
```

Open **http://127.0.0.1:8000/docs**.

> `test_backend.py` stubs the four Spaces, so it proves the *backend* logic without
> depending on anyone's Space being awake. Run it before every commit.

---

## Try the whole flow by hand

```bash
# enrol (clinician)
curl -X POST localhost:8000/v1/subjects -H 'Content-Type: application/json' \
  -d '{"mrn":"NHSL-2026-0142","enrolled_by":"dr.perera"}'
#  -> {"subject_id":"f52b29fc-...","pairing_code":"V8YV-HP33", ...}

# pair (patient types the code into their app)
curl -X POST localhost:8000/v1/subjects/pair -H 'Content-Type: application/json' \
  -d '{"pairing_code":"V8YV-HP33","app_user_id":"phone-aaa"}'

# demographics + GAD-7 (patient app, once)
curl -X POST localhost:8000/v1/ingest/contextual -H 'Content-Type: application/json' \
  -d '{"app_user_id":"phone-aaa","gender":"female","age":21,
       "edu":"bachelor'\''s degree","smoke":"never smokes","drink":"never drinks",
       "gad7_items":[2,2,1,2,1,1,2]}'

# fuse now — with only demographics you get NO tier, on purpose
curl -X POST localhost:8000/v1/fusion/run -H 'Content-Type: application/json' \
  -d '{"subject_id":"PASTE_SUBJECT_ID"}'

# add physiology + a note, then fuse again -> a real tier appears
```

---

## The five design decisions worth defending in a viva

### 1. The raw MRN is never stored

`identity.hash_mrn()` computes `HMAC-SHA256(MRN, pepper)`. The pepper lives in the
environment, **not** the database. A stolen database cannot be reversed to patient
identifiers without it.

HMAC rather than plain `sha256(mrn + pepper)` because HMAC is the construction
designed for keyed hashing and resists length-extension. And peppered at all
because an MRN has low entropy — an unsalted hash of it falls to a dictionary
attack in seconds.

The test suite scans **every row of every table** for the literal string
`NHSL-2026-0142` and asserts it is absent. That test is the evidence for your
ethics submission.

### 2. Patient separation is enforced by construction

Every query is scoped by `subject_id`. The pairing flow guarantees one subject_id
per patient across both apps. A phone already paired to one patient is **refused**
when someone tries to pair it to another — silently re-pointing it would cross two
clinical records.

Section 7 of the test suite is the separation demo your diagram asks for: two
patients, disjoint readings, verified that P001's physiology never appears in
P002's timeline.

### 3. The gate refuses to guess

Step 28 in code. Four conditions, each rejecting a specific failure:

- **freshness** — a physiological reading over 6h old is dropped (tested: 12h old → rejected)
- **≥2 usable modalities** — one stream is not fusion
- **≥1 time-varying modality** — the day-one guard. Demographics alone → **no tier**, band GREY, stated reason. Never "Low risk"
- **semantics** — status must be `ok` and the score finite

### 4. A failed component is missing, never zero

If C1 times out, the reading is stored with `status='error'` and `raw_score=NULL`.
The gate drops it, the weights renormalise, and the composite is computed from
what remains. Scoring a sleeping Space as `0.0` would read as *"this patient has
no physiological risk"* — a dangerous lie. Tested in section 9.

### 5. Two egress views, one source of truth

Both read the same `fusion_results` row.

- **Patient** gets `composite`, `band`, `updated_at`. Nothing else.
- **Clinician** gets per-modality scores, freshness, status flags, weights, contributions, the gate decision, and trend history.

The patient view deliberately withholds per-modality detail: a patient reading
*"your clinical notes score is 0.81"* with no clinician present is a harm, not
transparency. The test asserts the raw clinical note text never appears in either
response.

---

## What the tests actually verify

```
 1 · Health and configuration                    3 checks
 2 · Enrolment and pairing                       9 checks
 3 · MRN never stored in plaintext               2 checks
 4 · Day-one guard                               5 checks
 5 · Behavioural stored but excluded             4 checks
 6 · Full fusion                                 7 checks
 7 · PATIENT SEPARATION                          7 checks
 8 · Staleness — tightened freshness windows     7 checks
 8b· Effective-weight floor (regression test)    5 checks
 9 · New status vocabulary                       6 checks
10 · Component failure is missing not zero       4 checks
11 · Egress — two views                          9 checks
12 · Audit trail                                 5 checks
13 · Rejections and edge cases                   6 checks
                                        80 passed, 0 failed
```

Sections 8 and 8b are new — merged in from the service-contracts document review.
Full changelog in `TODAY_PLAN.md`.

---

## Moving to PostgreSQL

SQLite for dev, Postgres for NHSL. One line:

```bash
DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/r26ds012
```

The ORM is identical. Two caveats:

1. `Base.metadata.create_all()` is fine for the pilot but you should adopt
   **Alembic migrations** before real patient data — you will need to change the
   schema, and `create_all` cannot alter existing tables.
2. SQLite drops timezone info on round-trip; Postgres does not. The code
   defensively re-attaches UTC (`if captured_at.tzinfo is None`). Keep that.

---

## Known gaps — state these openly

1. **No authentication beyond a shared bearer token.** Real deployment needs
   per-clinician identity so the audit log records *who*, not just *what*.
2. **No conformal prediction yet.** Tier is a point estimate; the design calls for
   a tier *set* at 90% coverage.
3. **No RAG layer.**
4. **Fusion is not auto-triggered.** The diagram shows fusion firing on the
   ingestion event; right now you call `/v1/fusion/run`. Wiring the trigger into
   each ingest endpoint is ~3 lines, but do it deliberately — a fusion on every
   60-second physiological tick is 1,440 rows per patient per day.
5. **Reference distributions are still placeholders** until teammates send
   held-out score vectors.
6. **Withdrawal is modelled (`subject.status`) but there is no delete path.**
   Ethics will ask about data withdrawal — decide whether it is soft-flag or hard-delete.

---

## Next

The obvious next piece is the **auto-trigger** (gap 4) plus a debounce so
physiological ticks don't fuse 1,440 times a day. After that, the RAG layer.
