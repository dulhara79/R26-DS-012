# Today's plan

**Component 4 · R26-DS-012** · everything below is either done-and-tested right now, or a message you send in the next ten minutes.

---

## 1 · Send this to the team right now (before anything else)

Two things only your team can resolve — paste this into your group chat verbatim:

> **Two things need a team decision before we freeze the service contract:**
>
> **1. Component numbering is inconsistent across our documents.** The paper and my codebase use: C3 = clinical NLP (Dulhara/TC-WPN), C4 = demographic/contextual (me/DCAR). Kaushalya's service-contracts doc uses the opposite: C3 = intervention engine (Kaushalya), C4 = clinical NLP (Dulhara). We need ONE convention, written down, before any more contract work happens.
>
> **2. What does my component actually do?** Kaushalya's doc describes "C4" as Gradient Boosting → 4-tier label → KNN-CBR intervention recommendation. That's not what I've built — I've built the demographic/GAD-7 risk model (DCAR) feeding a reliability-weighted fusion, with RAG-based decision support downstream of fusion. If Kaushalya's doc describes the OLD design, say so explicitly so we don't freeze a contract for something nobody is building. If it's not the old design, we have two different components claiming the same slot and need to reconcile before I go further.
>
> Proposing: paper's numbering wins (C3=NLP, C4=demographic/fusion/RAG — mine), since it's what's actually implemented and tested. Intervention/recommendation logic, if still wanted, sits downstream of fusion as a separate stage — doesn't change anyone's model, just where in the pipeline it plugs in.

Do not proceed past this today without an answer on point 2. Point 1 is a naming fix; point 2 changes what gets built.

---

## 2 · What I merged into the code today — already done, already tested

I read Kaushalya's service-contracts document end to end and merged the genuinely better parts into what we'd already built, fixed one real bug it exposed, and re-ran everything.

**Test count: 66 → 80 passing, 0 failing.** Every new check is a new behaviour, not a relaxed one.

| # | Change | Source | Where |
|---|---|---|---|
| 1 | Full status vocabulary: `ok / warming_up / insufficient_data / poor_signal / no_support_set / not_validated / error` | adopted from the doc | `modality_clients.py` |
| 2 | C1 `warming_up` handling — a self-supervised model's score is noise before its personal baseline converges | adopted from the doc | `modality_clients.py`, tested §9 |
| 3 | C3 `no_support_set` handling — K=0 means meta-trained prototypes only, must be flagged | adopted from the doc | `modality_clients.py`, tested §9 |
| 4 | C3 now prefers `calibrated_probability` over `risk_score` over `score`, per the doc's explicit rule | adopted from the doc | `modality_clients.py` |
| 5 | Freshness windows tightened: C1 6h→**15min**, C2 2d→**7d**, C3 180d→**90d** | adopted from the doc | `gate.py`, tested §8 |
| 6 | **Effective-weight floor** — a modality now has to clear both the age cutoff AND a post-decay weight floor (5%) to count. This is a genuine bug fix: the old 6h/30min mismatch let a 3-hour-old physiological reading count as "fresh" while contributing ~1.6% of the composite. Two modalities where one is a rounding error was one modality wearing a disguise. | fixed, neither doc had this | `gate.py`, tested §8b — includes a regression test that reproduces the old bug and proves the floor catches it even if someone loosens a window again |
| 7 | C1 client now tries the frozen target contract (`POST /predict` with window+features) first, falls back to the legacy live endpoint (`GET /predict/{user_id}`) on 404 | reconciled: doc's target vs. what's actually live | `modality_clients.py` |
| 8 | Ingestion endpoint extended to carry `window_start/window_end/sampling_hz/features` (all optional, backward compatible) | adopted from the doc | `main.py` |
| 9 | Your DCAR Space now emits the full common envelope: `subject_id, modality, status, captured_at, computed_at, latency_ms` | adopted from the doc | `hf_space/app.py`, smoke-tested |
| 10 | DCAR now returns `status: "poor_signal"` and `score: null` (not a number) when coverage is below 60% | adopted from the doc's "don't emit a number built on nothing" principle | `hf_space/app.py`, smoke-tested |
| 11 | AUROC-derived weights, permutation-null exclusion rule, percentile harmonisation, continuous recency decay | **kept ours** — the doc's weights trace to a constant in a Dart file, ours trace to measurements | `fusion.py`, `harmonise.py` — unchanged |
| 12 | Identity: HMAC-peppered MRN hashing, pairing codes, patient separation | **kept ours** — the doc only asserts this should exist, we have it built and tested | `identity.py`, `main.py` — unchanged |
| 13 | `fusion_service/clients.py` and `clients_real.py` marked superseded (backend now owns Space-calling, per the diagram) | reconciled | docstring headers only, not deleted |

**What I deliberately did NOT adopt**, and why:

- **The doc's fixed weights (0.25/0.20/0.15/0.40).** No traceable justification. Ours are AUROC-derived with a stated exclusion rule. Keep ours.
- **The 4-band alert scheme (GREEN/AMBER/RED/DARK RED).** The paper specifies a 3-tier clinical output (Low/Medium/High). Changing this unilaterally is exactly the kind of thing that needs a team decision, not a code change — see §1.
- **Hosting the DCAR Space as Docker on HF without checking first.** The doc's claim about paid-plan requirements checked out against HF's own docs — see §3 below.

---

## 3 · Hosting — verified today, act on it now

Checked against Hugging Face's own documentation (verified in the previous message): creating a new Space that runs on compute (Gradio or Docker) requires a paid plan, and Static Spaces are the only free option for new compute Spaces. Free cpu-basic Spaces also sleep after 48 hours of inactivity, and that sleep timer cannot be changed on the free tier — confirmed by a Hugging Face forum thread reporting exactly this restriction.

**Do this today, first thing:**
1. Try creating your DCAR Space exactly as in the Hugging Face guide (Part A3). If it lets you create a Docker Space for free, you're done — some accounts still can.
2. If it demands payment: don't pay yet. Ask whether Dulhara's existing account (which already hosts `dulharakaushalya-tc-wpn-demo.hf.space` and predates this restriction) can host a second Space for you.
3. If neither works: skip HF for DCAR specifically. It's a small scikit-learn model — run it as a route inside the Central Backend instead of a separate Space. One less network hop, one less thing that can sleep.

**Do NOT put the Central Backend on Hugging Face regardless of the above** — a sleeping orchestrator with a fixed, non-configurable 48-hour timeout cannot be your integration point. Options, cheapest first: Oracle Cloud Always Free (try the signup this week, not demo week), or Render's ~$7/month Starter tier for the two months around your demo.

---

## 4 · Run the validation suite yourself, right now

```bash
cd central_backend
python test_backend.py
```

You should see `80 passed, 0 failed`. If you see fewer, something in your local copy didn't apply cleanly — tell me the exact failure and I'll fix it before you build on top of it.

---

## 5 · Today's build order

1. **Send the team message (§1).** Ten minutes. Nothing below should wait for a reply, but everything below should be treated as provisional until you get one.
2. **Check HF hosting (§3).** Ten minutes.
3. **Run the test suite (§4).** Two minutes. Confirms your local state matches what's been validated here.
4. **Deploy the updated DCAR Space** — same steps as before (Hugging Face guide Part A6), just re-push `hf_space/app.py`. It now speaks the common envelope.
5. **Run the Central Backend locally** against your real DCAR Space and the mock C1/C3 (`fusion_service/mock_components.py` still works for this — the mocks don't need to know about the new contract to be useful stand-ins).
6. **Once C1 and C3's real URLs are confirmed**, point `.env` at them and re-run `/v1/fusion/run` for a live subject end to end.

Everything in this list is independent of the team's answer on §1 point 1 (naming). Nothing in step 4–6 should start before an answer on point 2 (what your component is) — if the team says the intervention-engine design is still live, some of what's described here changes.

---

## 6 · Open items, unchanged from before

Still true, still worth stating openly rather than discovering late:

- Fusion is not yet auto-triggered on ingestion events (currently a manual `POST /v1/fusion/run`).
- Reference distributions for harmonisation are placeholders until C1/C3 send held-out score vectors.
- No conformal prediction yet — tier is a point estimate.
- No RAG layer yet.
- Auth is a single shared bearer token, not per-clinician identity.
- 3-tier vs 4-band alert scheme is unresolved (§2, deliberately not changed).
