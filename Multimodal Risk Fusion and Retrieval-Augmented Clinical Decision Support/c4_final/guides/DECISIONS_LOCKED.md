# Architecture decisions — locked

**Component 4 · R26-DS-012** · decided today, verified against the running codebase, zero code changes required.

---

## The three decisions

| Question | Decision |
|---|---|
| What is Component 4? | **DCAR + Fusion + RAG** — the demographic model, reliability-weighted fusion, and retrieval-augmented decision support. Not the GBDT/KNN-CBR intervention engine. |
| Component numbering | **Paper's convention.** C3 = clinical NLP (Dulhara/TC-WPN). C4 = demographic/contextual + fusion + RAG (you/DCAR). |
| Fusion output | **3-tier**: Low / Medium / High. |

All three were already true of the running code before this decision was recorded — verified just now:

- `fusion.py` line 67: `BANDS = [(0.33, "Low"), (0.66, "Medium"), (1.01, "High")]`
- Every file in `central_backend/` and `fusion_service/` already keys on `c3_clinical_nlp` and `c4_demographic`
- `test_backend.py`: 80/80 passing, unchanged

So today's decision didn't require touching the implementation. It closes the gap between what the code does and what the team's documents said — which was the actual risk, not the code.

**One thing worth noting for free:** your 3-tier answer specifically included "map to 4 colors for the UI only" as an acceptable variant, and that already exists — `LiveFusion.to_wire()` maps `Low/Medium/High/None` to `GREEN/AMBER/RED/GREY` for display. If the team later wants a fourth *tier* rather than a fourth *color* (a real Low/Medium/High/Critical distinction with its own threshold), that's still an open question — today's decision was about color count, not adding a decision boundary. Flag this distinction if it comes up again.

---

## Send this to the team — it's an announcement now, not a question

> **Component 4 architecture is locked, effective today:**
>
> - **C4 (mine) = DCAR demographic model → reliability-weighted fusion → RAG decision support.** This is what's built and tested (80 passing checks). It is not the GBDT/KNN-CBR intervention engine described in the service-contracts doc — if that design is still wanted by anyone, it needs its own slot in the architecture, separately scoped, because C4 is already spoken for.
> - **Numbering follows the paper:** C3 = clinical NLP (Dulhara), C4 = demographic/fusion (me). Kaushalya's service-contracts doc has these swapped — please treat that doc's C3/C4 labels as needing a rename pass, not as the source of truth for numbering.
> - **Fusion output is 3-tier** (Low/Medium/High), mapped to 4 display colors (GREEN/AMBER/RED/GREY) in the UI layer only. Not a 4-tier decision boundary.
>
> If anyone was building against Kaushalya's numbering or the intervention-engine framing, flag it now — better to reconcile this week than at integration.

---

## What's still actually open

Locking these three doesn't close everything from `TODAY_PLAN.md`. Still pending:

1. **Hosting check** — did the DCAR Docker Space create successfully, or did it hit the paid-plan wall? (§3 of `TODAY_PLAN.md`)
2. **Whether the intervention-engine design has a home elsewhere** — the team message above raises this but doesn't resolve it. If Kaushalya's GBDT/KNN-CBR work is real and ongoing, it needs an explicit position in the architecture (most naturally: downstream of your fusion output, consuming the composite + tier the way the original sequence diagram's "C3 Intervention" box did — just not numbered C3 or C4 anymore, since both are taken).
3. Auto-triggering fusion on ingestion events, real reference distributions from C1/C3, conformal prediction, RAG build — all still on the list, unchanged.

Send the announcement, resolve the hosting check, and you're clear to keep building on exactly what's already passing.
