# Weighted Fusion — design and justification

**Component 4 · R26-DS-012** · reference implementation in `fusion.py`

---

## 1 · The operational facts the design has to accommodate

You described four streams that behave completely differently in time. That asymmetry, not the model architecture, is what determines the weighting mechanism.

| Stream | Update cadence | Nature | Consequence for weighting |
|---|---|---|---|
| **C1 physiological** | every **1 minute** | acute autonomic state | Freshest evidence, but goes stale fast — a reading from three hours ago says nothing about now |
| **C2 behavioural** | n/a | withheld | Did not clear its permutation null (AUROC 0.5205 vs null 0.4991, *p* = 0.255) → weight **0** |
| **C3 clinical notes** | daily → **monthly** | expert clinical judgement | Highest quality, lowest frequency; a fresh note should dominate, a six-month-old note should not |
| **C4 demographic (yours)** | **once**, at first login | population prior | Never changes. Always "available", never informative about change |

A single fixed weight vector cannot serve all four, because **the same modality deserves a different weight at different moments**. A physiological score is the most informative thing in the system sixty seconds after it arrives and nearly worthless four hours later. A clinical note written yesterday should outrank a minute-old heart-rate anomaly; the same note six months later should not. Fixed weights are blind to this entirely.

Hence:

```
w_m(t)  =  ω_m  ·  ρ_m(Δt)  ·  c_m           α = w / Σw   over available streams
S(t)    =  Σ_m  α_m · p_m
```

Three factors, each answering a distinct question: **how good is this stream in general** (ω), **how current is this particular reading** (ρ), and **how much does this stream trust its own output right now** (c).

---

## 2 · Why not the learned softmax gate from the article

The article learns a scalar per modality through a small MLP and normalises with softmax, trained end-to-end on the task loss. That is the right answer **when you have labels**. You will have roughly 60–200 labelled NHSL fusion instances by the end of the pilot, imbalanced. A learned gate fitted on that will memorise the pilot cohort and generalise to nobody.

So the design keeps the article's *structure* — a normalised weight per modality, computed per patient per moment, with masked renormalisation over available streams — and replaces the *learned* scorer with a specified one whose every constant is externally justified. Two properties from the article are retained exactly and matter a great deal:

- **Masked softmax over missing modalities.** A missing stream is excluded from the normalisation rather than treated as zero risk. This is the single most important idea in the article for your setting, because C1 is missing whenever the strap is off and C2 is missing permanently.
- **Interpretability of α.** The article notes you can inspect the weights. Here that is not a nice-to-have — the clinician app's `FusionBar` renders α directly, so the weight vector *is* the explanation the psychiatrist sees.

The upgrade path is written into the code: when NHSL labels reach ~150, swap `base_weights()` for a constrained logistic regression fitted on those labels, and report the specified-weight version as the baseline it beat. That progression is itself a result.

---

## 3 · ω — base weight from informativeness above chance

```
ω_m  ∝  max(AUROC_m − 0.5, 0)      and  ω_m := 0  if component m fails its permutation null
```

**Why AUROC − 0.5.** It is the amount by which a component beats coin-flipping. A component at 0.74 carries roughly twice the discriminative signal of one at 0.62, and this formula says so without a free parameter to argue about. The alternative — picking weights by intuition, as the current app does with `0.25/0.20/0.15/0.40` — cannot be defended in a viva. "Why 0.40?" has no answer. "Proportional to measured discriminative power above chance" does.

**Why deployment-realistic AUROC, not best-case.** The physiological component reports AUROC **0.9919** on WESAD. Using that number would give it ~70% of the composite. But WESAD is laboratory-induced acute stress under LOSO evaluation, and the same model on real-world data (AffectiveROAD) fell to **0.5688**, recovering to **0.6191** only after per-drive recalibration. Your deployment is ambulatory ward monitoring, which resembles AffectiveROAD far more than WESAD. **Using the lab number would let a component that barely works in the field dominate a clinical decision aid.** The rule is: use the AUROC from the evaluation that most resembles deployment, and state which one you used.

**Why zero for behavioural.** Not an aesthetic choice. It is a **pre-registered exclusion rule**: any component that does not exceed its own permutation null contributes ω = 0. The behavioural component scored 0.5205 against a null of 0.4991, *p* = 0.255. Applying a stated rule uniformly is defensible; dropping one component because it underperformed is not. Write the rule into the methods section *before* the results section, exactly as your paper already does for the behavioural null.

**Resulting base weights:**

| Component | AUROC used | Source of that number | Clears null | ω |
|---|---|---|---|---|
| C1 physiological | 0.6191 | AffectiveROAD, recalibrated | yes | **0.230** |
| C2 behavioural | 0.5205 | GLOBEM held-out | **no** | **0.000** |
| C3 clinical notes | 0.7380 | MIMIC-IV 5-shot held-out | yes | **0.460** |
| C4 demographic | 0.6600 | *your notebook's test AUROC* | yes | **0.309** |

⚠️ Replace the C4 figure with the actual test AUROC your notebook produces. If your permutation *p* ≥ .05, set `CLEARS_PERMUTATION_NULL["c4_demographic"] = False` and report that your own component was excluded by your own rule. That would be an uncomfortable but genuinely strong result.

---

## 4 · ρ — recency, matched to each stream's cadence

```
ρ_m(Δt) = 2^(−Δt / H_m)          H = half-life;  H = ∞ (ρ = 1) for a prior
```

| Component | Half-life | Reason |
|---|---|---|
| C1 physiological | **30 min** | Autonomic arousal resolves over minutes to hours. At 30 min weight halves; at 3 h it is ~1.5% — which is correct, because a heart-rate anomaly from this morning tells you nothing about this afternoon. |
| C3 clinical notes | **30 days** | Matches the temporal-weighting principle already inside TC-WPN, which discounts older notes when forming prototypes. Fusion should not contradict the component's own assumption about how fast clinical documentation ages. |
| C4 demographic | **no decay** | A prior does not become *wrong* with time — it was never a measurement of the current moment. Age drifts, but not on a timescale that matters here. |
| C2 behavioural | 24 h | Defined but unused (ω = 0). |

This is what makes the fusion respond correctly to your stated cadences. Worked examples from `fusion.py`, same underlying scores throughout:

| Scenario | α physio | α notes | α demo | Reading |
|---|---|---|---|---|
| All fresh | 0.217 | 0.484 | 0.300 | Notes lead, as their discriminative power warrants |
| Strap off 3 h | **0.004** | 0.646 | 0.350 | Physiology correctly evaporates; notes take over |
| Note 3 months old | **0.503** | **0.147** | 0.350 | Reverses — live physiology now outranks stale documentation |

No hand-tuning produced that reversal. It falls out of the half-lives.

---

## 5 · c — reliability scaling

```
c_m = 0.5 + 0.5 · confidence_m · coverage_m          bounded in [0.5, 1.0]
```

Every component already publishes, or can cheaply publish, the two inputs: TC-WPN returns `confidence` and `entropy`; your DCAR service returns `confidence` (1 − normalised entropy over the four severity bands) and `coverage` (fraction of the five fields supplied); C1 can use wear-time fraction as coverage and forecast-interval width as confidence.

**Why bounded below at 0.5 rather than allowed to reach zero.** A stream that is uncertain is still evidence. Letting c → 0 would let one component's bad self-assessment silently delete it from the composite while the UI still shows it as present. Halving is a meaningful penalty; erasure is a lie.

---

## 6 · The prior cap — why your own component is deliberately limited

`PRIOR_CAP = 0.35`.

The AUROC formula gives C4 a base weight of 0.309, which after renormalisation when physiology is stale would rise well above that. It is capped, and the excess is redistributed across the time-varying streams. The reason:

> The clinical purpose of this system is to detect **change** — acute escalation. The demographic score is computed once and is mathematically constant for that patient forever. A constant cannot contribute any information about escalation. If it were allowed to dominate the composite, two patients with identical demographics would receive similar tiers regardless of what their physiology or their clinician's notes said, and the system would degrade into a demographic stereotype with a live-data veneer.

That is also a fairness argument, not only an accuracy one: the inputs are gender, age, education, smoking and drinking. Capping the influence of a stereotype-shaped prior on a clinical decision is the right default, and it is worth a sentence in the ethics section.

**Related guard.** If the *only* available stream is the demographic prior — which is exactly the situation at first login, before the strap is fitted and before any note exists — the service returns **no tier at all**, with a stated reason. It does not return a Low tier. Emitting "Low risk" for a newly admitted psychiatric patient on the basis of their age and education alone would be an actively dangerous output.

---

## 7 · Handling the 1-per-minute stream: smoothing and hysteresis

Two mechanisms in `LiveFusion`, both about clinical usability rather than accuracy:

- **EWMA smoothing** (α = 0.20, ≈15-minute effective window) on the physiological score. A single anxious minute is not an escalation; a rising trend is. Fusing raw minute-level scores would produce a composite that jitters continuously.
- **Tier hysteresis** — a tier change is emitted only after **3 consecutive** readings agree. In the test sequence, a six-minute spike pushes the composite to 0.663, just across the High boundary, for exactly one reading. Without hysteresis the clinician's screen flips to RED and back within two minutes. With it, nothing is emitted. Alert fatigue is the most common reason ward staff disable a monitoring tool, and a badge that flickers is the fastest route to it.

---

## 8 · What must still be verified empirically

The mechanism above is *specified*, which means it is defensible without labels — but it is not yet *validated*. Once the NHSL pilot has labels, report:

1. **Fusion vs. the best single modality.** The most important comparison in the component. If the composite does not beat C3 alone, say so plainly.
2. **Specified weights (this design) vs. learned weights** (constrained logistic regression) vs. **fixed equal weights** vs. **the app's current 0.25/0.20/0.15/0.40**.
3. **Ablation of each factor:** ω alone; ω·ρ; ω·c; full ω·ρ·c. This isolates whether recency and reliability weighting actually earn their complexity, which is the novelty claim.
4. **Missing-modality degradation curve** at 3, 2 and 1 available streams.
5. **Calibration** of the composite: Brier, ECE, reliability diagram — and re-fit the band edges (0.33/0.66) on validation rather than keeping the defaults.
6. **Sensitivity to the half-lives.** Sweep H_physio over {10, 30, 60, 120} min and H_note over {7, 30, 90} days and report how much the tier distribution moves. If results are wildly sensitive, the constants need empirical grounding rather than reasoning.
7. **A permutation null** on the composite, matching the protocol the rest of the project uses.

Pre-register that list before you look at any of the numbers, and say in the paper that you did.
