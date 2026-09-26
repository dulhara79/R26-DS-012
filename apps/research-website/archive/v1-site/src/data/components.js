export const components = [
  {
    id: "C1",
    owner: "Sendanayake H.D.",
    title: "Personalized Physiological Forecasting",
    question:
      "Can individualized physiological deviation provide useful short-horizon evidence of escalation?",
    method:
      "Self-supervised physiological anomaly detection using an LSTM autoencoder, followed by short-horizon physiological forecasting.",
    data: "WESAD · AffectiveROAD · PPG-DaLiA · EmoWear",
    status: "Active research signal",
    outputLabel: "Physiological signal / forecast",
    researchState: "validated-research-stream",
    provenance: "R26-DS-012 README · Component 1",
  },
  {
    id: "C2",
    owner: "Layathma B.M.A.S.",
    title: "Behavioural Vulnerability Mapping",
    question:
      "Do temporal behavioural graphs generalize beyond simpler passive-sensing baselines?",
    method:
      "GATv2 over 28-day GLOBEM behavioural graphs with participant-grouped, cross-cohort evaluation.",
    data: "GLOBEM · INS-W_1–4",
    status: "Experimental · excluded from active fusion",
    fusionState: "excluded",
    activeFusionWeight: 0,
    outputLabel: "Behavioural research signal",
    evidence:
      "Held-out AUROC 0.5205 · 95% CI 0.485–0.560 · permutation null 0.4991 · p=0.255",
    limitation:
      "The final held-out GATv2 result was not distinguishable from chance and did not outperform simpler baselines.",
    provenance: "R26-DS-012 README · Component 2 final v8 evidence",
  },
  {
    id: "C3",
    owner: "Kaushalya I.G.D.",
    title: "Clinical NLP / TC-WPN",
    question:
      "Can few-shot clinical NLP learn from small support sets while controlling patient leakage and label contamination?",
    method:
      "Bio_ClinicalBERT → 256-dimensional projection → Temporal-Consistency Weighted Prototypical Network.",
    data: "MIMIC-IV · MIMIC-III transfer",
    status: "Repository-supported held-out AUROC ≈0.738",
    outputLabel: "Clinical NLP signal",
    claimBoundary:
      "TC-WPN contributes a Clinical NLP signal; it is not overall patient risk.",
    provenance: "R26-DS-012 README · Component 3 paper-aligned benchmark",
  },
  {
    id: "C4",
    owner: "Seneviratne K.A.U.A.",
    title: "Reliability-Aware Fusion & Evidence Support",
    question:
      "How should heterogeneous evidence be combined when modalities differ in recency, reliability and validation quality?",
    method:
      "Contextual prior + eligible component outputs → reliability-weighted fusion → evidence-aware CARE-AnxRAG support.",
    data: "Component outputs · contextual variables · evidence corpus",
    status: "Research decision-support layer",
    outputLabel: "Current multimodal research assessment",
    missingEvidenceMeaning:
      "Unavailable evidence is not low anxiety or zero risk; the system can return insufficient evidence.",
    provenance: "R26-DS-012 README · Component 4",
  },
];
