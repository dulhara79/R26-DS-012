// Equal-layout homepage summaries of the R26-DS-012 components.
// Equal visual treatment does not imply equal validation or active fusion eligibility.
export const homeEvidenceHighlights = {
  C1: {
    title: "Wearable physiology",
    description: "Personalized anomaly detection and short-horizon forecasting from wearable measurements.",
    imageAlt: "Wearable physiological sensing research illustration",
    dataset: "WESAD · AffectiveROAD",
    status: "Laboratory and external evaluation; clinical validation pending",
    statusType: "limited",
  },
  C2: {
    title: "Behavioural graph learning",
    description: "Spatio-temporal graph analysis of longitudinal passive smartphone sensing.",
    imageAlt: "Smartphone-based longitudinal behavioural graph research illustration",
    dataset: "GLOBEM · 28-day sensing",
    status: "Held-out AUROC 0.5205 · fusion weight 0.0",
    statusType: "excluded",
  },
  C3: {
    title: "Clinical NLP · TC-WPN",
    description: "Few-shot anxiety-related classification of clinical notes using language-model representations.",
    imageAlt: "Clinical notes represented as few-shot learning prototypes",
    dataset: "MIMIC clinical notes",
    status: "Independent leakage-audited verification needed",
    statusType: "limited",
  },
  C4: {
    title: "Multimodal fusion & CARE-AnxRAG",
    description: "Reliability-aware integration of eligible signals and contextual inputs with source-grounded evidence retrieval.",
    imageAlt: "Multimodal fusion and evidence retrieval research illustration",
    dataset: "Contextual prior · fusion · CARE-AnxRAG",
    status: "Research decision-support prototype; not clinically validated",
    statusType: "limited",
  },
};