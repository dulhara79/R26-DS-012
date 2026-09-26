// High-level architecture. Sources: R26-DS-012 root README (component flows,
// fusion rule, privacy commitments), the Aura and ClinAnx app READMEs, and the
// C4 integration notes (DCAR inputs). Keep each statement traceable to those.

export const bigPicture = {
  eyebrow: "The big picture",
  title: "Four evidence streams, one careful decision path.",
  paragraphs: [
    "The framework treats each evidence stream as its own study, with its own data, model and validation contract. Raw modality streams are never pooled into one model: each component publishes an output and status metadata to a central backend.",
    "The backend checks eligibility before anything is combined. A component that fails its own permutation-null criterion receives zero base weight, which is why Component 2 is currently excluded. Unavailable or stale modalities are masked, never read as zero risk, and the contextual prior cannot produce a tier by itself.",
    "Reliability-weighted fusion turns the eligible outputs into a Low, Medium or High tier, or returns insufficient evidence. CARE-AnxRAG then attaches evidence that has passed relevance, quality and conflict checks. The Aura and ClinAnx apps only display the result; they never compute a score.",
  ],
};

export const stages = [
  "Evidence sources",
  "Component models",
  "Published outputs",
  "Eligibility and fusion",
  "Decision support",
];

// One lane per component, read left to right in the diagram.
export const lanes = [
  {
    id: "C1",
    slug: "wearable-forecasting",
    state: "eligible",
    source: ["ESP32-C3 chest strap", "ECG, respiration, motion, skin temperature"],
    model: ["LSTM autoencoder + forecaster", "60-second windows, 10 features"],
    output: ["Physiological signal", "Anomaly score and short-horizon forecast"],
  },
  {
    id: "C2",
    slug: "behavioural-graphs",
    state: "excluded",
    source: ["Smartphone passive sensing", "Screen, coarse location, app categories, movement, call/SMS counts"],
    model: ["GATv2 temporal graph", "28 days × 4 daily segments"],
    output: ["Behavioural research signal", "Logged only · fusion weight 0.0"],
  },
  {
    id: "C3",
    slug: "clinical-nlp",
    state: "eligible",
    source: ["Clinical notes", "Submitted through ClinAnx; MIMIC-IV in the benchmark"],
    model: ["TC-WPN", "Bio_ClinicalBERT + few-shot prototypes"],
    output: ["Clinical NLP signal", "Not overall patient risk"],
  },
  {
    id: "C4",
    slug: "fusion-and-rag",
    state: "prior",
    source: ["Contextual profile", "Gender, age, education, smoking, drinking"],
    model: ["DCAR prior", "Demographic / contextual model"],
    output: ["Contextual prior", "Cannot set a tier by itself"],
  },
];

export const hub = {
  gate: {
    title: "Central backend",
    items: ["Permutation-null gate", "Recency decay ρ(Δt)", "Reliability and coverage c", "Missing modalities masked"],
  },
  fusion: {
    title: "Reliability-weighted fusion",
    formula: "S(t) = Σ α_m × p_m",
  },
};

export const outcomes = {
  tiers: ["Low", "Medium", "High"],
  abstain: "Insufficient evidence",
  evidence: ["CARE-AnxRAG", "Cited, conflict-checked evidence, or abstention"],
  interfaces: ["Aura · ClinAnx", "Display only; no local scoring"],
};

// Component-by-component view of the same architecture.
export const roles = [
  {
    id: "C1",
    slug: "wearable-forecasting",
    title: "Wearable Biosensor Forecasting",
    input: "Chest-strap ECG, respiration, motion and skin temperature, summarised into 60-second windows of 10 features.",
    process: "An LSTM autoencoder learns the person's baseline; reconstruction error becomes an anomaly signal that a second LSTM uses for short-horizon forecasting.",
    output: "A physiological anomaly score and risk trajectory.",
    fusion: "Eligible input, weighted by informativeness, recency and coverage.",
  },
  {
    id: "C2",
    slug: "behavioural-graphs",
    title: "Spatio-Temporal Graph Learning",
    input: "Passive smartphone sensing collected by Aura, with location coarsened and communication reduced to counts on-device.",
    process: "28 days of sensing become a day × time-segment graph with missingness indicators; GATv2 is compared with flat baselines under leakage-free evaluation.",
    output: "A behavioural research signal, retained for logging and data-quality monitoring.",
    fusion: "Excluded: active fusion weight 0.0 after the final held-out result was not distinguishable from chance.",
  },
  {
    id: "C3",
    slug: "clinical-nlp",
    title: "Clinical NLP / TC-WPN",
    input: "Clinical notes, submitted by clinicians through ClinAnx; the benchmark uses a publication-clean MIMIC-IV cohort.",
    process: "Bio_ClinicalBERT embeddings feed a Temporal-Consistency Weighted Prototypical Network trained on patient-disjoint few-shot episodes.",
    output: "A Clinical NLP signal for the note, not an overall patient risk.",
    fusion: "Eligible input, weighted by informativeness, recency and coverage.",
  },
  {
    id: "C4",
    slug: "fusion-and-rag",
    title: "Multimodal Risk Fusion & RAG Decision Support",
    input: "A contextual profile (gender, age, education, smoking, drinking) plus the eligible outputs of C1 to C3.",
    process: "The DCAR prior is combined with eligible outputs by reliability-weighted fusion; CARE-AnxRAG retrieves, scores and conflict-checks supporting evidence.",
    output: "Low, Medium or High, or insufficient evidence, with cited guidance or an explicit abstention.",
    fusion: "Hosts the fusion rule and the evidence layer; its prior cannot produce a tier alone.",
  },
];
