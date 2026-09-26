// Research domain content. Sources: R26-DS-012 root README (overview, problem
// table, privacy commitments, tech stack) and the curated literature synthesis
// carried over from the v1 site. Keep every statement traceable to those sources.

export const overview = {
  lead: "Anxiety disorders affect millions of people worldwide, while access to continuous and personalized mental-health monitoring remains limited.",
  body: [
    "Clinical appointments provide important but episodic snapshots, and anxiety-related changes can also appear in physiology, everyday behaviour and clinical documentation at different timescales.",
    "This research investigates a multimodal alternative: four complementary components whose outputs are combined only when they are available, recent and validated.",
  ],
  paradigms:
    "The components do not all use the same learning paradigm. Component 1 is primarily self-supervised; Components 2 and 3 use labelled benchmark data for evaluation and training; Component 4 combines eligible component outputs with contextual information and evidence retrieval.",
};

export const problems = [
  [
    "Anxiety-related change is continuous and multimodal",
    "Clinical appointments capture only part of the person's changing state.",
  ],
  [
    "Physiological baselines differ across people",
    "A population threshold may not represent an individual's normal physiology.",
  ],
  [
    "Passive behavioural signals are noisy and cohort-dependent",
    "Apparent performance can disappear under leakage-free external evaluation.",
  ],
  [
    "Clinical NLP has limited labelled data",
    "Patient-level leakage and text-derived labels can strongly inflate results.",
  ],
  [
    "Modalities operate on different timescales",
    "A minute-level physiological signal, multi-week behaviour and a clinical note cannot be treated as identical evidence.",
  ],
  [
    "Missing or weak modalities are common",
    "A safe system must be able to withhold a decision rather than inventing a score.",
  ],
].map(([challenge, why]) => ({ challenge, why }));

export const timescales = [
  ["Seconds to minutes", "Physiology", "60-second wearable feature windows"],
  ["Days to weeks", "Behaviour", "28 days of passive smartphone sensing"],
  ["Clinical encounters", "Clinical language", "Documented clinical notes"],
  ["Background", "Context", "Demographic and contextual factors"],
].map(([when, stream, detail]) => ({ when, stream, detail }));

export const literatureThemes = [
  [
    "Physiological sensing",
    "Physiological signals can capture short-timescale change, but baselines differ substantially across people.",
    "Population-level thresholds may not represent an individual baseline.",
    "Motivates personalized anomaly modelling in C1.",
  ],
  [
    "Passive behavioural sensing",
    "Everyday sensing can encode longer-term behavioural patterns.",
    "Signals are noisy, cohort-dependent and vulnerable to leakage or weak external generalization.",
    "Motivates the participant-grouped C2 evaluation and its explicit exclusion when evidence is weak.",
  ],
  [
    "Clinical NLP",
    "Clinical documentation can contain relevant anxiety-related evidence.",
    "Label scarcity, patient overlap and text-derived labels can inflate apparent performance.",
    "Motivates patient-disjoint few-shot evaluation in C3.",
  ],
  [
    "Multimodal modelling",
    "Different evidence streams can be complementary.",
    "Availability, timescale and validation quality differ between modalities.",
    "Motivates reliability-aware fusion rather than equal weighting.",
  ],
  [
    "Uncertainty and reliability",
    "Weak or missing evidence must remain visible as uncertainty.",
    "A numeric output can create false certainty if quality gates are hidden.",
    "Motivates masking, abstention and explicit evidence states.",
  ],
  [
    "Evidence-aware decision support",
    "Retrieval can connect outputs with external evidence.",
    "Retrieved evidence can be stale, conflicting or low authority.",
    "Motivates CARE-AnxRAG provenance, contradiction checks, citation validation and abstention.",
  ],
].map(([theme, shows, limits, relevance]) => ({ theme, shows, limits, relevance }));

export const researchGap = {
  headline:
    "The challenge is not collecting more signals. It is understanding how much each signal should be trusted.",
  body: "A useful framework has to combine heterogeneous evidence without hiding uncertainty or assuming that every available modality deserves equal influence.",
  principle: "Missing evidence is not evidence of low anxiety.",
};

export const researchQuestion =
  "Can heterogeneous digital biomarkers be combined responsibly to build a more personalized and temporally aware picture of anxiety vulnerability?";

export const objectives = {
  main: "To develop and evaluate a multimodal digital-biomarker framework for personalized vulnerability mapping and acute escalation forecasting in young adults with anxiety disorders, with validation-aware outputs and explicit safety and fusion gates.",
  specific: [
    ["C1", "Learn each person's physiological baseline from wearable data and use deviations for short-horizon escalation forecasting."],
    ["C2", "Test whether spatio-temporal graphs of passive smartphone sensing generalize beyond simpler flat baselines under leakage-free evaluation."],
    ["C3", "Detect anxiety in clinical notes with few-shot learning while preventing patient leakage and label contamination."],
    ["C4", "Combine eligible component outputs with a contextual prior through reliability-weighted fusion and evidence-aware decision support."],
  ].map(([id, text]) => ({ id, text })),
};

export const methodology = [
  ["Problem formulation", "Define anxiety-related change as continuous, multimodal and person-specific."],
  ["Literature review", "Identify what each modality can show and how each one fails."],
  ["Dataset and signal selection", "Benchmarks WESAD, AffectiveROAD, PPG-DaLiA, EmoWear, GLOBEM and MIMIC-IV; a custom ESP32-C3 chest strap as the target wearable."],
  ["Component research", "Four components studied with the learning paradigm that suits each signal."],
  ["Leakage-aware evaluation", "Subject-level, participant-grouped and patient-disjoint protocols with permutation nulls."],
  ["Integration", "Components publish outputs and status metadata to a central backend."],
  ["Reliability-aware fusion", "Weights by informativeness, recency and coverage; unavailable modalities are masked."],
  ["System validation", "Contract tests, CI workflows and explicit abstention states."],
  ["Research outputs", "Component papers, dissertation and the integrated research prototype."],
].map(([step, detail]) => ({ step, detail }));

export const techStack = [
  ["Wearable and physiological ML", ["Python", "PyTorch", "NeuroKit2", "NumPy", "ESP32-C3"]],
  ["Behavioural graph ML", ["PyTorch Geometric", "scikit-learn", "Pandas", "GLOBEM"]],
  ["Clinical NLP", ["PyTorch", "Hugging Face", "Bio_ClinicalBERT", "MIMIC-IV"]],
  ["Fusion and RAG", ["FastAPI", "scikit-learn", "Chroma", "SQLite FTS5", "Ollama", "Docker"]],
  ["Mobile and infrastructure", ["Flutter", "Supabase", "GitHub Actions"]],
].map(([area, tools]) => ({ area, tools }));

export const commitments = [
  "Fusion consumes component-level outputs and status metadata, not raw modality streams as interchangeable features.",
  "Mobile location is coarsened on-device before research upload (latitude and longitude rounded to three decimals).",
  "App identities are pseudonymous participant codes rather than participant names.",
  "Communication sensing stores aggregate call and SMS counts, never message bodies, phone numbers, contact names or call content.",
  "Clinical data used for research must be de-identified and governed by the applicable dataset and institutional access requirements.",
  "Weak, unavailable, stale or unvalidated modalities can be withheld from fusion rather than treated as zero risk.",
  "Component 2's final output is not a validated clinical anxiety probability and currently receives zero active fusion weight.",
  "RAG outputs are evidence-gated and may abstain when relevance, quality, diversity or conflict checks fail.",
  "The integrated framework is research and clinical decision support, not a diagnostic medical device.",
  "Participants retain the right to withdraw in accordance with the approved study protocol.",
];
