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
    "Personalization",
    "Individual baselines can matter when interpreting changing signals.",
    "Personalization requires careful calibration and does not remove uncertainty.",
    "Connects the physiological and contextual framing.",
  ],
  [
    "Multimodal modelling",
    "Different evidence streams can be complementary.",
    "Availability, timescale and validation quality differ between modalities.",
    "Motivates reliability-aware fusion rather than equal weighting.",
  ],
  [
    "Uncertainty & reliability",
    "Weak or missing evidence must remain visible as uncertainty.",
    "A numeric output can create false certainty if quality gates are hidden.",
    "Motivates masking, abstention and explicit evidence states.",
  ],
  [
    "Forecasting",
    "Defined-horizon forecasting can be studied for specific signals.",
    "A component forecast should not be generalized into an unsupported multimodal future-event claim.",
    "Keeps physiological forecasting distinct from the current multimodal assessment.",
  ],
  [
    "Evidence-aware decision support",
    "Retrieval can connect outputs with external evidence.",
    "Retrieved evidence can be stale, conflicting or low authority.",
    "Motivates CARE-AnxRAG provenance, contradiction checks, citation validation and abstention.",
  ],
].map(([theme, demonstrates, limitations, relevance]) => ({
  theme,
  demonstrates,
  limitations,
  relevance,
  sourceStatus: "project-literature-synthesis",
  source: "R26-DS-012 repository research documentation",
}));
