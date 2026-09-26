// Component detail. Sources, per component:
//   C1: root README + "Self-supervised Physiological Biosensors/README.md" (Phase 1, April 2026)
//   C2: root README + "graph-behavioral-phenotyping-FULL-v8/README.md" (final v8 source of truth)
//   C3: root README + "Anxiety_Detection_TC_WPN/README_CLEAN_BENCHMARK.md"
//   C4: root README + "Care-AnxRAG/README.md" + C4 integration notes (DCAR inputs)
// Numbers must match those files exactly; label benchmark-stage results as such.

const repo = "https://github.com/dulhara79/R26-DS-012/tree/main/";

export const components = [
  {
    id: "C1",
    headline: "Learn the person, then notice change.",
    dataTitle: "A custom chest strap and four public benchmarks.",
    slug: "wearable-forecasting",
    modality: "Physiology",
    timescale: "Seconds to minutes",
    owner: "Sendanayake H.D.",
    studentId: "IT22107596",
    title: "Wearable Biosensor Forecasting",
    proposalTitle:
      "Personalized Self-Supervised Forecasting of Acute Anxiety Episodes Using Wearable Biosensors",
    tagline:
      "Personalized self-supervised physiological anomaly detection and short-horizon escalation forecasting using a custom chest-strap wearable.",
    coreIdea:
      "Learn each person's baseline physiology rather than requiring large numbers of labelled anxiety episodes. The model learns what resting physiology looks like; when it fails to reconstruct a window, that reconstruction error becomes an anomaly signal for a short-horizon forecasting stage.",
    image: "c1",
    source: `${repo}Self-supervised%20Physiological%20Biosensors`,
    hardware: [
      ["ECG / HRV", "AD8232", "Cardiac rhythm and R-R intervals"],
      [
        "Respiration",
        "BF350-3AA strain gauge",
        "Thoracic expansion and breathing rate",
      ],
      ["Inertial motion", "BMI160 IMU", "3-axis acceleration"],
      ["Skin temperature", "DS18B20", "Peripheral temperature"],
      [
        "Microcontroller",
        "ESP32-C3",
        "Data acquisition and wearable communication",
      ],
    ],
    features: [
      "Mean heart rate",
      "Mean R-R interval",
      "SDNN",
      "RMSSD",
      "Mean breathing rate",
      "Breathing-rate variability",
      "Mean temperature",
      "Temperature variability",
      "Mean acceleration magnitude",
      "Acceleration variability",
    ],
    pipeline: [
      "60-second physiological feature window",
      "LSTM autoencoder",
      "Reconstruction error",
      "Anomaly signal",
      "Short-horizon forecasting",
      "Physiological risk trajectory",
    ],
    datasets: [
      ["WESAD", "Laboratory stress"],
      ["AffectiveROAD", "Real-world driving stress"],
      ["PPG-DaLiA", "Daily activities"],
      ["EmoWear", "Video-elicited emotions"],
    ],
    evaluation: [
      "Strict leave-one-subject-out (LOSO) evaluation.",
      "Global LSTM autoencoder trained only on baseline data.",
      "Masked LSTM-AE variant compared in an ablation.",
      "Secondary LSTM forecasts onset probability from recent anomaly scores.",
    ],
    resultsLabel:
      "Phase 1 benchmark results reported in the Component 1 repository (April 2026)",
    results: [
      ["0.99", "AUROC on WESAD, with 0% false alarm rate"],
      ["0.97", "Combined AUROC across 25 subjects / drives"],
      ["0.88", "Combined forecasting AUROC"],
      ["11.83 min", "Average early warning time"],
      ["+20.7%", "F1 gain from the masked variant on noisier wrist PPG"],
    ],
    status: [
      "Personalization (per-subject fine-tuning of the global model) was listed as the next Phase 1 step.",
      "Phase 2 plans ethics clearance, a deployable strap, and a 4 to 6 week naturalistic study with 10 to 15 young adults.",
    ],
    fusionRole: "Eligible physiological input to fusion.",
  },
  {
    id: "C2",
    headline: "Does structure add signal?",
    dataTitle: "GLOBEM, four cohorts, one strict protocol.",
    slug: "behavioural-graphs",
    modality: "Behaviour",
    timescale: "Days to weeks",
    owner: "Layathma B.M.A.S.",
    studentId: "IT22171542",
    title: "Spatio-Temporal Graph Learning",
    proposalTitle:
      "Graph-Based Spatio-Temporal Behavioral Phenotyping for Personalized Anxiety Vulnerability Mapping",
    tagline:
      "Leakage-free spatio-temporal graph learning for anxiety vulnerability mapping from passive smartphone sensing.",
    coreIdea:
      "Evaluate whether temporal relationships in passive behavioural data provide generalizable anxiety-vulnerability information beyond simpler flat feature models, under a strict participant-grouped, cross-cohort protocol.",
    image: "c2",
    source: `${repo}graph-behavioral-phenotyping-FULL-v8`,
    cohorts: [
      ["INS-W_1", "2018", "Model and hyperparameter selection only"],
      ["INS-W_2", "2019", "Held-out evaluation"],
      ["INS-W_3", "2020", "Held-out evaluation"],
      ["INS-W_4", "2021", "Held-out evaluation"],
    ],
    target:
      "Released binary anx_weekly_subscale: the PHQ-4 anxiety subscale / GAD-2-derived label in GLOBEM.",
    pipeline: [
      "Previous 28 days of passive sensing",
      "Four daily segments: morning, afternoon, evening, night",
      "Node = one available day × segment cell",
      "Edges: adjacent segments within a day; same segment on consecutive days",
      "40 behavioural values + 40 missingness indicators = 80 node features",
      "GATv2 vs Logistic Regression, Random Forest, Gradient Boosting",
    ],
    evaluation: [
      "The same participant never appears in both train and test folds.",
      "Normalization, early stopping, calibration and threshold selection use training-fold data only.",
      "Each test fold is evaluated once, after every choice is fixed.",
      "Compared against a prevalence-preserving, participant-level permutation null.",
    ],
    resultsLabel: "Final v8 held-out results",
    results: [
      ["0.5205", "Held-out GATv2 AUROC"],
      ["0.485–0.560", "95% participant-clustered CI"],
      ["0.4991", "50-permutation null mean (p = 0.255)"],
      ["0.2270", "AUPRC"],
    ],
    baselines: [
      ["Logistic Regression", "0.5458"],
      ["Random Forest", "0.5617"],
      ["Gradient Boosting", "0.5681"],
      ["GATv2", "0.5205"],
    ],
    loco: [
      ["GATv2", "0.5529", "0.5036", "+0.0493"],
      ["Logistic Regression", "0.5832", "0.5553", "+0.0279"],
      ["Random Forest", "0.5654", "0.5417", "+0.0237"],
      ["Gradient Boosting", "0.5503", "0.5385", "+0.0117"],
    ],
    status: [
      "The final held-out GATv2 result was not distinguishable from chance, and the graph did not outperform simpler flat baselines.",
      "The pre-registered cohort-shift robustness hypothesis for the graph was not supported.",
      "Earlier vulnerability scores, hourly high-risk windows and phenotype outputs are historical exploratory work, not validated clinical outputs.",
    ],
    fusionRole:
      "Active fusion weight 0.0. Retained for research logging, data-quality monitoring and future model development.",
  },
  {
    id: "C3",
    headline: "Learn from few notes, without leakage.",
    dataTitle: "A publication-clean MIMIC-IV benchmark.",
    slug: "clinical-nlp",
    modality: "Clinical language",
    timescale: "Clinical encounters",
    owner: "Kaushalya I.G.D.",
    studentId: "IT22130648",
    title: "Clinical NLP / TC-WPN",
    proposalTitle:
      "Temporal-Confidence Weighted Prototypical Networks for Few-Shot Clinical Anxiety Detection",
    tagline:
      "Patient-disjoint few-shot clinical NLP for anxiety detection using Temporal-Consistency Weighted Prototypical Networks.",
    coreIdea:
      "Build a few-shot clinical NLP system that learns from small labelled support sets, without the same patient appearing in both support and query data and without deriving labels from the note text itself.",
    image: "c3",
    source: `${repo}Anxiety_Detection_TC_WPN`,
    benchmark: [
      "Patient-disjoint support and query episodes",
      "Structured cohort construction",
      "Fixed index-time policies",
      "Explicit leakage certificates",
      "Frozen episode plans",
      "Shallow baselines before neural comparison",
      "Blinded robustness arms",
      "MIMIC-III reserved for cross-dataset transfer, not mixed into training",
    ],
    policies: [
      ["at_or_before", "Concurrent anxiety detection"],
      ["strictly_before", "Prospective detection"],
      ["none", "Retrospective association only"],
    ],
    pipeline: [
      "Bio_ClinicalBERT",
      "256-dimensional projection",
      "Prototypical few-shot classifier",
      "Temporal weight (wT) · prototype-consistency weight (wC) · learned temperature (τ)",
      "Auxiliary cross-entropy head",
    ],
    evaluation: [
      "Primary experiment: anxiety versus other psychiatric illness in MIMIC-IV.",
      "Leakage removal is enforced by tests that run in CI, not by convention.",
      "wC measures how typical a support note is of its class prototype. It is not calibrated confidence.",
    ],
    resultsLabel: "Paper-aligned result reported in the repository",
    results: [
      [
        "≈0.738",
        "Clinical-note AUROC in the deployment-relevant held-out setting used by the fusion design",
      ],
    ],
    status: [
      "TC-WPN contributes a Clinical NLP signal. It is not overall patient risk.",
    ],
    fusionRole: "Eligible clinical-language input to fusion.",
  },
  {
    id: "C4",
    headline: "Trust decides the weight.",
    dataTitle: "A contextual prior and eligible outputs.",
    slug: "fusion-and-rag",
    modality: "Context + fusion",
    timescale: "Background and integration",
    owner: "Seneviratne K.A.U.A.",
    studentId: "IT22093950",
    title: "Multimodal Risk Fusion & RAG Decision Support",
    proposalTitle:
      "A Continuously Learning Personalized Anxiety Intervention Framework Using Multimodal Risk Signals",
    tagline:
      "Contextual risk modelling, reliability-weighted multimodal fusion and evidence-aware retrieval-augmented decision support.",
    coreIdea:
      "Combine a contextual prior with the outputs of components that are available, recent and validated, then attach evidence that has passed quality, relevance and conflict checks. The current C4 replaces the older intervention-engine design described in the proposal.",
    image: "c4",
    source: `${repo}Multimodal%20Risk%20Fusion%20and%20Retrieval-Augmented%20Clinical%20Decision%20Support`,
    pipeline: [
      "DCAR contextual / demographic prior",
      "+ eligible component outputs",
      "Reliability-weighted fusion",
      "Low / Medium / High tier, or insufficient evidence",
      "Retrieval-augmented decision support (CARE-AnxRAG)",
    ],
    dcar: "The DCAR prior uses only gender, age, education, smoking and drinking, so it shares no input with the other streams. It is deliberately prevented from producing a tier by itself.",
    formula: [
      ["w_m(t) = ω_m × ρ_m(Δt) × c_m", "Raw weight per modality"],
      ["α_m = w_m / Σw", "Normalized contribution"],
      ["S(t) = Σ α_m × p_m", "Fused score"],
    ],
    terms: [
      ["ω", "Deployment-relevant informativeness above chance"],
      ["ρ", "Modality-specific recency decay"],
      ["c", "Reliability and coverage scaling"],
    ],
    rag: [
      "Hybrid dense + lexical retrieval (Chroma and SQLite FTS5 BM25)",
      "Reciprocal Rank Fusion and CrossEncoder reranking",
      "Independent relevance gate",
      "Evidence authority, freshness and applicability scoring",
      "Provenance and source versioning in a SQLite ledger",
      "NLI-based contradiction detection",
      "Calibrated abstention when evidence is weak or conflicting",
      "Citation validation of generated answers",
      "Crisis and urgent-message routing before ordinary retrieval",
    ],
    evaluation: [
      "A component that fails its own permutation-null criterion receives zero base weight.",
      "Unavailable modalities are masked, never read as zero risk.",
      "UI colours may distinguish unavailable evidence, but there are only three clinical tiers.",
    ],
    status: [
      "CARE-AnxRAG is a research and engineering system, not a clinical device.",
      "Under current evidence, Component 2 is excluded from active fusion.",
    ],
    fusionRole: "Hosts the fusion rule and the evidence layer.",
  },
];

export const componentBySlug = Object.fromEntries(
  components.map((component) => [component.slug, component]),
);
