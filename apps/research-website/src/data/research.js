// Research domain content. Sources: R26-DS-012 root README (overview, problem
// table, privacy commitments, tech stack) and the curated literature synthesis
// carried over from the v1 site. Keep every statement traceable to those sources.

// Website-ready group abstract, grounded in the October 2026 R26-DS-012
// Final Group Report (abstract; chapters 3, 6 and 8). Retain the report's
// uncertainty about clinical validity and audited benchmark performance.
export const groupAbstract = {
  paragraphs: [
    "Anxiety-related vulnerability cannot be characterized adequately by isolated clinical observations or a single digital signal. Physiological activity, everyday behaviour, clinical narratives and contextual factors vary in measurement frequency, reliability and clinical meaning. This study presents R26-DS-012, a modular multimodal digital-biomarker research framework for investigating personalized vulnerability mapping and short-horizon physiological change in young adults with anxiety disorders. Its central premise is that evidence should contribute to an integrated assessment only when its provenance, recency and validation status support its use.",
    "Using distinct retrospective research datasets rather than a jointly measured patient cohort, the group developed wearable-based physiological anomaly modelling with +5- and +10-minute anomaly-index forecasting; longitudinal passive-sensing graph analysis; few-shot anxiety-related clinical-note classification; and reliability-aware fusion supported by contextual inputs and source-grounded retrieval. The integration layer applies eligibility, freshness and coverage checks, preserves unavailable evidence as missing, and supports abstention and traceable explanations across participant- and clinician-facing research applications.",
    "Evaluation provided component-specific rather than unified clinical evidence. The final behavioural graph model achieved a held-out AUROC of 0.5205 (permutation p = 0.255), resulting in its exclusion from active fusion. Archived clinical NLP experiments indicated note-classification potential, although a subsequent audit identified leakage risks requiring independently verified evaluation under corrected protocols. The group also reported data collection involving eight patients at the National Hospital of Sri Lanka; however, linked outcome records and prospective validation of the integrated score were not available for audit. The principal contribution is therefore an auditable, evidence-sensitive architecture with explicit failure safeguards, not a clinically validated diagnostic or acute-event prediction system.",
  ],
  keywords: [
    "Anxiety disorders",
    "Digital biomarkers",
    "Multimodal fusion",
    "Physiological anomaly forecasting",
    "Clinical natural language processing",
    "Evidence-aware decision support",
  ],
  source: "R26-DS-012 Final Group Report · October 2026",
};

// Concise group-level introduction for the homepage. The full paper-style
// abstract remains exclusive to the Domain page.
export const homeContext = {
  lead: "Anxiety-related vulnerability unfolds across physiological changes, everyday behaviour and clinical context—not within a single observation.",
  body: [
    "R26-DS-012 investigates wearable physiology, longitudinal behavioural signals and clinical notes as complementary but differently timed research evidence. Each component is evaluated on its own datasets and under its own limitations.",
    "The integrated prototype is designed to admit only eligible, sufficiently reliable evidence and to withhold unsupported conclusions. It is not a clinically validated diagnostic or acute-event prediction system.",
  ],
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

// Group-level synthesis adapted from R26-DS-012 Final Group Report
// (October 2026), Chapter 1 and Chapter 2.2 "Root causes and overall research gap".
// Citation numbers intentionally refer to methodologyReferences on this page,
// NOT the numbering used in the group-report PDF.
export const groupLiteratureReview = {
  introduction: [
    "The literature motivating R26-DS-012 concerns more than the prediction of anxiety from an individual sensor. It asks how clinical assessment, repeated digital observations and recorded patient context can inform one another without assuming that any single measurement is a clinical diagnosis. Screening instruments such as the PHQ-4 and GAD-7 capture defined constructs [5] [6], while digital datasets offer different windows onto physiology, behaviour and clinical documentation [1] [3] [4].",
    "Taken together, these sources point to three connected challenges. First, clinical encounters and digital measurements observe different parts of an individual's experience. Second, apparent predictive performance may not generalize across participants, populations, collection settings or outcome definitions. Third, integrating evidence into a decision-support workflow requires explicit treatment of quality, uncertainty and provenance rather than simply combining model scores.",
  ],
  themes: [
    {
      id: "observation",
      number: "01",
      title: "Clinical assessment and longitudinal observation",
      paragraphs: [
        "Traditional clinical assessment draws on patient history, interviews and purpose-specific screening instruments. The PHQ-4 and GAD-7 are useful examples of standardized measures, but neither turns an isolated digital observation into a verified diagnosis or a forecast of an acute event [5] [6]. Repeated measurements may reveal changes between appointments, yet they must be interpreted within the clinical and personal context in which they were collected.",
        "Published research resources illustrate both the opportunity and the constraint. WESAD records stress- and affect-related wearable signals in a laboratory setting [1]; AffectiveROAD documents physiological sensing in a driving environment [2]; GLOBEM supplies multi-year longitudinal behavioural measurements [3]; and MIMIC-IV provides hospital electronic health records [4]. These are complementary research sources, not one matched cohort with common anxiety-escalation labels. Their observations span different people, settings, outcomes and timescales.",
      ],
    },
    {
      id: "validity",
      number: "02",
      title: "Predictive validity, dataset shift and evaluation",
      paragraphs: [
        "Model performance is meaningful only in relation to the population, label and evaluation protocol that produced it. Stress-related physiology, retrospective behavioural associations and anxiety-coded clinical notes are not interchangeable targets. Evidence drawn from wearable, longitudinal sensing and hospital-record datasets therefore cannot be pooled into a single claim of clinical anxiety-prediction accuracy [1] [3] [4].",
        "The research literature and reporting guidance make careful validation central to interpretation: subject or patient overlap, missing observations, changes in collection conditions, weak comparison baselines and uncertain calibration can each distort conclusions. TRIPOD+AI emphasizes transparent reporting of prediction-model development and evaluation [12]. In a multimodal project, this means that an apparently sophisticated model is not automatically a trustworthy input to a combined assessment.",
      ],
    },
    {
      id: "integration",
      number: "03",
      title: "Evidence integration, uncertainty and explainability",
      paragraphs: [
        "Combining heterogeneous outputs presents a distinct problem from training their underlying models. Physiological deviations may be available over minutes, behavioural summaries over weeks, clinical notes at encounters and contextual factors over longer intervals. Missing or stale inputs must not be silently converted into reassuring scores, and unaligned numerical outputs should not be interpreted as calibrated probabilities of the same clinical event.",
        "Evidence retrieval adds another layer of responsibility. Retrieval-augmented generation provides a framework for consulting external material [11], but a retrieved passage can still be irrelevant, outdated or contradictory. Transparent prediction-model reporting [12] and explicit source provenance support the case for indicating when evidence is insufficient, where it came from and why an integrated output may need to be withheld. These are design and validation requirements, not evidence that a finished clinical system has been validated.",
      ],
    },
  ],
  synthesis: {
    title: "What the literature means for the group research",
    summary: "The shared gap is not simply that no single application combines four data types. It is the lack of an adequately validated, uncertainty-aware process for deciding which differently timed signals can contribute to an interpretable assessment—and which should be excluded or left undecided.",
    boundary: "The reviewed sources motivate an integrated research framework; they do not establish prospective clinical effectiveness. R26-DS-012 evaluates its streams on separate datasets and retains explicit evidence-quality and fusion boundaries.",
  },
};

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
  ["Dataset and signal selection", "C1: WESAD and AffectiveROAD; C2: GLOBEM; C3: MIMIC-IV and MIMIC-IV-Note; C4: contextual survey data and controlled fusion-policy fixtures. The ESP32-C3 strap is a research prototype, not a device independently evaluated for clinical use."],
  ["Component research", "C1 LSTM anomaly modelling and Ridge forecasting; C2 leakage-audited GATv2 with negative held-out results; C3 Bio_ClinicalBERT episodic prototypes; C4 DCAR, RAGF, CARE-X and CARE-AnxRAG."],
  ["Leakage-aware evaluation", "Subject-level, participant-grouped and patient-disjoint protocols with permutation nulls."],
  ["Integration", "Components publish outputs and status metadata to a central backend."],
  ["Reliability-aware fusion", "Weights by informativeness, recency and coverage; unavailable modalities are masked."],
  ["System validation", "Contract tests, CI workflows and explicit abstention states."],
  ["Research outputs", "Four October 2026 individual thesis drafts, an integrated group report and a non-diagnostic research prototype; public publication status is tracked separately."],
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
