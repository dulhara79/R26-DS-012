// Integrated group methodology, synthesized from the five project draft reports.
// The draft reports are internal working sources, NOT entries in the public bibliography.
// Numbered IEEE citations below identify independently published datasets, methods,
// instruments, and reporting guidance; they are not validation of the team's system.
export const groupMethodology = {
  introduction: "R26-DS-012 follows a modular design-and-evaluation process: define separate research targets for each signal, build and test the individual analytical services, apply explicit evidence-quality gates, and provide a shared, auditable assessment through the participant and clinician applications. The complete workflow is presented here as one group methodology rather than four independent clinical prediction studies.",
  stages: [
    {
      title: "Research design and data sources",
      paragraphs: [
        "The team conducted retrospective computational experiments using distinct sources because no single dataset contained time-aligned wearable, smartphone, clinical-note and contextual observations with independently adjudicated acute anxiety outcomes. Physiological development used WESAD [1], with AffectiveROAD serving as an external physiological-domain check [2]. Longitudinal behavioural modelling used the multi-year GLOBEM dataset [3], while clinical-note analysis used MIMIC-IV and associated de-identified narratives [4]. These datasets have different populations, measurement contexts and labels and were not treated as a matched multimodal participant cohort.",
        "The group also documents a small NHSL psychiatric-ward data-collection activity involving eight patients. This activity informs the intended clinical setting, but the reviewed material does not establish a linked, independently adjudicated outcome dataset for evaluating the complete fusion score. The reported experiments therefore remain benchmark-based and the overall solution remains a research prototype."
      ]
    },
    {
      title: "Acquisition, preprocessing and outcome definition",
      paragraphs: [
        "The physiological pathway uses ten cardiac, respiratory, temperature and movement features from one-minute windows, with participant-baseline normalisation, signal-quality controls and contiguous sequence requirements. The Aura wearable prototype transmits derived sensor features by Bluetooth to the mobile application and onward to a model service; equivalence between live-device features and benchmark-derived features has not yet been independently established.",
        "The behavioural pathway aggregates passive phone measurements over the preceding 28 days into day-by-time-segment representations with missingness indicators. Its retrospective weekly research outcome is derived from the anxiety items of the PHQ-4 instrument [5]. Clinical-note records are assigned research labels from structured diagnoses and filtered by patient and index time; patient-disjoint train/test and support/query separation is used to reduce leakage. The contextual pathway uses demographic and questionnaire-associated inputs, including the GAD-7 instrument [6], as a slowly changing prior rather than an acute-event label."
      ]
    },
    {
      title: "Analytical model development",
      paragraphs: [
        "For physiology, a baseline-trained LSTM autoencoder learns normal short-term signal patterns; reconstruction error is converted to a bounded C1 anomaly score. A separate Ridge regression model estimates the physiological anomaly score at +5 and +10 minutes. LSTM provides the recurrent modelling foundation [7], but these outputs are not estimated probabilities of a diagnosed anxiety episode.",
        "For behaviour, a GATv2 model represents temporal relationships between sensing segments [8] and is compared with non-graph baselines and a participant-preserving permutation null. Its held-out evidence did not justify an anxiety-risk contribution, so the research system withholds the behavioural model score from active fusion; descriptive baseline and change observations remain available.",
        "For clinical language, Bio_ClinicalBERT provides note embeddings [9] for a few-shot prototypical classifier [10]. TC-WPN evaluates temporal-recency and prototype-consistency weights through controlled ablations, frozen episodes and lexical-blinding checks. The evaluated outcome is concurrent discrimination between anxiety-coded and psychiatric-comparator records, not prospective patient-level escalation."
      ]
    },
    {
      title: "Reliability-aware integration and explainability",
      paragraphs: [
        "Each service returns a model-specific reading together with availability, timestamp and provenance information. The central backend checks identity, freshness, admissibility and completeness before requesting Reliability-Aware Gated Fusion (RAGF). The active policy requires at least two usable inputs, including a dynamic modality; the unsupported behavioural score is excluded and the contextual prior cannot independently determine the assessment. Eligible scores are harmonised and combined with normalised non-negative weights reflecting the configured reliability, recency and coverage rules. Missing or rejected evidence is recorded as unavailable, not silently replaced with a low score.",
        "CARE-X explains recorded contributions, exclusion reasons and fixed-panel counterfactual calculations using the stored assessment. Separately, CARE-AnxRAG uses evidence retrieval with source provenance and abstention when material is insufficient or conflicting; retrieval-augmented generation is the underlying information-access paradigm [11]. Neither retrieved guidance nor an explanation of the arithmetic substitutes for clinician judgment."
      ]
    },
    {
      title: "Application and system workflow",
      paragraphs: [
        "Aura provides participant-facing acquisition and bounded research views, while ClinAnx presents authorised clinician-facing context, notes, assessment provenance and attention-event states. The backend, rather than either user interface, owns the canonical participant identity, persisted fusion result and event transitions. This prevents client-specific recomputation from producing different authoritative assessments. The applications consume the same recorded result but expose information according to role and permission; system-level contract tests and deployment checks assess this architecture independently of clinical outcome validity."
      ]
    },
    {
      title: "Validation, reproducibility and research safeguards",
      paragraphs: [
        "Evaluation is matched to each scientific target. Physiology is assessed with participant-held-out discrimination, external-domain checks and forecast mean absolute error versus persistence. Behaviour is tested on held-out GLOBEM cohorts with baselines, permutation tests and domain-shift analyses. Clinical NLP uses patient-disjoint episodes, multiple seeds, weighting ablations and vocabulary-blinding sensitivity. Contextual estimation is evaluated separately on held-out questionnaire-associated observations; fusion and explanation are evaluated with deterministic scenarios, simulation and arithmetic checks. These metrics cannot be pooled into an accuracy figure for the complete system because the cohorts and outcomes are different.",
        "The team records experiment configurations, missingness and failure states, data provenance and controlled-access expectations. Transparent reporting principles for clinical prediction models provide a useful standard for distinguishing methodological and software verification from clinical validation [12]. Prospective validation on a consented, jointly measured target cohort and clinician-led usability and outcome assessment remain necessary before any diagnostic, treatment or real-world escalation claim."
      ]
    }
  ]
};

// IEEE-order bibliography: independent, externally published sources only.
export const methodologyReferences = [
  {number:1,citation:'P. Schmidt, A. Reiss, R. Duerichen, C. Marberger, and K. Van Laerhoven, "Introducing WESAD, a multimodal dataset for wearable stress and affect detection," in Proc. ACM Int. Conf. Multimodal Interaction (ICMI), 2018, pp. 400–408, doi: 10.1145/3242969.3242985.',url:"https://doi.org/10.1145/3242969.3242985"},
  {number:2,citation:'N. El Haouij, J.-M. Poggi, S. Sevestre-Ghalila, R. Ghozi, and M. Jaïdane, "AffectiveROAD system and database to assess driver’s attention," in Proc. 33rd ACM Symp. Applied Computing, 2018, pp. 800–803, doi: 10.1145/3167132.3167395.',url:"https://doi.org/10.1145/3167132.3167395"},
  {number:3,citation:'X. Xu et al., "GLOBEM Dataset: Multi-Year Datasets for Longitudinal Human Behavior Modeling Generalization," in Advances in Neural Information Processing Systems, vol. 35, Datasets and Benchmarks Track, 2022.',url:"https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c7e8a0821dfcb58a9a83cbd37cc8131-Abstract-Datasets_and_Benchmarks.html"},
  {number:4,citation:'A. E. W. Johnson et al., "MIMIC-IV, a freely accessible electronic health record dataset," Scientific Data, vol. 10, art. no. 1, 2023, doi: 10.1038/s41597-022-01899-x.',url:"https://doi.org/10.1038/s41597-022-01899-x"},
  {number:5,citation:'K. Kroenke, R. L. Spitzer, J. B. W. Williams, and B. Löwe, "An ultra-brief screening scale for anxiety and depression: The PHQ-4," Psychosomatics, vol. 50, no. 6, pp. 613–621, 2009, doi: 10.1176/appi.psy.50.6.613.',url:"https://doi.org/10.1176/appi.psy.50.6.613"},
  {number:6,citation:'R. L. Spitzer, K. Kroenke, J. B. W. Williams, and B. Löwe, "A brief measure for assessing generalized anxiety disorder: The GAD-7," Archives of Internal Medicine, vol. 166, no. 10, pp. 1092–1097, 2006, doi: 10.1001/archinte.166.10.1092.',url:"https://doi.org/10.1001/archinte.166.10.1092"},
  {number:7,citation:'S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997, doi: 10.1162/neco.1997.9.8.1735.',url:"https://doi.org/10.1162/neco.1997.9.8.1735"},
  {number:8,citation:'S. Brody, U. Alon, and E. Yahav, "How attentive are graph attention networks?," in Proc. Int. Conf. Learning Representations (ICLR), 2022.',url:"https://openreview.net/forum?id=F72ximsx7C1"},
  {number:9,citation:'E. Alsentzer et al., "Publicly available clinical BERT embeddings," in Proc. 2nd Clinical Natural Language Processing Workshop, 2019, pp. 72–78, doi: 10.18653/v1/W19-1909.',url:"https://doi.org/10.18653/v1/W19-1909"},
  {number:10,citation:'J. Snell, K. Swersky, and R. Zemel, "Prototypical networks for few-shot learning," in Advances in Neural Information Processing Systems, vol. 30, 2017.',url:"https://papers.neurips.cc/paper/6996-prototypical-networks-for-few-shot-learning"},
  {number:11,citation:'P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in Advances in Neural Information Processing Systems, vol. 33, 2020.',url:"https://papers.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html"},
  {number:12,citation:'G. S. Collins et al., "TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods," BMJ, vol. 385, art. e078378, 2024, doi: 10.1136/bmj-2023-078378.',url:"https://doi.org/10.1136/bmj-2023-078378"}
];
