// Copy for the landing hero scroll. Sources: R26-DS-012 root README and component READMEs.

export const heroVideo = {
  src: "/media/research-hero.mp4",
  poster: "/media/research-hero-poster.webp",
};

export const cinematicIntro = {
  eyebrow: "R26-DS-012 · SLIIT · 2026",
  title: "Understanding anxiety beyond a single moment.",
  fullTitle:
    "A Multimodal Digital Biomarker Framework for Personalized Vulnerability Mapping and Acute Escalation Forecasting in Young Adults with Anxiety Disorders",
  lead: "Physiology, everyday behaviour, clinical language and context, read together to map personal vulnerability and forecast acute escalation in young adults with anxiety.",
  tags: [
    "Wearable physiology",
    "Passive sensing",
    "Clinical NLP",
    "Reliability-aware fusion",
  ],
};

export const cinematicPanels = {
  signals: {
    eyebrow: "Anxiety across timescales",
    title: "Anxiety leaves traces across time.",
    body: "Physiology shifts in seconds, behaviour over weeks, clinical language at each encounter. The framework asks how these signals can be read together, responsibly.",
    facts: [
      ["4", "Evidence streams: physiology, behaviour, clinical, context"],
      ["3", "Fusion tiers, plus an explicit insufficient-evidence state"],
    ],
  },
  fusion: {
    eyebrow: "Reliability-aware fusion",
    title: "Missing evidence is not low risk.",
    body: "Fusion weights each modality by informativeness, recency and coverage, masks what is unavailable, and can abstain rather than invent a score.",
    action: "Explore the research",
  },
  components: {
    eyebrow: "Four components · one framework",
    title: "Each signal, studied on its own terms.",
  },
};

export const cinematicSights = [
  {
    id: "c1",
    href: "/components/wearable-forecasting",
    kicker: "C1 · Physiology",
    title: "Wearable Forecasting",
    body: "A chest strap learns each person's baseline; anomalies drive short-horizon forecasts.",
    icon: "pulse",
  },
  {
    id: "c2",
    href: "/components/behavioural-graphs",
    kicker: "C2 · Behaviour",
    title: "Behavioural Graphs",
    body: "GATv2 over 28-day GLOBEM sensing graphs, tested leakage-free across cohorts.",
    icon: "graph",
  },
  {
    id: "c3",
    href: "/components/clinical-nlp",
    kicker: "C3 · Clinical NLP",
    title: "TC-WPN",
    body: "Patient-disjoint few-shot prototypes over Bio_ClinicalBERT, AUROC ≈ 0.738.",
    icon: "notes",
  },
  {
    id: "c4",
    href: "/components/fusion-and-rag",
    kicker: "C4 · Fusion",
    title: "Reliability Fusion",
    body: "Weights by informativeness, recency and coverage into Low, Medium or High.",
    icon: "fusion",
  },
  {
    id: "rag",
    href: "/components/fusion-and-rag",
    kicker: "C4 · Evidence",
    title: "CARE-AnxRAG",
    body: "Evidence-aware retrieval with contradiction checks, citations and abstention.",
    icon: "evidence",
  },
];
