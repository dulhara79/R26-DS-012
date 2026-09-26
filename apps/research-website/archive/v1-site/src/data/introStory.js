import { researchMeta, researchNarrative } from "./research";

export const introChapters = [
  {
    id: "context",
    number: "01",
    eyebrow: "THE CONTEXT",
    title: "Anxiety is not a single moment.",
    body: researchNarrative.context.body,
    note: "Relevant evidence can unfold on very different timescales.",
    details: researchNarrative.timescales.map(([label, value]) => ({ label, value })),
    scene: "field",
  },
  {
    id: "why",
    number: "02",
    eyebrow: "WHY THIS RESEARCH",
    title: "Important change can happen between clinical snapshots.",
    body: researchNarrative.snapshot.body,
    note:
      "The goal is not to replace clinical judgement. It is to investigate whether additional digital evidence can make change more visible between encounters.",
    details: [
      { label: "SNAPSHOT", value: "A valuable assessment at one point in time." },
      { label: "BETWEEN", value: "Change can continue outside the clinical encounter." },
      { label: "PERSON", value: "The same signal can mean different things for different people." },
    ],
    scene: "snapshots",
  },
  {
    id: "literature",
    number: "03",
    eyebrow: "WHAT THE LITERATURE SHOWS",
    title: "Each signal can be informative. Each signal also has a different failure mode.",
    body:
      "Existing work motivates physiological sensing, passive behavioural sensing, clinical NLP, personalization and multimodal modelling. It also shows why reliability, leakage, generalization and unequal timescales cannot be ignored.",
    details: [
      { label: "PHYSIOLOGY", value: "Short-timescale change; individual baselines differ." },
      { label: "BEHAVIOUR", value: "Longer-term patterns; noisy and cohort-dependent." },
      { label: "CLINICAL NLP", value: "Documented evidence; label scarcity and leakage matter." },
      { label: "MULTIMODAL", value: "Complementary evidence; availability and validation quality differ." },
    ],
    scene: "literature",
  },
  {
    id: "gap",
    number: "04",
    eyebrow: "THE RESEARCH GAP",
    title: researchNarrative.gap.headline,
    body:
      "A useful framework has to combine heterogeneous evidence without hiding uncertainty or assuming that every available modality deserves equal influence.",
    note: "Missing evidence is not evidence of low anxiety.",
    details: researchNarrative.gap.challenges.slice(0, 4),
    scene: "gap",
  },
  {
    id: "abstract",
    number: "05",
    eyebrow: "RESEARCH ABSTRACT",
    title: "A multimodal framework for a more personalized, temporally aware view of anxiety vulnerability.",
    body: researchNarrative.hero.supporting,
    note: researchMeta.boundary,
    details: [
      { label: "SCOPE", value: "Young adults with anxiety disorders" },
      { label: "EVIDENCE", value: "Physiological · behavioural · clinical · contextual" },
      { label: "FOCUS", value: "Personalized vulnerability mapping and escalation assessment" },
      { label: "PRINCIPLE", value: "Reliability and uncertainty remain visible" },
    ],
    scene: "abstract",
  },
  {
    id: "purpose",
    number: "06",
    eyebrow: "WHAT THE RESEARCH AIMS TO FULFIL",
    title: "A research system that is useful because it knows what it does not know.",
    body:
      "The framework is designed to combine multiple forms of evidence while respecting personal baselines, data quality, different timescales and the limits of each model.",
    note: "Research and clinical decision support—not automated diagnosis.",
    details: [
      "Represent multiple timescales rather than one snapshot.",
      "Respect individual baselines and evidence quality.",
      "Keep unavailable or weak evidence visible as uncertainty.",
      "Keep current assessment distinct from future forecasting.",
    ],
    scene: "purpose",
  },
  {
    id: "question",
    number: "07",
    eyebrow: "THE RESEARCH QUESTION",
    title: researchNarrative.thesis.question,
    body:
      "The next sections show how the four research components, fusion logic and evaluation framework contribute to answering that question.",
    note: "NEXT · FOUR RESEARCH COMPONENTS",
    details: [],
    scene: "question",
  },
];
