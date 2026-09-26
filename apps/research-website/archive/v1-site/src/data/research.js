import { components } from "./components";
import { findings } from "./findings";
import { people, supervisors } from "./people";
export { components, findings, people, supervisors };

export const researchMeta = {
  id: "R26—DS—012",
  institution: "SLIIT",
  year: "2026",
  title:
    "A Multimodal Digital Biomarker Framework for Personalized Vulnerability Mapping and Acute Escalation Forecasting in Young Adults with Anxiety Disorders",
  boundary:
    "Research and clinical decision support prototype · not a diagnostic device",
};

export const researchNarrative = {
  hero: {
    headline: "Understanding anxiety beyond a single moment.",
    supporting:
      "A multimodal digital-biomarker research framework investigating how physiological, behavioural, clinical and contextual evidence can contribute to personalized vulnerability mapping and escalation assessment.",
  },
  context: {
    label: "01 / THE CONTEXT",
    headline: "Anxiety is not a single moment.",
    body: "In this research, anxiety-related change is considered across people and time. Relevant evidence may appear through physiology, everyday behaviour, clinical documentation and personal context rather than through one isolated observation.",
  },
  timescales: [
    ["NOW", "Physiological measurements"],
    ["DAYS / WEEKS", "Behavioural patterns"],
    ["CLINICAL ENCOUNTERS", "Documented clinical language"],
    ["BACKGROUND", "Contextual factors"],
  ],
  snapshot: {
    headline: "Important observations can happen between snapshots.",
    body: "Clinical encounters provide valuable assessments. The research question is whether additional digital evidence can help characterize change between encounters without replacing clinical judgment.",
  },
  gap: {
    headline:
      "The challenge is not collecting more signals. It is understanding how much each signal should be trusted.",
    challenges: [
      "Anxiety-related change is continuous and multimodal.",
      "People have different physiological baselines.",
      "Behavioural sensing may fail to generalize.",
      "Clinical NLP has limited labelled data and leakage risks.",
      "Modalities operate at different timescales.",
      "Unavailable information is common.",
      "Weak information should not be treated as certainty.",
    ],
  },
  thesis: {
    question:
      "Can heterogeneous digital biomarkers be combined responsibly to build a more personalized and temporally aware picture of anxiety vulnerability?",
  },
  interpretation: [
    "Personalization matters.",
    "Leakage-free validation matters.",
    "Complex models are not automatically superior.",
    "Modalities should not receive equal influence simply because they exist.",
    "Missing evidence is not evidence of low anxiety.",
    "Uncertainty must be represented explicitly.",
  ],
  methodology: [
    "Problem formulation",
    "Literature review",
    "Dataset / signal selection",
    "Component research",
    "Leakage-aware evaluation",
    "Integration",
    "Reliability-aware fusion",
    "System validation",
    "Research outputs",
  ],
  limitations: [
    "Research prototype; not a diagnostic device.",
    "Datasets do not represent every population.",
    "Modalities may be absent, stale or unreliable.",
    "Behavioural generalization remains limited under current evidence.",
    "Forecasting scope is defined and should not be exaggerated.",
    "Further external and prospective validation is required where applicable.",
  ],
};
