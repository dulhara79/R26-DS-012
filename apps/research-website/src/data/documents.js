// Documents and milestones. Only list items that exist; set `url` once a file is
// published (put PDFs in public/documents/ and use "/documents/<file>.pdf").

import { components } from "./components.js";

export const documents = [
  ...components.map((component) => ({
    group: "Project proposals",
    type: "Proposal report",
    title: component.proposalTitle,
    meta: `${component.id} · ${component.owner} · March 2026`,
    url: null,
  })),
  {
    group: "Source and artefacts",
    type: "Repository",
    title: "R26-DS-012 research repository",
    meta: "All four components, apps and CI workflows",
    url: "https://github.com/dulhara79/R26-DS-012",
  },
  {
    group: "Source and artefacts",
    type: "Diagram",
    title: "Project framework diagram",
    meta: "full.png in the research repository",
    url: "https://github.com/dulhara79/R26-DS-012/blob/main/full.png",
  },
  {
    group: "Source and artefacts",
    type: "Repository",
    title: "Research website source",
    meta: "This website",
    url: "https://github.com/dulhara79/anxiety_research_website",
  },
];

export const publicationNote =
  "No publications are listed yet. Entries are added here once a publication and its status are verified.";

// Scope changes between the March 2026 proposals and the current repository.
export const scopeChanges = [
  {
    id: "C2",
    before:
      "StudentLife, GPS/DBSCAN, hourly-risk and phenotype-deployment pipeline",
    after:
      "Final v8 GLOBEM spatio-temporal graph study with leakage-free, cross-cohort evaluation",
  },
  {
    id: "C3",
    before: '"Confidence" weight in the prototypical network',
    after:
      "Renamed prototype-consistency weight; it is not a calibrated confidence",
  },
  {
    id: "C4",
    before:
      "Continuously learning intervention engine (GBDT → KNN case-based reasoning)",
    after:
      "DCAR contextual prior, reliability-weighted fusion and CARE-AnxRAG decision support",
  },
];

// Add dated milestones here as they are confirmed (e.g. progress presentations).
export const milestones = [
  {
    date: "March 2026",
    title: "Project proposals submitted",
    detail:
      "Four individual proposal reports under the shared R26-DS-012 framework.",
  },
  {
    date: "April 2026",
    title: "Component 1 Phase 1 benchmarks completed",
    detail:
      "LSTM-AE, masked variant and forecasting module evaluated on four public datasets.",
  },
  {
    date: "2026",
    title: "Final component evaluations and integration",
    detail:
      "C2 final v8 study, publication-clean TC-WPN benchmark, and reliability-weighted fusion with CARE-AnxRAG.",
  },
];
