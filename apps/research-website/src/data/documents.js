// Documents and milestones. Only list items that exist; set `url` once a file is
// published (put PDFs in public/documents/ and use "/documents/<file>.pdf").

import { components } from "./components.js";

export const documents = [
  ...components.map((component) => ({
    group: "Project proposals",
    type: "Proposal report",
    title: component.proposalTitle,
    meta: `${component.id} · ${component.owner} · March 2026`,
    url:
      component.owner === "Sendanayake H.D."
        ? "https://mysliit.sharepoint.com/sites/CDAPSubmissionCloud/2026RegCloud/Forms/AllItems.aspx?viewid=db9415e4%2Dec70%2D4c80%2Dac71%2Dbb61beebb336&id=%2Fsites%2FCDAPSubmissionCloud%2F2026RegCloud%2FR26%2DDS%2D012%2DStudents%2F1%2E%20Project%20Proposal%2FIndividual%20Reports%2FR26%2DDS%2D012%5FIT22107596%5FSendanayake%20H%20D%2Epdf&parent=%2Fsites%2FCDAPSubmissionCloud%2F2026RegCloud%2FR26%2DDS%2D012%2DStudents%2F1%2E%20Project%20Proposal%2FIndividual%20Reports"
        : component.owner === "Seneviratne K.A.U.A."
          ? "https://mysliit.sharepoint.com/sites/CDAPSubmissionCloud/2026RegCloud/Forms/AllItems.aspx?viewid=db9415e4%2Dec70%2D4c80%2Dac71%2Dbb61beebb336&id=%2Fsites%2FCDAPSubmissionCloud%2F2026RegCloud%2FR26%2DDS%2D012%2DStudents%2F1%2E%20Project%20Proposal%2FIndividual%20Reports%2FR26%2DDS%2D012%5FIT22093950%5FSeneviratne%20K%20A%20U%20A%2Epdf&parent=%2Fsites%2FCDAPSubmissionCloud%2F2026RegCloud%2FR26%2DDS%2D012%2DStudents%2F1%2E%20Project%20Proposal%2FIndividual%20Reports"
          : component.owner === "Kaushalya I.G.D."
            ? "https://mysliit.sharepoint.com/sites/CDAPSubmissionCloud/2026RegCloud/Forms/AllItems.aspx?viewid=db9415e4%2Dec70%2D4c80%2Dac71%2Dbb61beebb336&id=%2Fsites%2FCDAPSubmissionCloud%2F2026RegCloud%2FR26%2DDS%2D012%2DStudents%2F1%2E%20Project%20Proposal%2FIndividual%20Reports%2FR26%2DDS%2D012%5FIT22130648%5FKaushalya%20I%20G%20D%2Epdf&parent=%2Fsites%2FCDAPSubmissionCloud%2F2026RegCloud%2FR26%2DDS%2D012%2DStudents%2F1%2E%20Project%20Proposal%2FIndividual%20Reports"
            : component.owner === "Layathma B.M.A.S."
              ? "https://mysliit.sharepoint.com/:b:/s/CDAPSubmissionCloud/IQA4fMHw38dITIa0c2ui1f1xAf36Ajem9_lXL4Cm5_9YcV4?e=b6H0U2"
              : null,
  })),
  {
    group: "Progress Presentation 1",
    type: "Presentation slides",
    title: "R26-DS-012 progress presentation 1",
    meta: "March 2026",
    url: "https://mysliit.sharepoint.com/:p:/r/sites/CDAPSubmissionCloud/_layouts/15/Doc.aspx?sourcedoc=%7B2BF82E73-41A4-4E6A-BE72-827E8C072C2B%7D&file=Progress%20Presentation%2001.pptx&action=edit&mobileredirect=true&wdwpf=doclib-t",
  },
  {
    group: "Progress Presentation 2",
    type: "Presentation slides",
    title: "R26-DS-012 progress presentation 2",
    meta: "June 2026",
    url: "https://mysliit.sharepoint.com/:p:/r/sites/CDAPSubmissionCloud/_layouts/15/Doc.aspx?sourcedoc=%7BD69672E9-7435-4131-93E9-DA031A9A71DF%7D&file=R26-DS-012_PP2.pptx&action=edit&mobileredirect=true&wdwpf=doclib-t",
  },

  {
    group: "Final Presentation",
    type: "Presentation slides",
    title: "R26-DS-012 final presentation",
    meta: "September 2026",
    url: "https://mysliit.sharepoint.com/sites/CDAPSubmissionCloud/2026RegCloud/Forms/AllItems.aspx?id=%2Fsites%2FCDAPSubmissionCloud%2F2026RegCloud%2FR26%2DDS%2D012%2DStudents%2F5%2E%20Final%20Report%20%26%20Presentation%2FFinal%20Presentation%20PPT&viewid=db9415e4%2Dec70%2D4c80%2Dac71%2Dbb61beebb336",
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
    title: "Component 1 physiological evaluation reported",
    detail:
      "Corrected 13-participant WESAD LOSO evaluation and external AffectiveROAD evaluation, with direct +5 and +10 minute Ridge forecasting.",
  },
  {
    date: "2026",
    title: "Final component evaluations and integration",
    detail:
      "C2 final v8 study, publication-clean TC-WPN benchmark, and reliability-weighted fusion with CARE-AnxRAG.",
  },
];
