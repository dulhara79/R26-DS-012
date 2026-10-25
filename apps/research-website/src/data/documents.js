import { driveFinalPresentation } from "./driveAssets.generated.js";
// Assessment-stage records. Only link exact files present in the shared
// project Drive folder; do not infer assessment dates from upload timestamps.
import { components } from "./components.js";

const driveFile = (id) => `https://drive.google.com/file/d/${id}/view`;
const proposalPdf = {
  C1: "1YMUy37Rg4co5sEGKX2eTDBJT4hJbrBN2",
  C2: "1QIXzKF3jqaezolvh8-jrY03ZYKJDNaee",
  C3: "1X35rkxxdf6gfhgKRdqVXt6EoNX80NY0d",
  C4: "1YbCryp0gpOo79RnUZbWPb5vuY1_ClPUe",
};

export const documentStages = [
  "TAF (Topic Assessment Form)",
  "Proposal",
];

export const presentationStages = [
  "Proposal",
  "Progress Presentation 1",
  "Progress Presentation 2",
  "Final Presentation",
];

export const documents = [
  {
    group: documentStages[0],
    type: "Topic Assessment Form",
    title: "R26-DS-012 Topic Assessment Form (TAF)",
    meta: "IT4010 Research Project · 2026",
    url: driveFile("1-Bdz4Qmgo4yoitX5S87QHpEKhIWpM-q1"),
  },
  ...components.map((component) => ({
    group: documentStages[1],
    type: "Individual proposal report",
    title: component.proposalTitle,
    meta: `${component.id} · ${component.owner} · March 2026`,
    url: proposalPdf[component.id] ? driveFile(proposalPdf[component.id]) : null,
  })),
];

// Presentation files are listed only on the dedicated Slides page.
export const presentationSlides = [
  {
    stage: presentationStages[0],
    type: "Group presentation slides",
    title: "R26-DS-012 proposal presentation",
    meta: "Archived slide deck (its printed date differs from the March proposal reports)",
    url: driveFile("1ApTHMyhwy_7qyqKdU8I2nJRCmJH1xrsz"),
  },
  {
    stage: presentationStages[1],
    type: "Group presentation slides",
    title: "R26-DS-012 progress presentation 1",
    meta: "Slides dated 11 May 2026 · Assessment date not independently verified",
    url: driveFile("1d0QkKTgOnnZ8BQhqOKrd6PtT-kcIIXmw"),
  },
  {
    stage: presentationStages[2],
    type: "Group presentation slides",
    title: "R26-DS-012 progress presentation 2",
    meta: "Slides available · Assessment date not verified",
    url: driveFile("1JXDryh9IUuU4EOtHUDQ9rJEmxSJaDY_v"),
  },
  {
    stage: presentationStages[3],
    type: "Group presentation slides",
    title: "R26-DS-012 final presentation",
    meta: "Final presentation PPTX · October 2026",
    url: driveFinalPresentation?.id ? driveFile(driveFinalPresentation.id) : null,
  },
];

export const publishedAdditionalDocuments = [
  { group: "Project Checklists", type: "Checklist evidence", title: "Research-paper master sheet", url: driveFile("1sk5KqAuHIwiRpk8eALrz1mOdozjvonxA"), meta: "Checklist 1 · Excel workbook" },
  { group: "Project Checklists", type: "Checklist evidence", title: "Data Analysis Report", url: driveFile("1tMMLm79HJ3BbSWL6t5fjge_kYmr0hYm7"), meta: "Checklist 2 · Word document" },
  { group: "Project Checklists", type: "Checklist evidence", title: "Ethical Analysis Report", url: driveFile("17VnKTTedkRSQy6hiGM2W8Ljnpqt7ywCe"), meta: "Checklist 3 · Word document" },
  { group: "Project Checklists", type: "Checklist evidence", title: "MLOps Report", url: driveFile("1mQpCBTro7Qdv_cVyWffBLmUcGgHTgi2z"), meta: "Checklist 4 · PDF" },
  { group: "Final Individual Reports", type: "Draft report", title: "IT22130648 draft report", url: driveFile("1MQtKVWmsgCmU1m5UGApDLUetgTTT3sjh"), meta: "October 2026 · Draft" },
  { group: "Final Individual Reports", type: "Draft report", title: "IT22171542 draft report", url: driveFile("1Zbiz_uGHt4PtcRo2MvsMp7fake8zRdeQ"), meta: "October 2026 · Draft" },
  { group: "Final Individual Reports", type: "Draft report", title: "IT22107596 draft report", url: driveFile("1beh1ZdJceaAtNsMFh5UOEK9LMPebfW2j"), meta: "October 2026 · Draft" },
  { group: "Final Individual Reports", type: "Draft report", title: "IT22093950 draft report", url: driveFile("1ldETZqTbNAZQySrahpnLDGHceE59UreU"), meta: "October 2026 · Draft" },
  { group: "Final Group Report", type: "Draft report", title: "R26-DS-012 group report draft", url: driveFile("1fbC7q_hvMMiJWNE7qjp19znk3b3kSHZF"), meta: "October 2026 · Draft" },
];

export const publicationNote =
  "Published papers will be listed only after their publication status and links are verified.";

// Sources for the records below:
// (1) "2026 Regular Batch - Timeline of the RP Project" (user-provided RP Team chart)
//     supplies DEADLINES; these are not evidence of individual completion.
// (2) IT4010 Research Project Module Outline, Version 2019-0, page 2,
//     supplies assessment WEIGHTS. Its continued validity for 2026 is unverified.
export const milestoneSources = {
  dates: "RP Team: 2026 Regular Batch — Timeline of the RP Project",
  weights: "IT4010 Research Project module outline, version 2019-0, page 2",
};

export const assessmentWeights = [
  { id: "proposal", title: "Proposal Presentation & Report", percent: 12 },
  { id: "pp1", title: "Progress Presentation I", percent: 15 },
  { id: "pp2", title: "Progress Presentation II", percent: 18 },
  { id: "final", title: "Final Presentation and VIVA", percent: 20 },
  { id: "final-report", title: "Final Report", percent: 19 },
  { id: "paper", title: "Research Paper (published)", percent: 10 },
  { id: "website", title: "Website", percent: 2 },
  { id: "logbook", title: "Research logbook, Status document 1 & 2", percent: 4 },
];

// The TAF is included because it is a project document; it does not appear
// in the supplied 2026 deadline chart, nor as a separately graded item
// in the 2019-0 module outline. Deadline strings below reproduce the chart.
export const milestones = [
  {
    id: "taf",
    title: "Topic Assessment Form (TAF)",
    deadline: null,
    from: null, to: null, weights: [],
    detail: "The project's Topic Assessment Form is available. The 2026 timeline chart does not give a separate TAF deadline or mark allocation.",
    href: "/documents#document-0",
  },
  {
    id: "proposal-submission",
    title: "Proposal Submission",
    deadline: "15 March 2026",
    from: "2026-03-15", to: "2026-03-15", weights: ["proposal"],
    detail: "Scheduled deadline for submission of the research proposal. The 12% shown in the module outline is shared with proposal presentation and report, not a separate submission allocation.",
    href: "/documents#docs-title",
  },
  {
    id: "proposal-presentation",
    title: "Proposal Presentation",
    deadline: "16–18 March 2026",
    from: "2026-03-16", to: "2026-03-18", weights: ["proposal"],
    detail: "Scheduled proposal presentation window. The available group presentation deck contains a later printed date, so it should not be represented as proof of a March presentation.",
    href: "/slides#slide-0",
  },
  {
    id: "checklist-submission",
    title: "Check List Submission",
    deadline: "13 May 2026",
    from: "2026-05-13", to: "2026-05-13", weights: [],
    detail: "Scheduled checklist submission. No separate percentage is specified for this step in the supplied module outline.",
    href: "/documents#docs-title",
  },
  {
    id: "pp1",
    title: "Progress Presentation 1",
    deadline: "11–13 May 2026",
    from: "2026-05-11", to: "2026-05-13", weights: ["pp1"],
    detail: "Scheduled first progress presentation window. The available slide deck shows 11 May 2026, a date within this period.",
    href: "/slides#slide-1",
  },
  {
    id: "pp2",
    title: "Progress Presentation 2",
    deadline: "31 August–2 September 2026",
    from: "2026-08-31", to: "2026-09-02", weights: ["pp2"],
    detail: "Scheduled second progress presentation window, highlighted in the supplied RP Team chart. The presentation deck is available in Drive.",
    href: "/slides#slide-2",
  },
  {
    id: "draft-thesis",
    title: "Draft Thesis Submission",
    deadline: "11 October 2026",
    from: "2026-10-11", to: "2026-10-11", weights: ["final-report"],
    detail: "Draft thesis submission deadline. The module outline's 19% refers to the Final Report category, not an independent draft-thesis mark.",
    href: "/documents#docs-title",
  },
  {
    id: "website-submission",
    title: "Website Submission",
    deadline: "11 October 2026",
    from: "2026-10-11", to: "2026-10-11", weights: ["website"],
    detail: "Scheduled website submission deadline. The module outline lists the Website as 2% of the course assessment.",
    href: "/",
  },
  {
    id: "final-checklist",
    title: "Final Check List Submission",
    deadline: "14 October 2026",
    from: "2026-10-14", to: "2026-10-14", weights: [],
    detail: "Scheduled final checklist submission. No separate percentage is listed for the checklist in the supplied module outline.",
    href: "/documents#docs-title",
  },
  {
    id: "final-presentation-viva",
    title: "Final Presentation and Viva",
    deadline: "19–21 October 2026",
    from: "2026-10-19", to: "2026-10-21", weights: ["final"],
    detail: "The 2026 schedule explicitly combines final presentation and viva within one window. The module outline also allocates a combined 20%; it does not split them into separate marks.",
    href: "/slides#slide-3",
  },
  {
    id: "website-evaluation-logbook",
    title: "Website Evaluation & Logbook Submission",
    deadline: "19–21 October 2026",
    from: "2026-10-19", to: "2026-10-21", weights: ["website", "logbook"],
    detail: "The 2026 schedule groups these events in one deadline window, but the supplied module outline lists Website (2%) and research logbook/status documents (4%) as different assessment components.",
    href: "/",
  },
  {
    id: "research-paper-submission",
    title: "Research Paper Submission",
    deadline: "23 October 2026",
    from: "2026-10-23", to: "2026-10-23", weights: ["paper"],
    detail: "Scheduled paper submission deadline. The older module outline calls the assessed category 'Research Paper (published)' (10%); submission alone should not be represented as verified publication.",
    href: "/documents#docs-title",
  },
  {
    id: "final-thesis",
    title: "Final Thesis Submission",
    deadline: "28 October 2026",
    from: "2026-10-28", to: "2026-10-28", weights: ["final-report"],
    detail: "Scheduled final thesis submission deadline. The module outline lists Final Report as 19%. Draft individual and group reports are linked on the Documents page.",
    href: "/documents#docs-title",
  },
  {
    id: "publication-evidence",
    title: "Research Paper Publication Evidence Submission",
    deadline: "1 December 2026",
    from: "2026-12-01", to: "2026-12-01", weights: ["paper"],
    detail: "Scheduled publication-evidence submission deadline. The module outline's 10% is for the published Research Paper category; acceptance or publication of the team's paper is not established by this timetable.",
    href: "/documents#docs-title",
  },
];
