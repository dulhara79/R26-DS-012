// Nested navigation uses only existing routes and section anchors.
// Keep the seven university-required top-level labels in data/site.js.
export const navigationMenus = {
  "/research": {
    title: "Explore the research domain",
    description: "Research context, individual components and evidence.",
    columns: [
      {
        label: "Research",
        links: [
          { label: "Domain overview", to: "/research" },
          { label: "Research problem", to: "/research#problem-title" },
          { label: "Literature review", to: "/research#literature-title" },
          { label: "Objectives", to: "/research#question-title" },
          { label: "Methodology", to: "/research#method-title" },
        ],
      },
      {
        label: "Components",
        links: [
          { label: "All four components", to: "/components" },
          { label: "C1 · Physiology", to: "/components/wearable-forecasting" },
          { label: "C2 · Behaviour", to: "/components/behavioural-graphs" },
          { label: "C3 · Clinical NLP", to: "/components/clinical-nlp" },
          { label: "C4 · Fusion & RAG", to: "/components/fusion-and-rag" },
        ],
      },
      {
        label: "System & evidence",
        links: [
          { label: "System architecture", to: "/system" },
          { label: "Aura participant app", to: "/system#aura-title" },
          { label: "ClinAnx clinician app", to: "/system#clinanx-title" },
          { label: "Research findings", to: "/findings" },
        ],
      },
    ],
    footer: { label: "Explore the full domain", to: "/research" },
  },
  "/documents": {
    title: "Research documents",
    columns: [
      {
        label: "Browse records",
        links: [
          { label: "All documents", to: "/documents#docs-title" },
          { label: "Topic Assessment Form", to: "/documents#document-0" },
          { label: "Proposal reports", to: "/documents#document-1" },
          { label: "Individual report drafts", to: "/documents#additional-document-4" },
          { label: "Group report draft", to: "/documents#additional-document-8" },
          { label: "Citation", to: "/documents#cite-title" },
        ],
      },
    ],
  },
  "/slides": {
    title: "Presentation archive",
    columns: [
      {
        label: "Group slide decks",
        links: [
          { label: "Presentation archive", to: "/slides#slides-title" },
          { label: "Proposal", to: "/slides#slide-0" },
          { label: "Progress Presentation 1", to: "/slides#slide-1" },
          { label: "Progress Presentation 2", to: "/slides#slide-2" },
          { label: "Final Presentation", to: "/slides#slide-3" },
        ],
      },
    ],
  },
  "/team": {
    title: "About the research team",
    columns: [
      {
        label: "People",
        links: [
          { label: "Team overview", to: "/team" },
          { label: "Student researchers", to: "/team#team-title" },
          { label: "Academic & clinical supervisors", to: "/team#sup-title" },
        ],
      },
    ],
  },
};
