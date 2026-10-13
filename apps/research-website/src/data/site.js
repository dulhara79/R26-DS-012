// Project-level facts. Source: R26-DS-012 root README (github.com/dulhara79/R26-DS-012).

export const site = {
  id: "R26-DS-012",
  institution: "Sri Lanka Institute of Information Technology (SLIIT)",
  institutionShort: "SLIIT",
  department: "Department of Computer Science",
  degree: "B.Sc. (Hons) Information Technology Specialized in Data Science",
  year: "2026",
  title:
    "A Multimodal Digital Biomarker Framework for Personalized Vulnerability Mapping and Acute Escalation Forecasting in Young Adults with Anxiety Disorders",
  shortTitle: "Multimodal Anxiety Biomarkers",
  repository: "https://github.com/dulhara79/R26-DS-012",
  websiteRepository: "https://github.com/dulhara79/anxiety_research_website",
  issues: "https://github.com/dulhara79/R26-DS-012/issues",
  institutionUrl: "https://www.sliit.lk/",
  campus: "SLIIT Malabe Campus, New Kandy Road, Malabe, Sri Lanka",
  boundary:
    "Research and clinical decision support prototype. Not a diagnostic device; outputs must not be read as a diagnosis of an anxiety disorder.",
};

// Seven main tabs follow the suggested university Research Website structure.
// Detailed Components, System and Findings pages remain available via Home,
// research links, search and footer rather than being deleted.
export const navigation = [
  { to: "/", label: "Home" },
  { to: "/research", label: "Domain" },
  { to: "/milestones", label: "Milestones" },
  { to: "/documents", label: "Documents" },
  { to: "/slides", label: "Slides" },
  { to: "/team", label: "About Us" },
  { to: "/contact", label: "Contact Us" },
];
