// Lightweight, browser-side index of the site's published pages, research data,
// component evidence and linked-document metadata. No remote search service.
import { navigation, site } from "./site.js";
import { components } from "./components.js";
import { documents, milestones, publishedAdditionalDocuments } from "./documents.js";
import {
  overview, problems, timescales, literatureThemes, researchGap,
  researchQuestion, objectives, methodology, techStack, commitments,
} from "./research.js";
import { architecture, aura, clinanx } from "./system.js";
import { groupMethodology, methodologyReferences } from "./methodologyDetail.js";
import { team, supervisors } from "./team.js";
import { cinematicIntro, cinematicPanels } from "./cinematicStory.js";

const excludedKeys = new Set(["url", "source", "profile", "image", "path", "prompt", "photo"]);
function plainText(value) {
  if (value == null) return "";
  if (typeof value === "string" || typeof value === "number") return String(value);
  if (Array.isArray(value)) return value.map(plainText).filter(Boolean).join(" · ");
  if (typeof value === "object") {
    return Object.entries(value)
      .filter(([key]) => !excludedKeys.has(key))
      .map(([, child]) => plainText(child)).filter(Boolean).join(" · ");
  }
  return "";
}

function record(category, title, href, description, content = "", keywords = "", fileUrl = null) {
  const detail = plainText(content);
  return {
    id: href + "::" + title,
    category,
    title,
    href,
    description,
    content: detail,
    keywords: plainText(keywords),
    fileUrl,
  };
}

const page = (title, href, description, content, keywords = "") =>
  record("Page", title, href, description, content, keywords);
const section = (title, href, description, content, keywords = "") =>
  record("Section", title, href, description, content, keywords);

export const searchIndex = [
  page("Overview", "/", "Project introduction and the multimodal anxiety research story.",
    [site.title, cinematicIntro, cinematicPanels, overview, problems, researchQuestion],
    "home research introduction research prototype"),
  page("Research", "/research", "Research domain, literature, objectives and methodology.",
    [overview, problems, timescales, literatureThemes, researchGap, objectives, methodology, groupMethodology, methodologyReferences, techStack],
    "domain literature survey research gap objectives technologies"),
  page("Components", "/components", "The four research components and how their signals connect.",
    components, "C1 C2 C3 C4"),
  page("System", "/system", "Integrated architecture, Aura participant app and ClinAnx clinician app.",
    [architecture, aura, clinanx], "mobile applications integration architecture"),
  page("Findings", "/findings", "Evaluations, evidence, limitations and component results.",
    components.map(c => [c.id, c.results, c.evaluation, c.status, c.baselines]),
    "results experiments AUROC F1 benchmark ablation evidence"),
  page("Milestones", "/milestones", "Assessment stages, recorded dates and pending marks.", milestones, "timeline viva final assessment marks"),
  page("Slides", "/slides", "Proposal, progress and final presentation slides.", documents.filter(d => d.type.includes("slides")), "downloads presentations pptx"),
  page("Documents", "/documents", "TAF, proposals, progress presentations, final presentation and citation.",
    [documents, publishedAdditionalDocuments, milestones], "downloads files reports slides PDF milestones"),
  page("Team", "/team", "The researchers and their academic and clinical supervisors.",
    [team, supervisors], "people authors researchers"),
  page("Contact", "/contact", "Project repository, SLIIT and research-team contact information.",
    [site.institution, site.department, team, supervisors],
    "email contact university faculty github"),

  section("Research challenge", "/#problem-title",
    "Why single observations do not describe anxiety across time.", problems,
    "anxiety problem motivation"),
  section("Research overview", "/research#overview-title",
    "Multiple signals and timescales.", [overview, timescales]),
  section("Research problem", "/research#problem-title",
    "The limits of episodic assessment and isolated signals.", problems),
  section("Literature survey", "/research#literature-title",
    "The physiological, behavioural, clinical and evidence-aware literature.", literatureThemes,
    "literature landscape related work review"),
  section("Research gap", "/research#gap-title",
    researchGap.headline, researchGap, "validation uncertainty reliability"),
  section("Research objectives", "/research#question-title",
    "Main research question and component-specific objectives.", [researchQuestion, objectives],
    "research question specific objectives"),
  section("Research methodology", "/research#method-title",
    "Research process, datasets, leakage controls and evaluation.", [groupMethodology, methodologyReferences],
    "method design approach dataset validation"),
  section("Methodology references", "/research#method-references-title", "Published IEEE-style external sources.", methodologyReferences),
  section("Technologies used", "/research#tech-title",
    "Technical tools and modelling technologies.", techStack),
  section("Fusion and eligibility", "/components#integration-title",
    "Why only usable, validated evidence contributes to fusion.",
    [components.map(c => c.fusionRole), commitments],
    "fusion gates C2 weight 0.0 missing evidence"),
  section("System architecture", "/system#system-architecture",
    "Components, central backend, fusion and application layers.", architecture,
    "data flow API architecture diagram"),
  section("Deployment architecture", "/system#arch-title",
    "Centralized model calls and system layers.", architecture),
  section("Aura participant application", "/system#aura-title",
    "Wearable sensing, questionnaires and passive Android collection.", aura,
    "patient participant mobile"),
  section("ClinAnx clinician console", "/system#clinanx-title",
    "Clinical notes, patient caseloads, fusion results and evidence.", clinanx,
    "doctor app clinical dashboard"),
  section("Privacy and research commitments", "/system#commit-title",
    "Safety, governance, privacy and research-use boundaries.", [aura.privacy, commitments]),
  section("Findings: interpretation", "/findings#fusion-finding",
    "Fusion gating and withholding unsupported results.", components.map(c => c.fusionRole),
    "validation insufficient evidence abstention"),
  section("Limitations", "/findings#limits-title",
    "Scope of the study and restrictions on interpreting predictions.",
    [components.map(c => c.status), commitments], "not diagnostic generalization"),
  section("Documents and downloads", "/documents#docs-title",
    "Assessment records from TAF to the final presentation.", documents,
    "download PDF PPT proposal"),
  section("Presentation archive", "/slides#slides-title", "Presentation slides and their publication status.", documents.filter(d => d.type.includes("slides"))),
  section("Research milestones", "/milestones#assessment-title",
    "Topic assessment, proposals, progress presentations and final presentation.", milestones,
    "timeline milestones assessment"),
  section("Cite this research", "/documents#cite-title",
    "Copy the research project BibTeX citation.", site.title,
    "citation bibtex authors 2026"),
  section("Research team", "/team#team-title",
    "Four student researchers, one research component each.", team,
    "people authors"),
  section("Academic and clinical supervisors", "/team#sup-title",
    "Project supervisors and clinical research guidance.", supervisors,
    "faculty professors"),
  section("Contact information", "/contact#contact-page",
    "Contact the research team and find institutional links.", [team, supervisors, site]),
  ...components.flatMap(c => {
    const root = "/components/" + c.slug;
    return [
      record("Component", c.id + " · " + c.title, root, c.tagline, c,
        [c.owner, c.modality, c.headline, c.studentId]),
      section(c.id + " · Core idea", root + "#idea-title", c.coreIdea, c),
      section(c.id + " · Datasets and signals", root + "#data-title",
        c.dataTitle || c.tagline,
        [c.benchmark, c.cohorts, c.nodeFeatures, c.target, c.dcar, c.policies, c.timescale],
        "data datasets cohort sensing"),
      section(c.id + " · Model and pipeline", root + "#pipeline-title",
        c.tagline, [c.pipeline, c.formula, c.terms],
        "methodology methods model architecture"),
      section(c.id + " · Evaluation results", root + "#results-title",
        c.resultsLabel || "Evaluation and experimental results.",
        [c.results, c.baselines, c.evaluation, c.loco],
        "auroc evaluation results metrics benchmarks"),
      section(c.id + " · Fusion status", root + "#status-title",
        c.fusionRole, [c.status, c.fusionWeight],
        "validation current status reliability"),
      ...(c.rag ? [section(c.id + " · CARE-AnxRAG", root + "#rag-title",
        "Evidence retrieval, relevance, provenance and abstention.", c.rag,
        "rag retrieval augmented generation")] : []),
      record("Finding", c.id + " · Findings and validation",
        "/findings#" + c.id + "-results",
        c.resultsLabel || "Component evaluation summary.",
        [c.results, c.evaluation, c.baselines, c.status],
        [c.title, c.modality, c.fusionRole]),
    ];
  }),
  ...publishedAdditionalDocuments.map((doc, i) => record("Document", doc.title, "/documents#additional-document-" + i, doc.meta, [doc.group, doc.type, doc.meta], "checklist report draft", doc.url)),
  ...documents.map((doc, i) => record(
    "Document", doc.title,
    "/documents#document-" + i,
    [doc.group, doc.type, doc.meta].filter(Boolean).join(" · "),
    [doc.group, doc.type, doc.meta],
    ["download", "file", "report", "slides", "TAF", "presentation"],
    doc.url && doc.url.startsWith("https://drive.google.com/file/d/") ? doc.url : null,
  )),
  ...team.map(person => record("Person", person.name, "/team#team-title",
    person.role, person, person.component)),
  ...supervisors.map(person => record("Person", person.name, "/team#sup-title",
    person.role, person, "supervisor academic clinical")),
];

export function normalizeSearch(value) {
  return String(value ?? "").normalize("NFKD")
    .toLocaleLowerCase("en")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[\u2010-\u2015]/g, "-")
    .replace(/[^a-z0-9]+/g, " ").trim().replace(/\s+/g, " ");
}

function excerpt(item, terms) {
  const summary = item.description || "";
  if (terms.some(term => normalizeSearch(summary).includes(term))) return summary.slice(0, 185);
  const content = item.content;
  const normalized = normalizeSearch(content);
  let offset = -1;
  for (const term of terms) {
    const found = normalized.indexOf(term);
    if (found >= 0 && (offset < 0 || found < offset)) offset = found;
  }
  // Use the authored summary rather than showing irrelevant or fragmented data.
  if (offset < 0) return summary.slice(0, 185);
  const start = Math.max(0, offset - 48);
  return (start ? "…" : "") + normalized.slice(start, start + 185).trim() +
    (normalized.length > start + 185 ? "…" : "");
}

export function searchSite(query, { category = "All", limit = 60 } = {}) {
  const phrase = normalizeSearch(query);
  if (!phrase) return [];
  const terms = [...new Set(phrase.split(" ").filter(Boolean))];
  const results = [];

  for (const item of searchIndex) {
    if (category !== "All" && item.category !== category) continue;
    const title = normalizeSearch(item.title);
    const description = normalizeSearch(item.description);
    const keywords = normalizeSearch(item.keywords);
    const content = normalizeSearch(item.content);
    const corpus = [title, description, keywords, content].join(" ");
    let coverage = 0, score = 0;
    for (const term of terms) {
      if (!corpus.includes(term)) continue;
      coverage += 1;
      score += title.includes(term) ? 28 : description.includes(term) ? 13 : keywords.includes(term) ? 9 : 3;
    }
    if (!coverage || coverage / terms.length < (terms.length <= 2 ? 1 : 0.7)) continue;
    if (title.includes(phrase)) score += 100;
    if (description.includes(phrase)) score += 35;
    if (keywords.includes(phrase)) score += 16;
    if (item.category === "Document" && /^(document|proposal|progress|taf|slide|report|presentation)/.test(phrase)) score += 8;
    score += (coverage / terms.length) * 35;
    results.push({ ...item, score, excerpt: excerpt(item, terms) });
  }
  return results
    .sort((a, b) => b.score - a.score || a.title.localeCompare(b.title))
    .slice(0, limit);
}

export const searchCategories = ["All", "Page", "Section", "Component", "Finding", "Document", "Person"];
export const suggestedSearches = ["Clinical NLP", "Results", "Proposal", "Fusion"];
