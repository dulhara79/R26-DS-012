import test from "node:test";
import assert from "node:assert/strict";
import { components } from "../src/data/components.js";
import { images, imageDriveIds, imageSrc } from "../src/data/images.js";
import { supervisors, team } from "../src/data/team.js";
import { documents, presentationSlides, presentationStages, publishedAdditionalDocuments, documentStages, milestones, assessmentWeights } from "../src/data/documents.js";



test("homepage gives C1–C4 equal visual treatment with correct evidence boundaries", async () => {
  const { homeEvidenceHighlights } = await import("../src/data/homeResearch.js");
  assert.deepEqual(Object.keys(homeEvidenceHighlights), ["C1", "C2", "C3", "C4"]);
  for (const component of components) {
    const highlight = homeEvidenceHighlights[component.id];
    assert.ok(highlight.title.length > 5);
    assert.ok(highlight.description.length > 25);
    assert.ok(highlight.imageAlt.length > 10);
    assert.ok(highlight.status.length > 15);
  }
  assert.match(homeEvidenceHighlights.C2.status, /0\.5205.*0\.0/);
  assert.match(homeEvidenceHighlights.C3.status, /leakage-audited verification/i);
  assert.match(homeEvidenceHighlights.C4.status, /not clinically validated/i);
  const { readFileSync } = await import("node:fs");
  const { fileURLToPath } = await import("node:url");
  const homepage = readFileSync(fileURLToPath(new URL("../src/pages/Home.jsx", import.meta.url)), "utf8");
  assert.match(homepage, /components\.map\(\(component, index\)/);
  assert.doesNotMatch(homepage, /home-research-integration|home-research-bridge|homeFusionHighlight/);
  assert.match(homepage, /name=\{component\.image\}/);
  assert.match(homepage, /to=\{"\/components\/" \+ component\.slug\}/);
});

test("homepage group context matches the current research export", async () => {
  const { homeContext, groupAbstract } = await import("../src/data/research.js");
  assert.ok(homeContext.lead.length > 60);
  assert.equal(homeContext.body.length, 2);
  assert.match(homeContext.body.join(" "), /not a clinically validated diagnostic/i);
  assert.equal(groupAbstract.paragraphs.length, 3);

  const { readFileSync } = await import("node:fs");
  const { fileURLToPath } = await import("node:url");
  const homeFile = fileURLToPath(new URL("../src/pages/Home.jsx", import.meta.url));
  const page = readFileSync(homeFile, "utf8");
  assert.match(page, /import \{ homeContext, timescales \} from ["']\.\.\/data\/research["']/);
  assert.doesNotMatch(page, /\boverview\.(?:lead|body)\b/);
});

test("group abstract preserves group-level evidence and clinical boundaries", async () => {
  const { groupAbstract } = await import("../src/data/research.js");
  assert.equal(groupAbstract.paragraphs.length, 3);
  const content = groupAbstract.paragraphs.join(" ");
  assert.match(content, /physiological|wearable/i);
  assert.match(content, /behavioural graph|passive-sensing graph/i);
  assert.match(content, /clinical-note classification/i);
  assert.match(content, /reliability-aware fusion/i);
  assert.match(content, /0\.5205/);
  assert.match(content, /p = 0\.255/);
  assert.match(content, /leakage risks/i);
  assert.match(content, /eight patients/);
  assert.match(content, /not a clinically validated diagnostic/i);
  assert.ok(groupAbstract.keywords.length >= 5);
});

test("group literature review is an integrated, properly cited synthesis", async () => {
  const { groupLiteratureReview } = await import("../src/data/research.js");
  const { methodologyReferences } = await import("../src/data/methodologyDetail.js");
  assert.equal(groupLiteratureReview.themes.length, 3);
  assert.deepEqual(groupLiteratureReview.themes.map((theme) => theme.id),
    ["observation", "validity", "integration"]);
  const allParagraphs = [
    ...groupLiteratureReview.introduction,
    ...groupLiteratureReview.themes.flatMap((theme) => theme.paragraphs),
  ];
  assert.ok(allParagraphs.every((paragraph) => paragraph.length > 110));
  assert.ok(allParagraphs.every((paragraph) => !/\bC[1-4]\b/.test(paragraph)),
    "Group literature should not be organized as individual component reports.");
  const citations = allParagraphs.flatMap((paragraph) =>
    [...paragraph.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])));
  assert.ok(citations.length >= 10);
  assert.ok(citations.every((number) =>
    methodologyReferences.some((reference) => reference.number === number)));
  assert.match(groupLiteratureReview.synthesis.boundary, /not establish prospective clinical effectiveness/i);
});

test("component slugs and ids are unique", () => {
  assert.equal(new Set(components.map((c) => c.slug)).size, components.length);
  assert.deepEqual(
    components.map((c) => c.id),
    ["C1", "C2", "C3", "C4"],
  );
});

test("C2 stays excluded from active fusion with its documented result", () => {
  const c2 = components.find((c) => c.id === "C2");
  assert.match(c2.fusionRole, /weight 0\.0/);
  assert.ok(c2.results.some(([value]) => value === "0.5205"));
});

test("every referenced image slot is registered with a path under /images", () => {
  const used = [
    ...components.map((c) => c.image),
    ...team.map((p) => p.photo),
    ...supervisors.map((p) => p.photo),
  ];
  for (const name of used) {
    assert.ok(images[name], `missing image slot ${name}`);
    assert.match(images[name].path, /^\/images\//);
  }
});

test("documents without a public file are marked unpublished, not linked", () => {
  for (const doc of documents) {
    assert.ok(doc.url === null || /^https?:\/\/|^\//.test(doc.url), doc.title);
  }
});

test("each declared image slot has its Drive file ID", () => {
  assert.deepEqual(Object.keys(imageDriveIds).sort(), Object.keys(images).sort());
  for (const name of Object.keys(images)) {
    assert.match(imageDriveIds[name], /^[A-Za-z0-9_-]{20,}$/);
    assert.match(imageSrc(name), /^https:\/\/drive\.google\.com\/thumbnail\?/);
  }
});

test("project's published documents use public Drive file URLs", () => {
  const publicationFiles = documents.filter(d => d.url && !["Repository","Diagram"].includes(d.type));
  assert.ok(publicationFiles.length >= 5);
  for (const item of publicationFiles) {
    assert.match(item.url, /^https:\/\/drive\.google\.com\/file\/d\//);
  }
  assert.match(presentationSlides.find(d => d.stage === "Final Presentation").url, /^https:\/\/drive\.google\.com\/file\/d\//);
});

test("presentation decks live only on Slides, with no duplicate document links", () => {
  assert.equal(presentationSlides.length, 4);
  assert.ok(documents.every((doc) => doc.type !== "Group presentation slides"));
  assert.ok(publishedAdditionalDocuments.every((doc) => doc.type !== "Group presentation slides"));
  assert.deepEqual(new Set(documents.map((doc) => doc.group)), new Set(documentStages));
  const documentUrls = new Set([...documents, ...publishedAdditionalDocuments].map((doc) => doc.url).filter(Boolean));
  for (const slide of presentationSlides) {
    assert.match(slide.url, /^https:\/\/drive\.google\.com\/file\/d\//);
    assert.equal(documentUrls.has(slide.url), false, "Presentation duplicated in Documents: " + slide.title);
  }
});

test("university navigation and milestones have honest placeholders", async () => {
  const { navigation } = await import("../src/data/site.js");
  assert.deepEqual(navigation.map((item) => item.label), ["Home","Domain","Milestones","Documents","Slides","About Us","Contact Us"]);
  assert.deepEqual(documentStages, ["TAF (Topic Assessment Form)", "Proposal"]);
  assert.deepEqual(presentationStages, ["Proposal", "Progress Presentation 1", "Progress Presentation 2", "Final Presentation"]);
  assert.deepEqual(presentationSlides.map((slide) => slide.stage), presentationStages);
  assert.deepEqual([...new Set(documents.map((item) => item.group))], documentStages);
  assert.deepEqual(milestones.map((item) => item.id), ["taf","proposal-submission","proposal-presentation","checklist-submission","pp1","pp2","draft-thesis","website-submission","final-checklist","final-presentation-viva","website-evaluation-logbook","research-paper-submission","final-thesis","publication-evidence"]);
  assert.equal(assessmentWeights.reduce((sum, item) => sum + item.percent, 0), 100);
  assert.equal(milestones.find((item) => item.id === "pp2").deadline, "31 August–2 September 2026");
  assert.ok(milestones.every((item) => Object.prototype.hasOwnProperty.call(item, "deadline")));
  assert.ok(milestones.every((item) => typeof item.detail === "string" && item.detail.length > 10));
});

test("researcher contacts and profiles match requested public details", () => {
  const expected = {
    IT22130648: ["dulhara.kaushalya79@gmail.com", "https://github.com/dulhara79"],
    IT22171542: ["senuvilayathma20@gmail.com", "https://github.com/SENUVI20"],
    IT22107596: ["hdsendanayake01@gmail.com", "https://github.com/DewduSendanayake"],
    IT22093950: ["uvisends24946@gmail.com", "https://github.com/UVINDUSEN"],
  };
  for (const person of team) assert.deepEqual([person.email, person.github], expected[person.id]);
  assert.deepEqual(supervisors.map(s => new URL(s.scholar).searchParams.get("user")), [
    "_I6MLzcAAAAJ", "_8a_hOEAAAAJ", "Pn5cuN0AAAAJ",
  ]);
});

test("group methodology uses continuous IEEE citations and external published references", async () => {
  const { groupMethodology, methodologyReferences } = await import("../src/data/methodologyDetail.js");
  assert.ok(groupMethodology.introduction.length > 100);
  assert.ok(groupMethodology.stages.length >= 5);
  assert.deepEqual(methodologyReferences.map(r => r.number), Array.from({length: methodologyReferences.length}, (_, i) => i + 1));
  const inline = groupMethodology.stages.flatMap(s => s.paragraphs).join(" ");
  const cited = [...inline.matchAll(/\[(\d+)\]/g)].map(match => Number(match[1]));
  assert.ok(cited.length >= 10);
  assert.ok(cited.every(n => methodologyReferences.some(r => r.number === n)));
  assert.ok(methodologyReferences.every(r => r.url.startsWith("https://")));
  assert.ok(methodologyReferences.every(r => !/thesis|draft/i.test(r.citation)));
  assert.match(inline, /behavioural model score from active fusion/);
});

test("replaced team photographs and final PPTX resolve to refreshed Drive assets", async () => {
  const { driveImageFiles, driveFinalPresentation } = await import("../src/data/driveAssets.generated.js");
  assert.equal(driveImageFiles["team-c1"].id, "1MAP0Alp3ucGCa-TyMGEt9fckz9kivMay");
  assert.equal(driveImageFiles["team-c2"].id, "1MAAxUv_hSmEY_UZTZKnuOdLfCu-Idvx1");
  assert.equal(driveFinalPresentation.id, "1TmXYrnLGQUQ-4KWPSDHCgw2rNxJqnOCr");
  const { imageSrc } = await import("../src/data/images.js");
  const url = imageSrc("team-c1");
  assert.match(url, /&v=/);
  assert.ok(url.includes(driveImageFiles["team-c1"].id));
});
