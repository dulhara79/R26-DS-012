import test from "node:test";
import assert from "node:assert/strict";
import { searchCategories, searchIndex, searchSite, normalizeSearch } from "../src/data/search.js";
import { documentStages, documents, presentationSlides, publishedAdditionalDocuments } from "../src/data/documents.js";
import { components } from "../src/data/components.js";

test("search index contains all public document records and components", () => {
  const indexedDocuments = searchIndex.filter(item => item.category === "Document");
  assert.equal(indexedDocuments.length, documents.length + publishedAdditionalDocuments.length);
  for (const component of components) {
    assert.ok(searchIndex.some(item => item.category === "Component" && item.href === "/components/" + component.slug));
  }
  assert.deepEqual(documentStages, ["TAF (Topic Assessment Form)", "Proposal"]);
  assert.equal(searchIndex.filter(item => item.category === "Presentation").length, presentationSlides.length);
});

test("document result deep links point to the matching rendered list row", () => {
  for (let i = 0; i < publishedAdditionalDocuments.length; i++) {
    assert.ok(searchIndex.some(item => item.category === "Document" && item.href === "/documents#additional-document-" + i));
  }
  for (let i = 0; i < documents.length; i++) {
    assert.ok(searchIndex.some(item => item.category === "Document" && item.href === "/documents#document-" + i));
  }
});

test("slide search results open their matching rows on Slides", () => {
  for (let i = 0; i < presentationSlides.length; i++) {
    const slide = presentationSlides[i];
    assert.ok(searchIndex.some((item) => item.category === "Presentation" && item.title === slide.title && item.href === "/slides#slide-" + i));
    assert.ok(!searchIndex.some((item) => item.category === "Document" && item.title === slide.title));
  }
});

test("domain search indexes the group abstract and current literature synthesis", () => {
  assert.ok(searchIndex.some(item => item.title === "Group research abstract" && item.href === "/research#overview-title"));
  assert.ok(searchIndex.some(item => item.title === "Group literature review" && item.href === "/research#literature-title"));
  assert.ok(searchSite("group abstract", { category: "Section" }).some(item => item.href === "/research#overview-title"));
  assert.ok(searchSite("clinical observation longitudinal", { category: "Section" }).length > 0);
});

test("queries match pages, documents, people, and component evidence", () => {
  assert.ok(searchSite("TC-WPN").some(item => item.href.includes("/components/clinical-nlp")));
  assert.ok(searchSite("progress presentation 1").some(item => item.category === "Presentation" && item.href.startsWith("/slides")));
  assert.ok(searchSite("TAF").some(item => item.category === "Document"));
  assert.ok(searchSite("0.5205").some(item => item.href.includes("behavioural") || item.href.includes("findings")));
  assert.ok(searchSite("supervisor").some(item => item.category === "Person" || item.href === "/team"));
  assert.ok(searchSite("CARE AnxRAG").length > 0);
  assert.deepEqual(searchSite(""), []);
});

test("full search category filters and Unicode normalization behave predictably", () => {
  assert.equal(normalizeSearch("  Clinical–NLP / TC-WPN  "), "clinical nlp tc wpn");
  assert.ok(searchSite("proposal", { category: "Document" }).every(item => item.category === "Document"));
  assert.ok(searchCategories.includes("Section"));
  assert.ok(searchCategories.includes("Presentation"));
  assert.ok(searchSite("a made up unfindable phrase that does not exist").length === 0);
});

test("search catalog contains only internal navigation targets", () => {
  for (const item of searchIndex) {
    assert.ok(item.href.startsWith("/"), item.href);
    assert.equal(item.href.includes("http"), false);
  }
});
