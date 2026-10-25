import test from "node:test";
import assert from "node:assert/strict";
import { navParentForPath } from "../src/lib/navParent.js";

test("top-level pages highlight their own navigation tab", () => {
  const direct = ["/", "/research", "/milestones", "/documents", "/slides", "/team", "/contact"];
  for (const path of direct) assert.equal(navParentForPath(path), path);
});

test("research detail and supporting pages highlight Domain", () => {
  const researchRoutes = [
    "/components",
    "/components/wearable-forecasting",
    "/components/behavioural-graphs",
    "/components/clinical-nlp",
    "/components/fusion-and-rag",
    "/system",
    "/findings",
    "/results",
  ];
  for (const route of researchRoutes) {
    assert.equal(navParentForPath(route), "/research", route);
  }
});

test("legacy Publications redirects are grouped under Documents", () => {
  assert.equal(navParentForPath("/publications"), "/documents");
});

test("nav mapping tolerates trailing slashes without marking the Home tab active", () => {
  assert.equal(navParentForPath("/components/clinical-nlp/"), "/research");
  assert.equal(navParentForPath("/research/"), "/research");
  assert.equal(navParentForPath("/"), "/");
  assert.equal(navParentForPath("/unknown"), "/unknown");
});
