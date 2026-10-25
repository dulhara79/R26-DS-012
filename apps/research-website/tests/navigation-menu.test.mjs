import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { navigation } from "../src/data/site.js";
import { navigationMenus } from "../src/data/navigationMenus.js";
import { components } from "../src/data/components.js";
import { presentationSlides, documents, publishedAdditionalDocuments } from "../src/data/documents.js";

const menuKeys = ["/research", "/documents", "/slides", "/team"];

test("dropdowns retain the seven university-required primary links", () => {
  assert.deepEqual(navigation.map((item) => item.label), [
    "Home", "Domain", "Milestones", "Documents", "Slides", "About Us", "Contact Us",
  ]);
  assert.deepEqual(Object.keys(navigationMenus), menuKeys);
  assert.equal(navigationMenus["/research"].columns.length, 3);
  assert.deepEqual(navigation.filter((item) => !navigationMenus[item.to]).map((item) => item.to),
    ["/", "/milestones", "/contact"]);
});

test("dropdown targets point to existing site routes and sections", () => {
  const routes = new Set(navigation.map((item) => item.to));
  ["/components", "/system", "/findings", ...components.map((item) => "/components/" + item.slug)]
    .forEach((route) => routes.add(route));

  const sectionIds = {
    "/research": ["problem-title", "literature-title", "question-title", "method-title"],
    "/system": ["aura-title", "clinanx-title"],
    "/documents": ["docs-title", "cite-title",
      ...documents.map((_, i) => "document-" + i),
      ...publishedAdditionalDocuments.map((_, i) => "additional-document-" + i)],
    "/slides": ["slides-title", ...presentationSlides.map((_, i) => "slide-" + i)],
    "/team": ["team-title", "sup-title"],
  };

  for (const [parent, menu] of Object.entries(navigationMenus)) {
    assert.ok(menu.title && menu.title.length > 7, parent);
    assert.ok(menu.columns.length > 0, parent);
    for (const column of menu.columns) {
      assert.ok(column.label && column.links.length > 0, parent);
      const localTargets = new Set();
      for (const { label, to } of column.links) {
        assert.ok(label && to.startsWith("/"), to);
        assert.ok(!localTargets.has(to), "Repeated target: " + to);
        localTargets.add(to);
        const [pathname, hash] = to.split("#");
        assert.ok(routes.has(pathname), "Unknown route: " + pathname);
        if (hash) assert.ok(sectionIds[pathname]?.includes(hash), "Unknown section: " + to);
      }
    }
  }
});

test("navigation supports mouse, keyboard, escape, and mobile expand controls", () => {
  const file = fileURLToPath(new URL("../src/components/layout/Navbar.jsx", import.meta.url));
  const navSource = readFileSync(file, "utf8");
  assert.match(navSource, /onMouseEnter/);
  assert.match(navSource, /onMouseLeave/);
  assert.match(navSource, /onFocusCapture/);
  assert.match(navSource, /onBlurCapture/);
  assert.match(navSource, /Escape/);
  assert.match(navSource, /aria-expanded=\{expanded\}/);
  assert.match(navSource, /aria-controls=\{dropdownId\}/);
  assert.match(navSource, /location\.hash/);
  assert.match(navSource, /onClick=\{closeMenus\}/);
});
