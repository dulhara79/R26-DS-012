import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { navigation } from "../src/data/site.js";
import { componentBySlug } from "../src/data/components.js";

const read = (relative) => readFileSync(fileURLToPath(new URL(relative, import.meta.url)), "utf8");

test("unknown SPA routes and missing component slugs both reach the 404 page", () => {
  const app = read("../src/App.jsx");
  const componentDetail = read("../src/pages/ComponentDetail.jsx");
  assert.match(app, /<Route path="\*" element=\{<NotFound \/>\} \/>/);
  assert.match(componentDetail, /if \(!c\) return <NotFound \/>/);
  assert.equal(componentBySlug["not-a-real-component"], undefined);
});

test("404 screen provides an explanation and paths back to the published website", () => {
  const view = read("../src/pages/NotFound.jsx");
  assert.match(view, /useLocation/);
  assert.match(view, /Requested path:/);
  assert.match(view, /Return home/);
  assert.match(view, /Explore research/);
  assert.match(view, /Browse C1–C4/);
  assert.match(view, /navigation\.filter/);
  assert.ok(navigation.length >= 7);
});

test("document titles correctly identify unknown pages", () => {
  const manager = read("../src/components/layout/ScrollManager.jsx");
  assert.match(manager, /Not Found/);
  assert.match(manager, /componentBySlug/);
});
