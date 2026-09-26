import test from "node:test";
import assert from "node:assert/strict";
import { components } from "../src/data/components.js";
import { images } from "../src/data/images.js";
import { supervisors, team } from "../src/data/team.js";
import { documents } from "../src/data/documents.js";

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
