import test from "node:test";
import assert from "node:assert/strict";
import { milestones } from "../src/data/documents.js";
import { JOURNEY_WIDTH, JOURNEY_START_Y, JOURNEY_ROW_HEIGHT, journeyHeight, journeyPath, journeyStops } from "../src/lib/journeyPath.js";

test("all 14 milestones retain their sequence and route stop coordinates", () => {
  const stops = journeyStops(milestones.length);
  assert.equal(stops.length, 14);
  assert.equal(stops[0].y, JOURNEY_START_Y);
  assert.equal(stops.at(-1).y, JOURNEY_START_Y + 13 * JOURNEY_ROW_HEIGHT);
  assert.ok(stops.every((item) => item.x > 0 && item.x < JOURNEY_WIDTH));
  assert.ok(stops.every((item, i) => item.side === (i % 2 === 0 ? "left" : "right")));
  assert.ok(journeyHeight(milestones.length) > stops.at(-1).y);
});

test("curved road passes every milestone without changing stored deadline information", () => {
  const stops = journeyStops(milestones.length);
  const d = journeyPath(stops);
  assert.equal(d.startsWith(`M ${stops[0].x} ${stops[0].y}`), true);
  assert.equal((d.match(/ C /g) || []).length, milestones.length - 1);
  assert.ok(d.endsWith(stops.at(-1).x + " " + stops.at(-1).y));
  assert.equal(milestones.find((item) => item.id === "publication-evidence").deadline, "1 December 2026");
});

test("mobile path is inside its dedicated narrow SVG strip", () => {
  const stops = journeyStops(milestones.length, true);
  assert.ok(stops.every((point) => point.x >= 0 && point.x <= 104));
  assert.equal(journeyPath([]), "");
  assert.match(journeyPath(stops), /^M /);
});

test("stage numbering and generous spacing remain consistent from 01 to 14", () => {
  assert.deepEqual(journeyStops(milestones.length).map(({ index }) => String(index + 1).padStart(2, "0")),
    Array.from({ length: 14 }, (_, i) => String(i + 1).padStart(2, "0")));
  assert.ok(JOURNEY_ROW_HEIGHT >= 180);
  assert.ok(JOURNEY_START_Y >= 120);
});
