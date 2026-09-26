import test from "node:test";
import assert from "node:assert/strict";
import {
  clamp,
  lerp,
  segmentInOut,
  smoothstep,
} from "../src/lib/scrollMath.js";

test("clamp bounds values to the unit range by default", () => {
  assert.equal(clamp(-1), 0);
  assert.equal(clamp(2), 1);
  assert.equal(clamp(0.4), 0.4);
});

test("smoothstep eases between its edges", () => {
  assert.equal(smoothstep(0, 10, -5), 0);
  assert.equal(smoothstep(0, 10, 5), 0.5);
  assert.equal(smoothstep(0, 10, 50), 1);
});

test("lerp interpolates linearly", () => {
  assert.equal(lerp(10, 20, 0.25), 12.5);
});

test("segmentInOut is fully active only between enter and exit", () => {
  assert.equal(segmentInOut(0, 560, 900, 1300, 1620).active, 0);
  assert.equal(segmentInOut(1100, 560, 900, 1300, 1620).active, 1);
  assert.equal(segmentInOut(2000, 560, 900, 1300, 1620).active, 0);
});
