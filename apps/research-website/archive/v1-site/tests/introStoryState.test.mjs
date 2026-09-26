import test from "node:test";
import assert from "node:assert/strict";
import { chapterIndexFromProgress } from "../src/utils/introStoryState.js";

test("maps scroll progress to stable chapter indexes", () => {
  assert.equal(chapterIndexFromProgress(0, 7), 0);
  assert.equal(chapterIndexFromProgress(0.14, 7), 0);
  assert.equal(chapterIndexFromProgress(0.15, 7), 1);
  assert.equal(chapterIndexFromProgress(0.5, 7), 3);
  assert.equal(chapterIndexFromProgress(0.99, 7), 6);
  assert.equal(chapterIndexFromProgress(1, 7), 6);
});

test("clamps out-of-range progress and handles empty chapter counts", () => {
  assert.equal(chapterIndexFromProgress(-1, 7), 0);
  assert.equal(chapterIndexFromProgress(3, 7), 6);
  assert.equal(chapterIndexFromProgress(0.5, 0), 0);
});
