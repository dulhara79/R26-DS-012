import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  initialNavScroll, advanceNavScroll,
  NAV_TOP_BOUNDARY, NAV_HIDE_BOUNDARY, NAV_HIDE_TRAVEL, NAV_SHOW_TRAVEL,
} from "../src/lib/navScroll.js";

const advance = (state, ...positions) =>
  positions.reduce((previous, y) => advanceNavScroll(previous, y), state);

test("full at top, hidden on downward reading, compact on upward travel", () => {
  const top = initialNavScroll(0);
  assert.equal(top.mode, "full");
  const nearTop = advance(top, 20, NAV_TOP_BOUNDARY);
  assert.equal(nearTop.mode, "full");
  const beyondTop = advanceNavScroll(nearTop, NAV_HIDE_BOUNDARY - 1);
  assert.equal(beyondTop.mode, "compact");
  const hidden = advanceNavScroll(beyondTop, NAV_HIDE_BOUNDARY + NAV_HIDE_TRAVEL);
  assert.equal(hidden.mode, "hidden");
  const notEnoughUp = advanceNavScroll(hidden, hidden.y - (NAV_SHOW_TRAVEL - 2));
  assert.equal(notEnoughUp.mode, "hidden");
  const compact = advanceNavScroll(notEnoughUp, notEnoughUp.y - 3);
  assert.equal(compact.mode, "compact");
  assert.equal(advanceNavScroll(compact, 0).mode, "full");
});

test("small direction reversals do not flicker the navbar", () => {
  const hidden = advance(initialNavScroll(0), 130, 260);
  assert.equal(hidden.mode, "hidden");
  const tinyUp = advance(hidden, 258, 257);
  assert.equal(tinyUp.mode, "hidden");
  const tinyDown = advance(tinyUp, 259, 261);
  assert.equal(tinyDown.mode, "hidden");
  const visible = advance(tinyDown, 258, 254, 250);
  assert.equal(visible.mode, "compact");
  const tinyReverse = advance(visible, 251, 253);
  assert.equal(tinyReverse.mode, "compact");
  const hiddenAgain = advance(tinyReverse, 257, 263, 269);
  assert.equal(hiddenAgain.mode, "hidden");
  assert.ok(NAV_HIDE_TRAVEL > NAV_SHOW_TRAVEL);
});

test("page top and browser overscroll reset the state", () => {
  assert.equal(initialNavScroll(-120).mode, "full");
  assert.equal(initialNavScroll(300).mode, "compact");
  assert.equal(advanceNavScroll(initialNavScroll(500), -100).mode, "full");
  assert.equal(advanceNavScroll(initialNavScroll(500), Number.NaN).y, 0);
});

test("compact navigation keeps only the brand icon, with an accessible home link", () => {
  const nav = readFileSync(fileURLToPath(new URL("../src/components/layout/Navbar.jsx", import.meta.url)), "utf8");
  const css = readFileSync(fileURLToPath(new URL("../src/components/layout/nav-scroll.css", import.meta.url)), "utf8");
  assert.match(nav, /className="brand" to="\/" onClick=\{closeMenus\} aria-label="R26-DS-012 — Home"/);
  assert.match(nav, /<LogoMark \/>/);
  assert.match(nav, /<div className="brand-text">/);
  assert.match(css, /\.site-nav--compact \.brand-text \{ display: none; \}/);
  assert.doesNotMatch(css, /\.site-nav--full \.brand-text \{ display: none; \}/);
});

test("navbar keeps focus, dropdown, mobile and reduced-motion safeguards", () => {
  const nav = readFileSync(fileURLToPath(new URL("../src/components/layout/Navbar.jsx", import.meta.url)), "utf8");
  const css = readFileSync(fileURLToPath(new URL("../src/components/layout/nav-scroll.css", import.meta.url)), "utf8");
  assert.match(nav, /window\.addEventListener\("scroll", onScroll, \{ passive: true \}\)/);
  assert.match(nav, /requestAnimationFrame/);
  assert.match(nav, /const interacting = open \|\| expandedMenu !== null \|\| focusWithin/);
  assert.match(nav, /onFocusCapture=\{\(\) => setFocusWithin\(true\)\}/);
  assert.match(css, /\.site-nav--hidden:focus-within/);
  assert.match(css, /max-width: 1180px/);
  assert.match(css, /prefers-reduced-motion: reduce/);
});
