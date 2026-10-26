import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { hashUrlForDirectPath, normalizeDirectPath } from "../src/lib/directPath.js";

test("bare invalid URLs reach the existing hash-router 404 path", () => {
  assert.equal(hashUrlForDirectPath({ pathname: "/gfdgfs" }), "/#/gfdgfs");
  assert.equal(hashUrlForDirectPath({ pathname: "/unknown/nested/path" }), "/#/unknown/nested/path");
  assert.equal(hashUrlForDirectPath({ pathname: "/components/not-real" }), "/#/components/not-real");
});

test("existing hash routes remain unchanged and direct known pages work", () => {
  assert.equal(hashUrlForDirectPath({ pathname: "/" }), null);
  assert.equal(hashUrlForDirectPath({ pathname: "/", hash: "#/gfdgfs" }), null);
  assert.equal(hashUrlForDirectPath({ pathname: "/index.html" }), null);
  assert.equal(hashUrlForDirectPath({ pathname: "/research" }), "/#/research");
  assert.equal(hashUrlForDirectPath({ pathname: "/slides" }), "/#/slides");
  assert.equal(hashUrlForDirectPath({
    pathname: "/research", search: "?src=mail", hash: "#literature-title",
  }), "/#/research?src=mail#literature-title");
  assert.equal(hashUrlForDirectPath({ pathname: "/research", hash: "#/slides" }), "/#/slides");
});

test("normalization replaces rather than pushes history entries", () => {
  const calls = [];
  const history = {
    state: { source: "test" },
    replaceState: (...args) => calls.push(args),
  };
  assert.equal(normalizeDirectPath({ pathname: "/gfdgfs" }, history), "/#/gfdgfs");
  assert.deepEqual(calls, [[{ source: "test" }, "", "/#/gfdgfs"]]);
  assert.equal(normalizeDirectPath({ pathname: "/", hash: "#/gfdgfs" }, history), null);
  assert.equal(calls.length, 1);
});

test("Vercel serves SPA document for unrecognized non-file routes", () => {
  for (const path of ["../../vercel.json", "../vercel.json"]) {
    const url = new URL(path, import.meta.url);
    const config = JSON.parse(readFileSync(fileURLToPath(url), "utf8"));
    assert.ok(config.rewrites.some(
      (route) => route.source === "/(.*)" && route.destination === "/index.html"
    ), path);
  }
  const app = readFileSync(fileURLToPath(new URL("../src/App.jsx", import.meta.url)), "utf8");
  const entry = readFileSync(fileURLToPath(new URL("../src/main.jsx", import.meta.url)), "utf8");
  assert.match(app, /<HashRouter>/);
  assert.match(app, /<Route path="\*" element=\{<NotFound \/>\} \/>/);
  assert.match(entry, /normalizeDirectPath\(window\.location, window\.history\)/);
});

test("bundled JavaScript and CSS use root-based URLs for nested direct paths", () => {
  const vite = readFileSync(fileURLToPath(new URL("../vite.config.js", import.meta.url)), "utf8");
  const html = readFileSync(fileURLToPath(new URL("../index.html", import.meta.url)), "utf8");
  // Exact strings avoid unnecessary regex escaping for basic asset paths.
  assert.ok(vite.includes("base: '/'"), "Vite base must use root-relative assets");
  assert.ok(html.includes('href="/favicon.svg"'), "Favicon must use root path");
  assert.ok(html.includes('href="/site.webmanifest"'), "Manifest must use root path");
});
