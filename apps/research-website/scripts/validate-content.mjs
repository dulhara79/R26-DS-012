// Content guard: blocks claims and legacy designs that the research repository
// no longer supports, and checks that IMAGES.md lists every image slot.
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(".");
const files = [];

function collect(target) {
  if (!fs.existsSync(target)) return;
  if (fs.statSync(target).isDirectory()) {
    for (const name of fs.readdirSync(target)) collect(path.join(target, name));
  } else if (/\.(js|jsx|css|html)$/.test(target)) {
    files.push(target);
  }
}

collect(path.join(root, "src"));
collect(path.join(root, "index.html"));
const text = files.map((file) => fs.readFileSync(file, "utf8")).join("\n");

const forbidden = [
  "Adaptive Intervention Engine",
  "KNN BallTree",
  "diagnoses anxiety",
  "clinically validated",
  "images.unsplash.com",
  "pexels.com",
];
const required = [
  "not a diagnostic",
  "insufficient evidence",
  "weight 0.0",
  "0.5205",
  "CARE-AnxRAG",
  "prefers-reduced-motion",
];

const bad = forbidden.filter((term) => text.includes(term));
const missing = required.filter((term) => !text.includes(term));

const { images } = await import(new URL("../src/data/images.js", import.meta.url));
const guide = fs.readFileSync(path.join(root, "IMAGES.md"), "utf8");
const undocumented = Object.values(images)
  .map((slot) => slot.path)
  .filter((imagePath) => !guide.includes(imagePath));

if (bad.length || missing.length || undocumented.length) {
  console.error("Content validation failed", { bad, missing, undocumented });
  process.exit(1);
}

console.log("Research content validation passed.");
