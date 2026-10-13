// University submission budget: 20 MB of deployed site files, not source or Drive assets.
import fs from "node:fs";
import path from "node:path";
const dist = path.resolve("dist");
const limit = 20_000_000; // conservative decimal MB
if (!fs.existsSync(dist)) {
  console.error("Missing dist/: run the Vite build first.");
  process.exit(1);
}
function walk(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap(item => {
    const file = path.join(directory, item.name);
    return item.isDirectory() ? walk(file) :
      item.isFile() ? [{ file: path.relative(dist, file), bytes: fs.statSync(file).size }] : [];
  });
}
const files = walk(dist);
const bytes = files.reduce((sum, file) => sum + file.bytes, 0);
console.log(`Site size: ${bytes.toLocaleString()} / ${limit.toLocaleString()} bytes (${(bytes / 1e6).toFixed(2)} MB)`);
console.log("Largest files:", files.sort((a,b) => b.bytes-a.bytes).slice(0,5));
if (bytes > limit) {
  console.error("FAIL: site exceeds university's 20 MB disk space cap.");
  process.exit(1);
}
console.log("PASS: compiled site fits the 20 MB budget.");
