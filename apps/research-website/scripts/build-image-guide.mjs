// Regenerates IMAGES.md from src/data/images.js. Run: npm run images:guide
import fs from "node:fs";
import { images, imageStyle } from "../src/data/images.js";

const slots = Object.entries(images);
const generated = slots.filter(([, slot]) => !slot.prompt.includes("Do not generate"));
const real = slots.filter(([, slot]) => slot.prompt.includes("Do not generate"));

const lines = [
  "# Image guide",
  "",
  "Every image on the site has a fixed slot. Save a file at the path shown (inside `public/`) and it replaces the placeholder automatically; no code changes are needed. While a file is missing, `npm run dev` shows the expected path and size on the placeholder.",
  "",
  "## General rules",
  "",
  "- **Format:** `.webp` for illustrations (quality 80 to 85), `.jpg` for portraits. Keep each file under about 400 KB.",
  "- **Size:** export at the pixel size listed; the aspect ratio must match or the image is cropped from the centre.",
  "- **Style prefix for ChatGPT:** start every prompt with the style line below so all images match the landing page.",
  "- **Accuracy:** images are illustrative. Do not add charts, numbers, labels or text inside the images; the site states results in text.",
  "- **People:** use real, consented photos for the team. Do not generate faces for real people.",
  "",
  "> **Style line:** " + imageStyle,
  "",
  "## Images to generate",
  "",
  "| File (in `public/`) | Size | Ratio | Where it appears |",
  "|---|---|---|---|",
  ...generated.map(([, s]) => `| \`${s.path}\` | ${s.size} | ${s.ratio.replace(/ /g, "")} | ${s.usedOn} |`),
  "",
  "### Prompts",
  "",
  ...generated.flatMap(([name, s]) => [`**\`${s.path}\`** (${name})`, "", `> ${imageStyle} ${s.prompt}`, ""]),
  "## Real photos (do not generate)",
  "",
  "| File (in `public/`) | Size | Who |",
  "|---|---|---|",
  ...real.map(([, s]) => `| \`${s.path}\` | ${s.size} square | ${s.prompt.replace("Real portrait photo of ", "").replace(" Do not generate.", "")} |`),
  "",
  "Crop portraits square, with the face centred in the upper half; the site displays them in circles and cards.",
  "",
];

fs.writeFileSync(new URL("../IMAGES.md", import.meta.url), lines.join("\n"));
console.log(`IMAGES.md written with ${slots.length} slots.`);
