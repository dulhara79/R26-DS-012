import fs from 'node:fs';
import path from 'node:path';

const scanTargets = [path.resolve('src'), path.resolve('tailwind.config.js')];
const files = [];

function collect(target) {
  const stat = fs.statSync(target);
  if (stat.isDirectory()) {
    for (const name of fs.readdirSync(target)) collect(path.join(target, name));
    return;
  }
  if (/\.(js|jsx|css)$/.test(target)) files.push(target);
}

scanTargets.forEach(collect);
const text = files.map((file) => fs.readFileSync(file, 'utf8')).join('\n');

const forbidden = [
  'images.unsplash.com',
  'pexels.com',
  'Plus Jakarta Sans',
  'Outfit',
  'JetBrains Mono',
  'Adaptive Intervention Engine',
  'KNN BallTree',
  'grad-multi',
  'float-badge',
  'ring-glow',
  'glow-blue',
  'glow-teal',
  'glow-violet',
  'spin-slow',
  'pulse-soft',
];

const required = ['GLOBEM', 'CARE-AnxRAG', '0.5205', 'TC-WPN', 'Active fusion weight 0.0'];
const bad = forbidden.filter((term) => text.includes(term));
const missing = required.filter((term) => !text.includes(term));

if (bad.length || missing.length) {
  console.error('Content/design validation failed', { bad, missing });
  process.exit(1);
}

console.log('Research content/design validation passed.');
