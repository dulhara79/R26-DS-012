import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve('.');
const src = path.join(root, 'src');
const files = [];

function collect(target) {
  if (!fs.existsSync(target)) return;
  const stat = fs.statSync(target);
  if (stat.isDirectory()) {
    for (const name of fs.readdirSync(target)) collect(path.join(target, name));
    return;
  }
  if (/\.(js|jsx|css|html|svg)$/.test(target)) files.push(target);
}

collect(src);
collect(path.resolve('public/media/context-longitudinal-field.svg'));
collect(path.resolve('tailwind.config.js'));
collect(path.resolve('index.html'));
const text = files.map((file) => fs.readFileSync(file, 'utf8')).join('\n');

const requiredFiles = [
  'src/data/components.js',
  'src/data/literature.js',
  'src/data/findings.js',
  'src/data/documents.js',
  'src/data/publications.js',
  'src/data/timeline.js',
  'src/data/people.js',
  'src/data/architecture.js',
  'src/components/hero/ResearchHero.jsx',
  'src/components/hero/HeroMedia.jsx',
  'src/components/hero/AnimatedHeading.jsx',
  'src/components/story/AnxietyContext.jsx',
  'src/components/story/ObservationTimeline.jsx',
  'src/components/story/SnapshotProblem.jsx',
  'src/components/story/LiteratureLandscape.jsx',
  'src/components/story/ResearchGap.jsx',
  'src/components/story/ResearchThesis.jsx',
  'src/components/research/ModalityOverview.jsx',
  'src/components/research/ComponentsStory.jsx',
  'src/components/research/FusionStory.jsx',
  'src/components/research/ArchitectureStory.jsx',
  'src/components/research/CurrentForecastSplit.jsx',
  'src/components/evidence/FindingsPreview.jsx',
  'src/components/documents/DocumentCollection.jsx',
  'src/components/team/ResearchTeam.jsx',
  'src/components/motion/MotionReveal.jsx',
  'src/components/motion/TextReveal.jsx',
  'src/hooks/useSectionNavigation.js',
  'public/media/context-longitudinal-field.svg',
];

const absentFiles = requiredFiles.filter((file) => !fs.existsSync(path.resolve(file)));
const forbidden = [
  'images.unsplash.com', 'pexels.com', 'Plus Jakarta Sans', 'Outfit', 'JetBrains Mono',
  'Adaptive Intervention Engine', 'KNN BallTree', 'grad-multi', 'float-badge', 'ring-glow',
  'glow-blue', 'glow-teal', 'glow-violet', 'spin-slow', 'pulse-soft',
  'flow.google.com/project/', 'flow.google.com/shared/video/',
  '!failed && !reduced && <video',
  'loop={!reduced}',
  'const SEAMLESS_LOOP_FADE_SECONDS = 1\n',
  'viewport={{ once: true',
];
const required = [
  'Understanding anxiety beyond a single moment.',
  'Anxiety is not a single moment.',
  'What we can observe is only part of the picture.',
  'GLOBEM', 'CARE-AnxRAG', '0.5205', 'TC-WPN', 'Active fusion weight 0.0',
  'Clinical NLP signal', 'not overall patient risk', 'insufficient evidence',
  'CURRENT', 'FORWARD', '/media/research-hero.mp4', 'prefers-reduced-motion',
  'whileHover', 'whileInView', 'layoutId="nav-active-indicator"', 'scrollIntoView',
  'data-media-state', 'onLoadedData',
  'SEAMLESS_LOOP_FADE_SECONDS', 'motion.video', 'onTimeUpdate', 'seamless-video-layer',
  'activeLayer', 'transitioningRef',
  'HERO_PLAYBACK_RATE = 0.7', 'LOOP_START_SECONDS', 'LOOP_END_TRIM_SECONDS',
  'video.playbackRate = HERO_PLAYBACK_RATE', 'fadeMediaSeconds', 'loopEnd',
  'hero-heading-safe', 'nav-shell--compact', 'SCROLL_COMPACT_THRESHOLD',
  'useScroll', 'useMotionValueEvent', 'layout="position"',
  'context-longitudinal-art', 'contextWaveGradient', 'context-observation-node',
  'context-visual-wrap', 'context-wave-path',
  'context-section-endcap', 'context-art-anchor', 'context-visual-wrap{position:absolute',
  'text-reveal--eyebrow', 'text-reveal--heading', 'text-reveal--body',
  'viewport={{ once: false', 'data-text-reveal',
];

const bad = forbidden.filter((term) => text.includes(term));
const missing = required.filter((term) => !text.includes(term));

if (absentFiles.length || bad.length || missing.length) {
  console.error('Content/design validation failed', { absentFiles, bad, missing });
  process.exit(1);
}

console.log('Research content/design validation passed.');
