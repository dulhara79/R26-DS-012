// Every image slot on the site. Drop a file at `path` (under public/) and it
// replaces the placeholder automatically. IMAGES.md documents the same list.

export const imageStyle =
  "Soft cinematic editorial illustration at dawn. Palette: deep teal #24524f, mist blue #a9c8d5, cream #fdf1e1, peach #e8b9a9, lavender #aaa7d6. Calm, hopeful, scientific. No text, no logos, no watermarks.";

export const images = {
  homeContext: {
    path: "/images/home/context.webp",
    ratio: "4 / 5",
    size: "1200 × 1500",
    usedOn: "Home · 'Why this research' section",
    prompt:
      "A young adult sitting by a window at dawn, seen from behind or in soft silhouette, calm posture, a light chest-strap wearable faintly visible under a shirt, a smartphone resting nearby. Gentle light, sense of time passing.",
  },
  c1: {
    path: "/images/components/c1-wearable.webp",
    ratio: "4 / 3",
    size: "1600 × 1200",
    usedOn: "Home component card, Components page, C1 page",
    prompt:
      "A minimalist fabric chest-strap wearable with a small sensor pod, floating on a soft background, a glowing ECG line and a breathing waveform flowing out of it.",
  },
  c2: {
    path: "/images/components/c2-behaviour.webp",
    ratio: "4 / 3",
    size: "1600 × 1200",
    usedOn: "Home component card, Components page, C2 page",
    prompt:
      "A smartphone lying flat, with an abstract glowing lattice graph rising above it: a grid of nodes arranged as days by four times of day, connected by thin lines.",
  },
  c3: {
    path: "/images/components/c3-clinical-notes.webp",
    ratio: "4 / 3",
    size: "1600 × 1200",
    usedOn: "Home component card, Components page, C3 page",
    prompt:
      "A few blank, unreadable clinical note pages dissolving into points of light that gather into two separate soft clusters, representing few-shot prototypes. No readable words.",
  },
  c4: {
    path: "/images/components/c4-fusion.webp",
    ratio: "4 / 3",
    size: "1600 × 1200",
    usedOn: "Home component card, Components page, C4 page",
    prompt:
      "Four coloured light streams (peach, lavender, cyan, cream) flowing into a glass prism; one lavender stream is faint and dotted. Beside it, an open book emitting small citation-like glowing tabs.",
  },
  bannerResearch: {
    path: "/images/banners/research.webp",
    ratio: "21 / 9",
    size: "2400 × 1030",
    usedOn: "Research page header background",
    prompt:
      "Wide landscape of soft rolling hills at dawn whose ridge lines subtly trace heartbeat waveforms, mist in the valleys, large empty sky for a title.",
  },
  bannerComponents: {
    path: "/images/banners/components.webp",
    ratio: "21 / 9",
    size: "2400 × 1030",
    usedOn: "Components page header background",
    prompt:
      "Four soft light paths crossing a misty valley toward a single bridge at dawn, wide composition, empty sky at the top.",
  },
  bannerSystem: {
    path: "/images/banners/system.webp",
    ratio: "21 / 9",
    size: "2400 × 1030",
    usedOn: "System page header background",
    prompt:
      "An elegant stone arch bridge over calm water at dawn, faint data lines running along its deck, wide composition, empty sky.",
  },
  bannerFindings: {
    path: "/images/banners/findings.webp",
    ratio: "21 / 9",
    size: "2400 × 1030",
    usedOn: "Findings page header background",
    prompt:
      "Calm lake at dawn with measured concentric ripples spreading from one point, a sense of careful observation, wide composition, empty sky.",
  },
  bannerDocuments: {
    path: "/images/banners/documents.webp",
    ratio: "21 / 9",
    size: "2400 × 1030",
    usedOn: "Documents page header background",
    prompt:
      "A quiet reading desk by a large window at dawn with a stack of bound reports and loose pages, soft light, wide composition, empty wall space.",
  },
  bannerTeam: {
    path: "/images/banners/team.webp",
    ratio: "21 / 9",
    size: "2400 × 1030",
    usedOn: "Team page header background",
    prompt:
      "Seven soft lanterns glowing along a path through misty hills at dawn (four student lights, three guiding lights slightly higher), wide composition.",
  },
  bannerContact: {
    path: "/images/banners/contact.webp",
    ratio: "21 / 9",
    size: "2400 × 1030",
    usedOn: "Contact page header background",
    prompt:
      "An open wooden gate on a hillside path at dawn leading toward a quiet valley in soft mist, welcoming mood, wide composition, empty sky.",
  },
  appAura: {
    path: "/images/system/aura-app.webp",
    ratio: "9 / 16",
    size: "900 × 1600",
    usedOn: "System page · Aura participant app",
    prompt:
      "Prefer a real screenshot of the Aura Flutter app. If generating: a clean smartphone mockup showing a calm dashboard with a heart-rate line and a behavioural context card, no readable personal data.",
  },
  appClinAnx: {
    path: "/images/system/clinanx-console.webp",
    ratio: "4 / 3",
    size: "1600 × 1200",
    usedOn: "System page · ClinAnx clinician console",
    prompt:
      "Prefer a real screenshot of the ClinAnx app. If generating: a tablet mockup of a clinician console with a patient chart, three tier labels and an evidence panel, fictional placeholder data only.",
  },
  "team-c1": {
    path: "/images/team/sendanayake.jpg",
    ratio: "1 / 1",
    size: "800 × 800",
    usedOn: "Team page and home team strip",
    prompt: "Real portrait photo of Sendanayake H.D. Do not generate.",
  },
  "team-c2": {
    path: "/images/team/layathma.jpg",
    ratio: "1 / 1",
    size: "800 × 800",
    usedOn: "Team page and home team strip",
    prompt: "Real portrait photo of Layathma B.M.A.S. Do not generate.",
  },
  "team-c3": {
    path: "/images/team/kaushalya.jpg",
    ratio: "1 / 1",
    size: "800 × 800",
    usedOn: "Team page and home team strip",
    prompt: "Real portrait photo of Kaushalya I.G.D. Do not generate.",
  },
  "team-c4": {
    path: "/images/team/seneviratne.jpg",
    ratio: "1 / 1",
    size: "800 × 800",
    usedOn: "Team page and home team strip",
    prompt: "Real portrait photo of Seneviratne K.A.U.A. Do not generate.",
  },
  "sup-1": {
    path: "/images/team/thelijjagoda.jpg",
    ratio: "1 / 1",
    size: "800 × 800",
    usedOn: "Team page",
    prompt:
      "Real portrait photo of Prof. Samantha Thelijjagoda (with permission). Do not generate.",
  },
  "sup-2": {
    path: "/images/team/weerasinghe.jpg",
    ratio: "1 / 1",
    size: "800 × 800",
    usedOn: "Team page",
    prompt:
      "Real portrait photo of Dr. Mahima Weerasinghe (with permission). Do not generate.",
  },
  "sup-3": {
    path: "/images/team/suraweera.jpg",
    ratio: "1 / 1",
    size: "800 × 800",
    usedOn: "Team page",
    prompt:
      "Real portrait photo of Dr. Chathurie Suraweera (with permission). Do not generate.",
  },
};
