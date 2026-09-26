# Image guide

Every image on the site has a fixed slot. Save a file at the path shown (inside `public/`) and it replaces the placeholder automatically; no code changes are needed. While a file is missing, `npm run dev` shows the expected path and size on the placeholder.

## General rules

- **Format:** `.webp` for illustrations (quality 80 to 85), `.jpg` for portraits. Keep each file under about 400 KB.
- **Size:** export at the pixel size listed; the aspect ratio must match or the image is cropped from the centre.
- **Style prefix for ChatGPT:** start every prompt with the style line below so all images match the landing page.
- **Accuracy:** images are illustrative. Do not add charts, numbers, labels or text inside the images; the site states results in text.
- **People:** use real, consented photos for the team. Do not generate faces for real people.

> **Style line:** Soft cinematic editorial illustration at dawn. Palette: deep teal #24524f, mist blue #a9c8d5, cream #fdf1e1, peach #e8b9a9, lavender #aaa7d6. Calm, hopeful, scientific. No text, no logos, no watermarks.

## Images to generate

| File (in `public/`) | Size | Ratio | Where it appears |
|---|---|---|---|
| `/images/home/context.webp` | 1200 × 1500 | 4/5 | Home · 'Why this research' section |
| `/images/components/c1-wearable.webp` | 1600 × 1200 | 4/3 | Home component card, Components page, C1 page |
| `/images/components/c2-behaviour.webp` | 1600 × 1200 | 4/3 | Home component card, Components page, C2 page |
| `/images/components/c3-clinical-notes.webp` | 1600 × 1200 | 4/3 | Home component card, Components page, C3 page |
| `/images/components/c4-fusion.webp` | 1600 × 1200 | 4/3 | Home component card, Components page, C4 page |
| `/images/banners/research.webp` | 2400 × 1030 | 21/9 | Research page header background |
| `/images/banners/components.webp` | 2400 × 1030 | 21/9 | Components page header background |
| `/images/banners/system.webp` | 2400 × 1030 | 21/9 | System page header background |
| `/images/banners/findings.webp` | 2400 × 1030 | 21/9 | Findings page header background |
| `/images/banners/documents.webp` | 2400 × 1030 | 21/9 | Documents page header background |
| `/images/banners/team.webp` | 2400 × 1030 | 21/9 | Team page header background |
| `/images/banners/contact.webp` | 2400 × 1030 | 21/9 | Contact page header background |
| `/images/system/aura-app.webp` | 900 × 1600 | 9/16 | System page · Aura participant app |
| `/images/system/clinanx-console.webp` | 1600 × 1200 | 4/3 | System page · ClinAnx clinician console |

### Prompts

**`/images/home/context.webp`** (homeContext)

> Soft cinematic editorial illustration at dawn. Palette: deep teal #24524f, mist blue #a9c8d5, cream #fdf1e1, peach #e8b9a9, lavender #aaa7d6. Calm, hopeful, scientific. No text, no logos, no watermarks. A young adult sitting by a window at dawn, seen from behind or in soft silhouette, calm posture, a light chest-strap wearable faintly visible under a shirt, a smartphone resting nearby. Gentle light, sense of time passing.

**`/images/components/c1-wearable.webp`** (c1)

> Soft cinematic editorial illustration at dawn. Palette: deep teal #24524f, mist blue #a9c8d5, cream #fdf1e1, peach #e8b9a9, lavender #aaa7d6. Calm, hopeful, scientific. No text, no logos, no watermarks. A minimalist fabric chest-strap wearable with a small sensor pod, floating on a soft background, a glowing ECG line and a breathing waveform flowing out of it.

**`/images/components/c2-behaviour.webp`** (c2)

> Soft cinematic editorial illustration at dawn. Palette: deep teal #24524f, mist blue #a9c8d5, cream #fdf1e1, peach #e8b9a9, lavender #aaa7d6. Calm, hopeful, scientific. No text, no logos, no watermarks. A smartphone lying flat, with an abstract glowing lattice graph rising above it: a grid of nodes arranged as days by four times of day, connected by thin lines.

**`/images/components/c3-clinical-notes.webp`** (c3)

> Soft cinematic editorial illustration at dawn. Palette: deep teal #24524f, mist blue #a9c8d5, cream #fdf1e1, peach #e8b9a9, lavender #aaa7d6. Calm, hopeful, scientific. No text, no logos, no watermarks. A few blank, unreadable clinical note pages dissolving into points of light that gather into two separate soft clusters, representing few-shot prototypes. No readable words.

**`/images/components/c4-fusion.webp`** (c4)

> Soft cinematic editorial illustration at dawn. Palette: deep teal #24524f, mist blue #a9c8d5, cream #fdf1e1, peach #e8b9a9, lavender #aaa7d6. Calm, hopeful, scientific. No text, no logos, no watermarks. Four coloured light streams (peach, lavender, cyan, cream) flowing into a glass prism; one lavender stream is faint and dotted. Beside it, an open book emitting small citation-like glowing tabs.

**`/images/banners/research.webp`** (bannerResearch)

> Soft cinematic editorial illustration at dawn. Palette: deep teal #24524f, mist blue #a9c8d5, cream #fdf1e1, peach #e8b9a9, lavender #aaa7d6. Calm, hopeful, scientific. No text, no logos, no watermarks. Wide landscape of soft rolling hills at dawn whose ridge lines subtly trace heartbeat waveforms, mist in the valleys, large empty sky for a title.

**`/images/banners/components.webp`** (bannerComponents)

> Soft cinematic editorial illustration at dawn. Palette: deep teal #24524f, mist blue #a9c8d5, cream #fdf1e1, peach #e8b9a9, lavender #aaa7d6. Calm, hopeful, scientific. No text, no logos, no watermarks. Four soft light paths crossing a misty valley toward a single bridge at dawn, wide composition, empty sky at the top.

**`/images/banners/system.webp`** (bannerSystem)

> Soft cinematic editorial illustration at dawn. Palette: deep teal #24524f, mist blue #a9c8d5, cream #fdf1e1, peach #e8b9a9, lavender #aaa7d6. Calm, hopeful, scientific. No text, no logos, no watermarks. An elegant stone arch bridge over calm water at dawn, faint data lines running along its deck, wide composition, empty sky.

**`/images/banners/findings.webp`** (bannerFindings)

> Soft cinematic editorial illustration at dawn. Palette: deep teal #24524f, mist blue #a9c8d5, cream #fdf1e1, peach #e8b9a9, lavender #aaa7d6. Calm, hopeful, scientific. No text, no logos, no watermarks. Calm lake at dawn with measured concentric ripples spreading from one point, a sense of careful observation, wide composition, empty sky.

**`/images/banners/documents.webp`** (bannerDocuments)

> Soft cinematic editorial illustration at dawn. Palette: deep teal #24524f, mist blue #a9c8d5, cream #fdf1e1, peach #e8b9a9, lavender #aaa7d6. Calm, hopeful, scientific. No text, no logos, no watermarks. A quiet reading desk by a large window at dawn with a stack of bound reports and loose pages, soft light, wide composition, empty wall space.

**`/images/banners/team.webp`** (bannerTeam)

> Soft cinematic editorial illustration at dawn. Palette: deep teal #24524f, mist blue #a9c8d5, cream #fdf1e1, peach #e8b9a9, lavender #aaa7d6. Calm, hopeful, scientific. No text, no logos, no watermarks. Seven soft lanterns glowing along a path through misty hills at dawn (four student lights, three guiding lights slightly higher), wide composition.

**`/images/banners/contact.webp`** (bannerContact)

> Soft cinematic editorial illustration at dawn. Palette: deep teal #24524f, mist blue #a9c8d5, cream #fdf1e1, peach #e8b9a9, lavender #aaa7d6. Calm, hopeful, scientific. No text, no logos, no watermarks. An open wooden gate on a hillside path at dawn leading toward a quiet valley in soft mist, welcoming mood, wide composition, empty sky.

**`/images/system/aura-app.webp`** (appAura)

> Soft cinematic editorial illustration at dawn. Palette: deep teal #24524f, mist blue #a9c8d5, cream #fdf1e1, peach #e8b9a9, lavender #aaa7d6. Calm, hopeful, scientific. No text, no logos, no watermarks. Prefer a real screenshot of the Aura Flutter app. If generating: a clean smartphone mockup showing a calm dashboard with a heart-rate line and a behavioural context card, no readable personal data.

**`/images/system/clinanx-console.webp`** (appClinAnx)

> Soft cinematic editorial illustration at dawn. Palette: deep teal #24524f, mist blue #a9c8d5, cream #fdf1e1, peach #e8b9a9, lavender #aaa7d6. Calm, hopeful, scientific. No text, no logos, no watermarks. Prefer a real screenshot of the ClinAnx app. If generating: a tablet mockup of a clinician console with a patient chart, three tier labels and an evidence panel, fictional placeholder data only.

## Real photos (do not generate)

| File (in `public/`) | Size | Who |
|---|---|---|
| `/images/team/sendanayake.jpg` | 800 × 800 square | Sendanayake H.D. |
| `/images/team/layathma.jpg` | 800 × 800 square | Layathma B.M.A.S. |
| `/images/team/kaushalya.jpg` | 800 × 800 square | Kaushalya I.G.D. |
| `/images/team/seneviratne.jpg` | 800 × 800 square | Seneviratne K.A.U.A. |
| `/images/team/thelijjagoda.jpg` | 800 × 800 square | Prof. Samantha Thelijjagoda (with permission). |
| `/images/team/weerasinghe.jpg` | 800 × 800 square | Dr. Mahima Weerasinghe (with permission). |
| `/images/team/suraweera.jpg` | 800 × 800 square | Dr. Chathurie Suraweera (with permission). |

Crop portraits square, with the face centred in the upper half; the site displays them in circles and cards.
