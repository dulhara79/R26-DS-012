# Archived: research website v1

This folder holds the previous version of the R26-DS-012 website (light glass
design with the looping hero video, immersive introduction chapters and the
older section components). It was replaced by the cinematic-scroll site in
`../../src` and is **not built or served**. Vite only compiles `src/`.

Contents:

| Path | What it was |
|---|---|
| `src/` | All v1 pages, components, data, hooks and CSS |
| `tests/` | v1 unit test for the intro chapter state helper |
| `scripts/validate-content.mjs` | v1 content/design guard (expected v1 file names and strings) |
| `public/media/` | v1 hero video and longitudinal context SVG |
| `tailwind.config.js`, `postcss.config.js`, `index.html` | v1 build configuration |

To restore v1, copy these files back over the project root and reinstate
`tailwindcss` in `postcss.config.js`.
