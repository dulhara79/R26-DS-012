# Hosting the research site under 20 MB

## University criteria

The supplied `Website.pdf`, page 16, specifies a **maximum 20 MB disk space** and lists **WordPress, HTML, CSS** as allowed technologies. React/Vite compatibility with the university host must be confirmed separately.

## Media

21 real image assets were found in the project's [shared Drive folder](https://drive.google.com/drive/folders/1rZO2JY0AXn-8hMcWATE3dAFqUXSGwaPd). All previously missing image slots now use Drive thumbnail URLs derived from the matching IDs in `src/data/images.js`. Source images are not copied into the site build.

Google Drive is not a dedicated image CDN, and thumbnail embedding is best-effort; public view permissions and unsigned browser playback must be verified. If a thumbnail cannot load, an image slot offers a link to the original Drive file. Several Drive items with `.webp` or `.jpg` names report actual `image/png` MIME types; renaming does not compress the image.

## Hero video

The existing **6,667,003 byte** MP4 stays in `public/media/research-hero.mp4` so the existing autoplay, crossfade and playback control work. Google Drive's file-preview URL is not a reliable raw MP4 source. The duplicate 6.67 MB archived copy was deleted from the Git tree. The hero poster is requested from Drive.

## Documents

The four proposal PDFs, proposal presentation, approved topic assessment form, Progress Presentation 1, and Progress Presentation 2 are linked to individual Drive files. The final presentation and final reports are **not yet available** in the folder, which only has instructions; do not describe them as published.

## Validation and submission

1. Run `cd research-website && npm ci && npm run build`. The production build invokes `npm run size:check`, which fails when `dist/` exceeds **20,000,000 bytes**.
2. Upload the contents of `dist/` only, not the source checkout, `node_modules/`, repository archives, or the Drive folders.
3. Open the deployed site in a signed-out browser and test each image, team portrait, hero video and document link. Drive permission metadata retrieved through the connector was insufficient to prove anonymous-reader access.
4. Confirm with the university whether a static React/Vite-generated HTML/CSS/JavaScript site is accepted, as the supplied handout does not list JavaScript expressly.

The archived MP4 removal reduces current checked-in file bytes, but the repository history and hosting usage are different metrics. The size script validates the actual compiled distribution instead of relying on GitHub's repository-size display.
