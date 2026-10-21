# Static Google Drive publishing

The website now links directly to the verified public Google Drive files. No Google Cloud service account or scheduled Drive API workflow is needed to display the existing images, presentations, checklists or draft reports.

The verified URLs are kept in `src/data/driveAssets.generated.js` (images and PPTX) and `src/data/documents.js` (project documents). These are static data; uploading a *new* file with a new Drive ID requires a code change and Vercel redeployment. Replacing file contents using Google Drive **Manage versions** keeps the same file ID, but Drive thumbnail/browser caches may delay updated image previews.

The former hourly Drive API polling workflow was disabled because service-account folder listing repeatedly returned HTTP 403. Re-enable only if authenticated automation is separately repaired and tested.
