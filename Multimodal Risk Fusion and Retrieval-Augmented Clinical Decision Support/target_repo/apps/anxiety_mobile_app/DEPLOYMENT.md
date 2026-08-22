# Deployment & CI/CD Guide

This document explains the automated pipeline and how to manage production builds for the Anxiety Research Mobile App.

## 1. Automated Pipeline

We use **GitHub Actions** to automate our workflow. There are two main workflows:

### A. Pull Request Validation (`validate_and_build.yml`)
- **Trigger**: Every time a Pull Request is opened or updated against the `main` branch.
- **Actions**: 
  - Runs `flutter analyze` to check for code quality issues.
  - Runs `flutter test` to execute widget and unit tests.
  - Builds a debug APK to ensure the code is buildable.

### B. Sync and Release (`sync_and_release.yml`)
- **Trigger**: Every time code is pushed or merged into the `main` branch.
- **Actions**:
  1. **Research Repo Sync**: Automatically clones `dulhara79/R26-DS-012` and copies the latest code into `apps/anxiety_mobile_app/`.
  2. **Production Build**: Builds a release APK with obfuscation and debug info splitting.
  3. **Artifact Upload**: The resulting `app-release.apk` is uploaded to the GitHub Actions run as an artifact.

---

## 2. Required GitHub Secrets

To make the pipeline work, you **must** add the following secrets to your GitHub repository (**Settings > Secrets and variables > Actions**):

| Secret Name | Description |
| :--- | :--- |
| `RESEARCH_REPO_PAT` | A Personal Access Token (PAT) with `repo` scope to allow pushing to the research repository. |
| `SCRIPT_URL` | The Google Apps Script URL used for data collection. |
| `AUTH_TOKEN` | The authentication token for the backend API. |

---

## 3. Production App Signing

To build a signed APK for distribution (e.g., via Play Store or manual install), you need a keystore.

### Local Setup
1. Create a file named `android/key.properties` (this file is ignored by Git).
2. Follow the template in `android/key.properties.example`.
3. Place your `.jks` or `.keystore` file in `android/app/`.

### CI Setup (Optional)
If you want the GitHub Action to produce a **signed** APK, you will need to:
1. Encode your keystore file to Base64.
2. Add the Base64 string and credentials to GitHub Secrets.
3. Update the `sync_and_release.yml` to decode and use these secrets.

---

## 4. Manual Syncing
If you ever need to manually sync the code to the research repo:
1. Ensure you have the research repo cloned locally.
2. Copy the contents of this repo (excluding `.git`) to the `apps/anxiety_mobile_app/` directory in the research repo.
3. Commit and push from the research repo.
