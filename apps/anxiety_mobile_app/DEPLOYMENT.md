# Deployment & CI/CD Guide

This document explains the automated pipeline and how to manage production builds for the Anxiety Research Mobile App.

## 1. Automated Pipeline

We use **GitHub Actions** to automate our workflow. There are two main workflows:

### A. Pull Request Validation (`validate_and_build.yml`)
- **Trigger**: Every time a Pull Request is opened or updated against the `main` branch.
- **Actions** (Flutter 3.47.5):
  - Runs `flutter analyze` to check for code quality issues.
  - Runs `flutter test` to execute widget and unit tests.
  - Runs formatting and analysis on changed Dart files, all Flutter tests, and a debug Android APK build.

### B. Sync and Release (`sync_and_release.yml`)
- **Trigger**: Every time code is pushed or merged into the `main` branch.
- **Actions**:
  1. **Research Repo Sync**: Automatically clones `dulhara79/R26-DS-012` and copies the latest code into `apps/anxiety_mobile_app/`.
  2. **Validation Build**: Builds a debug APK without production configuration.
  3. **Production Build**: Requires signing and `BACKEND_BASE`; fails the main-branch release job when absent. Builds signed release APK and AAB with obfuscation and debug info splitting.

---

## 2. Required GitHub Secrets

To make the pipeline work, you **must** add the following secrets to your GitHub repository (**Settings > Secrets and variables > Actions**):

| Secret Name | Description |
| :--- | :--- |
| `RESEARCH_REPO_PAT` | A Personal Access Token (PAT) with `repo` scope to allow pushing to the research repository. |
| `BACKEND_BASE` | HTTPS URL of the Central Backend (required for production builds). |
| `KEYSTORE_BASE64`, `KEYSTORE_PASSWORD`, `KEY_PASSWORD`, `KEY_ALIAS` | Android release signing material (all required for production artifacts). |

---

## 3. Production App Signing

To build a signed APK for distribution (e.g., via Play Store or manual install), you need a keystore.

### Local Setup
1. Create a file named `android/key.properties` (this file is ignored by Git).
2. Follow the template in `android/key.properties.example`.
3. Place your `.jks` or `.keystore` file in `android/app/`.

### CI Setup
For GitHub Actions to produce a signed release, you need to:
1. Encode your keystore file to Base64.
2. Add the Base64 string and credentials to GitHub Secrets.
3. Set `BACKEND_BASE` as a GitHub Actions secret. The workflow decodes signing material and fails the release job if signing or backend configuration is absent.

An Android release build uses the `release` signing config. Keep `android/key.properties` and keystores outside Git. Review the patient and clinician flow against a staged backend before distributing a build. Removing a published token from the source tree does not revoke it; rotate it in the deployed Apps Script properties.

---

## 4. Manual Syncing
If you ever need to manually sync the code to the research repo:
1. Ensure you have the research repo cloned locally.
2. Copy the contents of this repo (excluding `.git`) to the `apps/anxiety_mobile_app/` directory in the research repo.
3. Commit and push from the research repo.
