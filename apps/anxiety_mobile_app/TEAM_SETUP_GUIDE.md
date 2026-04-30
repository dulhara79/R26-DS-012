# 🛠 Team Setup & Deployment Guide
**Project: SLIIT Anxiety Research Mobile App (2026)**

This guide ensures all team members can build and deploy the application with identical security and configuration settings.

---

## 1. Environment Setup
Every team member must have:
*   **Flutter SDK**: [Download](https://docs.flutter.dev/get-started/install). Verify with `flutter doctor`.
*   **Java JDK 17**: Required for building the Android APK.
*   **Android SDK**: API Level 34 (Android 14) or higher.

---

## 2. Google Sheets Backend Setup
The app uses a custom Google Apps Script to handle data. **One person** should host the master script:

1.  Create a new Google Sheet.
2.  Go to **Extensions > Apps Script**.
3.  Copy the code from `google_apps_script/doPost.gs` into the editor.
4.  **Initialize Security**:
    *   Find the function `setupScript()` in the script editor.
    *   Select it in the toolbar and click **Run**. This creates the study folder and sets the `AUTH_TOKEN`.
5.  **Deploy**:
    *   Click **Deploy > New Deployment**.
    *   Select **Web App**.
    *   Set "Execute as" to **Me**.
    *   Set "Who has access" to **Anyone**. (The `AUTH_TOKEN` will protect it).
    *   Copy the **Web App URL**. This is your `SCRIPT_URL`.

---

## 3. Building the Application
To protect research methodology, we use **Obfuscated Builds**. Use the following values for all team builds:

*   **Master Auth Token**: `7c09db655b5f697a4faf0b18a517d5fb` (Set by `setupScript`)

### Build Command (Windows PowerShell):
```powershell
flutter build apk --obfuscate --split-debug-info=./debug-info `
  --dart-define=SCRIPT_URL="YOUR_WEB_APP_URL" `
  --dart-define=AUTH_TOKEN="7c09db655b5f697a4faf0b18a517d5fb"
```

---

## 4. Participant Onboarding Flow
When testing the app, follow this sequence:
1.  **Login**: Enter a unique Participant ID (e.g., `P001`).
2.  **Permissions**: Accept **all** permissions (Location, Usage, etc.).
3.  **Profile**: Fill in the demographic data. This is a **one-time** setup.
4.  **Battery**: If the Dashboard shows a warning, tap it and set the app to **"Unrestricted"** in Android Battery Settings.

---

## 5. Modifying the App
*   **Theme**: Colors and fonts are managed in `lib/theme/app_theme.dart`.
*   **Data Types**: To add new sensors, update `lib/services/background/data_collector.dart`.
*   **Sync Logic**: Managed in `lib/services/background/background_service_helper.dart`.

---

## 6. Common Issues & Fixes
*   **"Unauthorized" Error**: Your `AUTH_TOKEN` in the build command does not match the one in Google Script. Run `diagnoseSetup()` in Apps Script to verify.
*   **Missing Data**: Ensure you have a stable internet connection for the first sync. The app will queue data offline if the connection is lost.
*   **Build Failure**: Run `flutter clean` then `flutter pub get` to reset the build cache.

---
**Lead Developer**: Dulhara Kaushalya
**Study Period**: May 2026 - June 2026
