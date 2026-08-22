# 🧠 Mindful Tracker: Anxiety Research Study

A professional Flutter-based **Digital Phenotyping** application designed for longitudinal anxiety research studies. The app collects high-frequency data (Location, App Usage, Activity) and EMA (Ecological Momentary Assessment) ratings while maintaining strict participant privacy.

## 🚀 Key Research Features

*   **Offline-First Resilience**: Implements a 10,000-item local queue. Data is never lost, even if the phone restarts or internet is disconnected for days.
*   **Persistent Monitoring**: Uses a high-priority Android Foreground Service with an "Auto-Restart Watchdog" to ensure 24/7 data collection.
*   **Privacy-Hardened Backend**:
    *   **GPS Fuzzing**: Coordinates are rounded to ±1km to protect participant home addresses.
    *   **App Masking**: Specific app names (e.g., "WhatsApp") are categorized (e.g., "Social Media") before being stored.
    *   **Multi-Sheet Architecture**: Automatically creates a separate, private Google Sheet for every unique Participant ID.
*   **Research Tools**: Integrated GAD-7 weekly assessments and 3x daily EMA check-ins (Morning, Afternoon, Evening) with custom notification scheduling.

## 🛠 Project Structure

*   `lib/services/background/`: Core data collection and notification engines.
*   `lib/pages/`: UI for Dashboard, Login, and Demographic onboarding.
*   `google_apps_script/`: The backend logic for Google Sheets integration.
*   `theme/`: A calming, modern design system based on Google Fonts (Poppins).

## 📦 Getting Started

### 1. Prerequisites
*   Flutter SDK (Stable)
*   Android SDK (API 34+)
*   Google Account (for the Sheets backend)

### 2. Setup
1.  Fetch dependencies: `flutter pub get`
2.  Deploy the Google Apps Script found in `google_apps_script/doPost.gs`.
3.  Set the `AUTH_TOKEN` in the script using `setupScript()`.

### 3. Build & Deploy
For research deployment, use the hardened build command to enable obfuscation:

```bash
flutter build apk --obfuscate --split-debug-info=./debug-info \
  --dart-define=SCRIPT_URL="YOUR_GOOGLE_SCRIPT_URL" \
  --dart-define=AUTH_TOKEN="YOUR_SECRET_TOKEN"
```

## 🔒 Security
All API keys and authentication tokens are injected at build-time using `--dart-define`. This ensures no secrets are hardcoded in the source code or committed to Version Control.

## 📜 License
This project is intended for clinical research. All data collection follows ethical guidelines for participant anonymity.
