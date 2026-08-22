# 🌿 Aura - Mindfulness Tracker

A premium, Flutter-based **Digital Phenotyping & Physiological Monitoring** application designed for longitudinal anxiety research and personal wellness tracking. Aura collects high-frequency physiological data, behavioral signals, and Ecological Momentary Assessment (EMA) ratings while maintaining strict privacy and presenting insights in a beautifully designed, calming interface.

## ✨ Key Features

*   **Real-time Physiological Dashboard**: 
    *   Simulated live monitoring of Heart Rate, Breathing Rate, Body Temperature, and Motion (g-force).
    *   **Anxiety Risk Score**: Dynamically calculated on a 0-100 scale using clinical thresholds.
    *   **Personalized Advice**: Actionable wellness recommendations generated based on the user's real-time anxiety risk score.
    *   **30-Day Trend Charts**: Visualized historical data to help users understand their physiological cycles and stress patterns.
*   **Digital Phenotyping**: Non-intrusively monitors behavioral patterns, app usage categories, and screen time to correlate digital habits with mental wellbeing.
*   **Offline-First Resilience**: Implements a robust local queue. Data is stored safely and synced to the cloud when internet connectivity is restored.
*   **Background Monitoring**: Utilizes a highly optimized Android Foreground Service to ensure continuous, battery-efficient data collection.
*   **Privacy-Hardened Engine**:
    *   **GPS Fuzzing**: Location data is obfuscated to protect user privacy.
    *   **App Categorization**: Specific app names are masked into broad categories (e.g., "Social Media").
*   **Clinical Assessments**: Integrated weekly GAD-7, monthly PSS-10 assessments, and daily EMA check-ins with reliable local push notifications.

## 🛠 Architecture & Tech Stack

*   **Framework**: Flutter (Dart)
*   **State Management & UI**: Responsive, beautifully animated components using Google Fonts (Poppins), `fl_chart` for data visualization, and a premium glassmorphism aesthetic.
*   **Background Execution**: `flutter_background_service`, `connectivity_plus`, `sensors_plus`, and `battery_plus`.
*   **Backend Sync**: Synchronizes via `http` to a private, multi-sheet Google Apps Script backend.

## 📦 Getting Started

### 1. Prerequisites
*   Flutter SDK (Stable)
*   Android SDK (API 34+)
*   Google Account (for the backend data sink)

### 2. Setup
1.  Fetch dependencies: `flutter pub get`
2.  Deploy the Google Apps Script found in `google_apps_script/doPost.gs`.
3.  Set the `AUTH_TOKEN` in your script using the provided setup instructions.

### 3. Build & Deploy
For secure deployment, build using the following command (injecting your secrets securely at build time):

```bash
flutter build apk --obfuscate --split-debug-info=./debug-info \
  --dart-define=SCRIPT_URL="YOUR_GOOGLE_SCRIPT_URL" \
  --dart-define=AUTH_TOKEN="YOUR_SECRET_TOKEN"
```

## 🔒 Privacy & Security
All API keys and authentication tokens are injected dynamically at build-time using `--dart-define`. The application is built with an absolute priority on user anonymity and data encryption.

Developed for clinical research and personal mindfulness tracking. All data collection follows ethical guidelines for user privacy and anonymity.
