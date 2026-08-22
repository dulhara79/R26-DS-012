# Anxiety Research — User Guide (APK)

This guide explains how to install and use the Anxiety Research mobile application APK on Android devices. Share this with study participants to ensure proper setup, permissions, and safe use.

---

## Important: Read Before Installing

- This app collects device sensor and usage information (location, accelerometer events, app usage summaries, call/SMS counts) and daily self-report ratings for research purposes.
- Only install and use the app if you have been asked by a research team and provided informed consent.
- If you have privacy concerns, contact the study coordinator before installing.

---

## What the App Does (User Summary)

- Runs a lightweight background service to collect anonymized research data.
- Allows you to record anxiety events by pressing the on-screen pad (pressure-based intensity).
- Sends batched data to the research server. If there is no network, data is queued locally and retried later.
- Prompts a short daily rating (0–5) at a time you set; tapping the notification opens the in-app rating dialog.

---

## Before You Start (Requirements)

- Android device (the APK is Android-only). iOS builds are not distributed as APKs.
- Android 8.0 (Oreo) or later recommended for background/notification behavior.
- Internet connection for data upload (offline caching available).

---

## Installing the APK

Two common ways: using the device file manager or ADB (for advanced users).

1. Install from device file manager

- Copy the `app.apk` file to your device (via USB, email, or cloud storage).
- On the device, open `Settings` → `Apps & notifications` → `Special app access` → `Install unknown apps` and allow the app (e.g., your file manager or browser) to install unknown apps.
- Open the APK in a file manager and tap `Install`.

2. Install via ADB (developer)

```bash
adb install path/to/app.apk
```

After install, open the app from your launcher.

---

## First-Time Setup (Step-by-step for participants)

1. Open the app. You will see a two-step setup screen.
2. Tap **Grant Secure Access**. The app will request several permissions — allow them so the app can collect required data.
   - Expected permissions: Location (foreground & background), Notifications, Phone, SMS, Usage Access, and request to ignore battery optimizations. The app will explain why each permission is needed.
3. After granting permissions, enter your Participant ID in the **Enter Participant ID** field. This ID is required so the research server can link your responses to the study (it does not store your name unless you supply it).
4. Tap **Initialize Session**. This starts the background service and enables data collection.

Notes:

- If any permission is permanently denied, the app will ask you to open system settings to re-enable it. Follow the on-screen instructions.
- If the app asks for **Usage Access**, you must enable it in system settings: usually `Settings` → `Digital Wellbeing & parental controls` or `Settings` → `Apps` → `Special app access` → `Usage access` → enable for this app.

---

## Using the App

Main screen (Monitoring Dashboard):

- The large circular pad: press and hold to record an anxiety event. Vary pressure to indicate intensity. The app will send a `Touch_Event` with intensity to the research server (throttled to avoid excess records).
- Status bar indicates the system is active and recording.
- Settings icon opens **Daily Check-in Settings** (see next section).

Daily Check-in (Rating):

- The app can show a daily notification at a time you set; tapping the notification will open a 0–5 rating dialog to record your daily stress level.

Stop data collection:

- To stop all collection, either uninstall the app or revoke the app permissions from system settings. Uninstalling will stop and remove all app data by default.

---

## Settings — Daily Check-in

- Open the settings (gear icon) to enable or disable the daily rating and select the notification time.
- The app will show the notification once per day and will not prompt again after you submit a rating for that day.

---

## Notifications & Background Behavior

- The app runs a background (foreground) service to ensure reliable collection; it uses a persistent notification when active.
- If notifications are not appearing, verify notification permission is enabled and that battery optimization is disabled for the app (see Troubleshooting).

---

## Data, Privacy & Security (Plain language)

- What is collected: anonymized or pseudonymized research data such as location coordinates, accelerometer spikes (movement), app usage summaries (package names + usage time), call counts (incoming/outgoing/missed totals), SMS counts, touch event intensity, and daily ratings.
- How it's stored: temporarily on-device in a local queue if network is unavailable; items are stored in app storage and retried automatically.
- How it's transmitted: uploaded over HTTPS to the research team server. If you have questions about server storage duration or access, contact the study coordinator.
- Consent: do not install unless you have read and signed the study consent form and been instructed by the research team.

Recommendations for participants:

- Use a device you are comfortable sharing the described data from, or opt out of participation if uncomfortable.
- Ask the research team for details on data retention, anonymization, and deletion.

---

## Troubleshooting (Common Issues)

- Notifications not appearing
  - Ensure **Notifications** are allowed for the app in system settings.
  - Disable battery optimization for the app: `Settings` → `Battery` → `Battery optimization` → find the app → select `Don't optimize` / `Allow background activity`.

- App stopped collecting after some time
  - Many devices have aggressive battery managers that kill background services. Whitelist the app in device manufacturer-specific settings (e.g., Samsung, Xiaomi, Huawei). Also ensure `REQUEST_IGNORE_BATTERY_OPTIMIZATIONS` permission was granted if prompted.

- Daily rating didn't appear at the set time
  - Confirm `Enable daily rating` is on and the correct time is set in settings.
  - Verify device time and timezone are correct.

- Events not uploaded (appearing delayed)
  - The app caches events when offline and retries automatically. Check your network connection or wait until device returns online.

- Permissions revoked accidentally
  - Re-open the app and follow prompts or open system `Settings` → `Apps` → select the app → `Permissions` and re-enable required permissions.

If problems persist, contact the research team with the following details: device model, Android version, steps to reproduce, and approximate time of issue.

---

## Safety & Support

- If at any time participation causes distress, stop using the app and contact the study contact/supervisor immediately.
- Contact: provide researcher contact details here (email/phone) — replace this text with the research team's contact information before distribution.

---

## Uninstalling the App

- Standard uninstall from device Settings or app drawer: long-press app icon → `Uninstall`, or `Settings` → `Apps` → select app → `Uninstall`.

---

## Frequently Asked Questions (FAQ)

- Q: Will my name or messages be uploaded?
  - A: The app does not automatically upload message bodies or contact names. It collects counts (e.g., number of SMS today) and call counts. If the study requires additional identifiers, the research team will inform you.

- Q: How often is data uploaded?
  - A: The app batches events and uploads periodically (roughly every 6 seconds after events are generated for batching, and periodic heavy syncs every 15 minutes). If uploads fail, they are retried.

- Q: Can I pause the study temporarily?
  - A: You can disable permissions or uninstall the app to pause. Check with the research team for recommended procedures to resume participation.

---

## Versioning & Support Notes for Distributors

- App version is visible in app settings or installer metadata. Keep participants informed of updates and re-consent if the study protocol changes.

---