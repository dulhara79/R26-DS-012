# Building ClinAnx — Handbook Phase 9 research release

This guide defines the reproducible mobile build procedure for the R26-DS-012 ClinAnx research prototype.

> **Research prototype — not a diagnostic device.** Do not treat a successful build as clinical deployment approval.

## 1. Pinned mobile release inputs

For the Phase 9 research-release candidate:

- App version: `1.0.0+1` from `pubspec.yaml`
- Flutter SDK for CI/release verification: `3.47.4`
- Dart/package dependency graph: committed `pubspec.lock`
- Source revision: inject the exact commit used for the build as `BUILD_REVISION`
- Android Java target: 17
- Minimum Android SDK: repository/Flutter configuration, with launcher configuration requiring API 23+

Before building, record:

```bash
flutter --version
git rev-parse HEAD
git status --short
```

The working tree should be clean. Do not build a research artifact from uncommitted changes.

## 2. Security boundary

ClinAnx must not contain a reusable privileged Central Backend/service credential. Central clinical requests use the signed-in clinician session bearer. Service-to-service secrets belong on the Central Backend.

Remote endpoints carrying participant/clinical data must use HTTPS. Plain HTTP is allowed only for loopback development.

Never commit:

- clinician passwords;
- service-account private keys;
- backend/model service credentials;
- raw participant identifiers;
- Firebase server credentials;
- signing keystores/passwords.

## 3. Required/supported build defines

| Define                    |              Research/study build | Purpose                                           |
| ------------------------- | --------------------------------: | ------------------------------------------------- |
| `BACKEND_BASE`            |                          required | Central Backend URL; HTTPS for remote hosts       |
| `AUTH_BASE`               | required for real participant use | clinician authentication service                  |
| `APP_VERSION`             |                          required | version shown in research build identity          |
| `BUILD_ENVIRONMENT`       |                          required | e.g. `study`, `staging`, `study-dr`               |
| `BUILD_REVISION`          |                          required | exact Git revision used for the artifact          |
| `DEMO_DATA`               |                   must be `false` | prevents synthetic fixtures in participant builds |
| `TCWPN_BASE`              |                          optional | unauthenticated `/health` warm-up only            |
| `PUSH_FIREBASE_SLOT`      |       optional, default `primary` | selects Primary or Secondary Firebase build slot  |
| `FIREBASE_PRIMARY_*`      |   required for Primary push build | Firebase client routing configuration             |
| `FIREBASE_SECONDARY_*`    |   required for Secondary DR build | Firebase client routing configuration             |
| `AUTH_SALT`, `AUTH_LOCAL` |                     demo/dev only | synthetic local authentication                    |

## 4. Primary research APK

PowerShell:
```powershell
# Load .env variables into the current PowerShell session
Get-Content .env | ForEach-Object {
    $line = $_.Trim()

    # Skip empty lines and comments
    if ($line -eq "" -or $line.StartsWith("#")) {
        return
    }

    # Split only on the first =
    $parts = $line -split "=", 2

    if ($parts.Count -eq 2) {
        $name  = $parts[0].Trim()
        $value = $parts[1].Trim()

        # Remove surrounding single/double quotes if present
        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        Set-Item -Path "Env:$name" -Value $value
    }
}
```

```powershell
$REVISION = git rev-parse HEAD

flutter build apk --release `
  --dart-define=BACKEND_BASE=$env:BACKEND_BASE `
  --dart-define=AUTH_BASE=$env:AUTH_BASE `
  --dart-define=TCWPN_BASE=$env:TCWPN_BASE `
  --dart-define=APP_VERSION=1.0.0+1 `
  --dart-define=BUILD_ENVIRONMENT=study `
  --dart-define=BUILD_REVISION=$REVISION `
  --dart-define=DEMO_DATA=false `
  --dart-define=PUSH_FIREBASE_SLOT=primary `
  --dart-define=FIREBASE_PRIMARY_API_KEY=$env:FIREBASE_PRIMARY_API_KEY `
  --dart-define=FIREBASE_PRIMARY_APP_ID=$env:FIREBASE_PRIMARY_APP_ID `
  --dart-define=FIREBASE_PRIMARY_SENDER_ID=$env:FIREBASE_PRIMARY_SENDER_ID `
  --dart-define=FIREBASE_PRIMARY_PROJECT_ID=$env:FIREBASE_PRIMARY_PROJECT_ID
```

## 5. Secondary Firebase disaster-recovery APK

Build a separate artifact using the Secondary project configuration:

```powershell
# Load .env variables into the current PowerShell session
Get-Content .env | ForEach-Object {
    $line = $_.Trim()

    # Skip empty lines and comments
    if ($line -eq "" -or $line.StartsWith("#")) {
        return
    }

    # Split only on the first =
    $parts = $line -split "=", 2

    if ($parts.Count -eq 2) {
        $name  = $parts[0].Trim()
        $value = $parts[1].Trim()

        # Remove surrounding single/double quotes if present
        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        Set-Item -Path "Env:$name" -Value $value
    }
}
```

```powershell
$REVISION = git rev-parse HEAD

flutter build apk --release `
  --dart-define=BACKEND_BASE=$env:BACKEND_BASE `
  --dart-define=AUTH_BASE=$env:AUTH_BASE `
  --dart-define=TCWPN_BASE=$env:TCWPN_BASE `
  --dart-define=APP_VERSION=1.0.0+1 `
  --dart-define=BUILD_ENVIRONMENT=study-dr `
  --dart-define=BUILD_REVISION=$REVISION `
  --dart-define=DEMO_DATA=false `
  --dart-define=PUSH_FIREBASE_SLOT=secondary `
  --dart-define=FIREBASE_SECONDARY_API_KEY=$env:FIREBASE_SECONDARY_API_KEY `
  --dart-define=FIREBASE_SECONDARY_APP_ID=$env:FIREBASE_SECONDARY_APP_ID `
  --dart-define=FIREBASE_SECONDARY_SENDER_ID=$env:FIREBASE_SECONDARY_SENDER_ID `
  --dart-define=FIREBASE_SECONDARY_PROJECT_ID=$env:FIREBASE_SECONDARY_PROJECT_ID
```

The Secondary build is not an instant runtime Firebase switch. FCM registration tokens are project-specific. Persistent server AttentionEvents plus polling remain the runtime recovery mechanism.

## 6. Demo-only build

Use synthetic/demo data only:

```powershell
$REVISION = git rev-parse HEAD

flutter build apk --debug `
  --dart-define=BACKEND_BASE=http://127.0.0.1:8000 `
  --dart-define=APP_VERSION=1.0.0+1 `
  --dart-define=BUILD_ENVIRONMENT=demo `
  --dart-define=BUILD_REVISION=$REVISION `
  --dart-define=DEMO_DATA=true
```

If local demo authentication is used, provide `AUTH_LOCAL` and `AUTH_SALT` outside source control.

## 7. Verification before artifact acceptance

Run:

```bash
flutter pub get
flutter test
flutter analyze
```

Then verify on the installed build:

1. Settings shows the expected app version, environment, build revision, backend host and Firebase slot/project.
2. No secret value is displayed in Settings or logs.
3. Sign-in and session-expiry behavior are correct.
4. Dashboard/Patients/Activity can distinguish unavailable/offline/stale state from Low risk.
5. Current assessment and forecast are visibly separate.
6. C2 is labelled experimental/excluded.
7. Notification permission denial does not disable Activity/polling fallback.
8. Primary and Secondary builds identify their matching Firebase project.
9. A push notification contains no clinical detail and opens by event identity.
10. Receiving a notification does not ACK/RESOLVE the event.

Record manual results in `docs/qa/phase8_usability_checklist.md` and release evidence in `docs/release/TEST_EVIDENCE.md`.

## 8. Artifact identity and hash

Android output:

```text
build/app/outputs/flutter-apk/app-release.apk
```

Create a SHA-256 hash and record it in the release sheet.

PowerShell:

```powershell
Get-FileHash build/app/outputs/flutter-apk/app-release.apk -Algorithm SHA256
```

Linux/macOS:

```bash
sha256sum build/app/outputs/flutter-apk/app-release.apk
```

## 9. Important Android packaging limitation

The current repository still uses the example Android application ID `lk.sliit.r26ds012.clinanx` and the `release` build type is currently wired to the debug signing configuration. Therefore a generated `--release` APK must be treated as a **research/development artifact, not a production-signed distribution artifact**.

Before external study distribution, the team must configure an approved unique application ID and a protected release keystore/signing process. Do not commit the keystore or its passwords.

## 10. Release handoff

For each accepted artifact, record in `docs/release/PHASE9_RESEARCH_RELEASE.md` or the project release sheet:

- Git commit SHA / `BUILD_REVISION`;
- app version;
- Flutter version;
- environment name;
- backend host/environment identifier;
- Firebase slot and project ID;
- APK SHA-256;
- automated CI run links/IDs;
- manual smoke-test date/tester;
- known limitations/exceptions.
