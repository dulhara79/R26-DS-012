# Building ClinAnx — Handbook Phase 8

This guide follows the `R26-DS-012 System Integration Implementation Handbook` hardening rules.

## Security boundary

ClinAnx **must not** contain a reusable Central Backend/service credential. Central Backend requests use the signed-in clinician session bearer. Service-to-service secrets belong on the Central Backend.

Remote application-facing endpoints that can carry participant/clinical data must use HTTPS. Plain HTTP is permitted only for loopback development (`localhost`, `127.0.0.1`, `::1`).

## Required / supported defines

| Define | Research/study build | Purpose |
|---|---:|---|
| `BACKEND_BASE` | required | Central Backend URL; HTTPS for remote hosts |
| `AUTH_BASE` | required for real participant use | clinician authentication service |
| `APP_VERSION` | required for reproducible release | version displayed in research build identity |
| `BUILD_ENVIRONMENT` | required for reproducible release | e.g. `demo`, `study`, `staging` |
| `BUILD_REVISION` | required for reproducible release | Git commit/revision |
| `DEMO_DATA` | must be `false` for participant use | explicit synthetic fixture opt-in |
| `TCWPN_BASE` | optional | unauthenticated `/health` warm-up only |
| `PUSH_FIREBASE_SLOT` | optional, default `primary` | `primary` or `secondary` DR build slot |
| `FIREBASE_PRIMARY_*` | required for primary push build | Firebase FCM routing configuration |
| `FIREBASE_SECONDARY_*` | required for secondary DR build | Firebase FCM routing configuration |
| `AUTH_SALT`, `AUTH_LOCAL` | demo/dev only | local demonstration authentication |

Do **not** add `BACKEND_TOKEN`, model tokens, service-account private keys, passwords, or backend service credentials to a mobile build.

## Primary release build (PowerShell)

```powershell
$REVISION = git rev-parse --short HEAD

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

## Secondary Firebase disaster-recovery build

Build a separate APK/app package with the secondary slot selected:

```powershell
$REVISION = git rev-parse --short HEAD

flutter build apk --release `
  --dart-define=BACKEND_BASE=$env:BACKEND_BASE `
  --dart-define=AUTH_BASE=$env:AUTH_BASE `
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

The secondary project is **not** a runtime API-key swap. FCM registration tokens are project-specific. Runtime resilience remains: server-persisted AttentionEvents + FCM when available + polling fallback.

## Demo/local-auth build

Local accounts are for synthetic/demo use only. Do not use this mode with real participant data.

```powershell
$REVISION = git rev-parse --short HEAD
flutter build apk --debug `
  --dart-define=BACKEND_BASE=$env:BACKEND_BASE `
  --dart-define=AUTH_LOCAL=$env:AUTH_LOCAL `
  --dart-define=AUTH_SALT=$env:AUTH_SALT `
  --dart-define=APP_VERSION=1.0.0+1 `
  --dart-define=BUILD_ENVIRONMENT=demo `
  --dart-define=BUILD_REVISION=$REVISION `
  --dart-define=DEMO_DATA=true
```

## Before a research/study release

1. `flutter test` passes.
2. `flutter analyze` reports no issues.
3. `BACKEND_BASE` and `AUTH_BASE` use HTTPS remote URLs.
4. `DEMO_DATA=false`.
5. `BUILD_REVISION` matches the commit used to build the APK/app.
6. Settings/research build information identifies the correct environment/backend/Firebase slot without displaying secrets.
7. Primary and secondary Firebase builds use their matching project configuration.
8. No reusable backend/model credential is compiled into the app.
9. The Phase 8 usability checklist is executed with synthetic/demo participants before research release.

Output for Android release: `build/app/outputs/flutter-apk/app-release.apk`.
