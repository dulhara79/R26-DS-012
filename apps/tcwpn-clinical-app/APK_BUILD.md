# Building the ClinAnx APK

> **The previous version of this file was wrong and produced a broken APK.**
> It passed `FUSION_BASE`, `C1_BASE`, `C2_BASE`, `C3_BASE` and `HF_TOKEN`, none
> of which `Env` reads any more, and it never passed `BACKEND_BASE`. Following
> it built an app whose Central Backend URL was the empty string: every clinical
> call failed at `ApiClient._send` with "No base URL configured for this
> service." Delete any local script that still carries those defines.

---

## 1. Defines

| Define | Required | Value |
|---|---|---|
| `BACKEND_BASE` | **yes** | Central Backend base URL, no trailing slash |
| `BACKEND_TOKEN` | **yes** | matches `BACKEND_API_TOKEN` server-side |
| `TCWPN_BASE` | no | TC-WPN Space — `/health` warm-up only |
| `AUTH_BASE` | no | clinician auth service; omit for local demo mode |
| `AUTH_SALT` | no | local-mode password salt |
| `AUTH_LOCAL` | no | local-mode account table |
| `DEMO_DATA` | no | **`false` for any build that touches real patients** |
| `DISABLE_TLS_PINNING` | no | emulator behind a debugging proxy only; never ship |

Anything else you may remember — `FUSION_BASE`, `C1_BASE`, `C2_BASE`, `C3_BASE`,
`C4_BASE`, `CARE_RAG_BASE`, `HF_TOKEN` — is not configuration for this app. The
backend owns every one of those services.

## 2. Launcher icons

```bash
flutter pub get
dart run flutter_launcher_icons
```

## 3. Debug APK

```bash
flutter build apk --debug `
  --dart-define=BACKEND_BASE=$BACKEND_BASE `
  --dart-define=BACKEND_TOKEN=$BACKEND_TOKEN `
  --dart-define=TCWPN_BASE=$TCWPN_BASE `
```

## 4. Release APK

```bash
flutter build apk --release `
  --dart-define=BACKEND_BASE=$BACKEND_BASE `
  --dart-define=BACKEND_TOKEN=$BACKEND_TOKEN `
  --dart-define=TCWPN_BASE=$TCWPN_BASE `
  --dart-define=DEMO_DATA=false
```

Output: `build/app/outputs/flutter-apk/app-release.apk`

PowerShell uses a backtick for line continuation; the defines are identical.

### Keeping the token out of your shell history

Read it from the environment (`$BACKEND_TOKEN` above) or from a file:

```bash
flutter build apk --release --dart-define-from-file=dart_defines.json
```

`dart_defines.json` is a flat JSON object of the same keys. Copy it from
`.env.example`, fill it in, and confirm it is gitignored — it is not committed,
and it must never be.

## 5. Before you ship a release build

- `BACKEND_TOKEN` came from the environment, not from a source file.
- `DEMO_DATA=false`, so the seeded demonstration patient is gone.
- `DISABLE_TLS_PINNING` is **not** set.
- `kPinnedHosts` in `lib/core/security/pinned_certificates.dart` includes your
  Central Backend host, and `kPinsReviewBy` is a real date. As of this writing
  it lists only the TC-WPN Space, which carries nothing but a health ping, while
  the backend that carries every note is unpinned. Regenerate with
  `python tool/pin_certs.py`.
- `git grep -nE "hf_[A-Za-z0-9]{20,}|BACKEND_TOKEN\s*=\s*['\"]"` returns nothing.

## 6. Verify the build is actually talking to the backend

Install the APK, open **Settings**, and check the Central Backend health row. If
it reads "No base URL configured", `BACKEND_BASE` did not reach the build —
almost always a quoting problem in the shell, not a backend fault.
