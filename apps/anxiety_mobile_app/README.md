# Aura

Flutter Android app for longitudinal anxiety research (SLIIT research group **R26-DS-012**).

Aura combines a BLE chest strap, privacy-preserving phone sensing, self-reports, and participant-facing behavioural context. It is offline-first and uses pseudonymous participant IDs.

> **Research use only.** Not a medical device, diagnosis, or emergency service. Behavioural observations are personal-baseline context- not a clinical anxiety score.

## Features

- **Home**: gentle body-reading feedback and risk band from the research backend
- **Body**: live vitals from `ChestStrap_V3` over Bluetooth
- **Activity: personal baseline and change detection from passive phone sensing (descriptive only)
- **Profile**: demographics, Participant ID QR, privacy / data rights, appearance

Also includes informed consent, anxiety check-ins, EMA (3× daily), weekly GAD-7 and PSS-10, and an optional clinician summary PDF that the participant can share.

## How data moves

```text
Chest strap (BLE)  →  Body tab  →  physiological backend + Central Backend
Phone sensors      →  local offline queue  →  Supabase  →  daily processing  →  Activity tab
Self-reports       →  offline queue (+ optional backend ingest)
```

Passive events include screen on/off/unlock, coarsened location (~15 min), app-usage categories (~15 min), movement windows (5 min), daily call/SMS counts, battery status, and service heartbeats. Location is rounded on-device; app packages become categories; call/SMS content is never uploaded. Chest-strap vitals are kept on the physiological path and are not mixed into the behavioural event stream.

## Privacy

- Participant codes look like `P_` + 16 hex characters
- Mobile Supabase access uses a publishable key and anonymous auth — never a service-role key in the APK
- Clinician PDF is shared only if the participant chooses; it is not a risk score
