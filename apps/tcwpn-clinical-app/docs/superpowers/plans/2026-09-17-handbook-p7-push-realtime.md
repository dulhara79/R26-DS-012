# Handbook Phase 7 Push / Realtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the ClinAnx mobile side of System Integration Handbook Phase 7 with FCM push delivery, `/v1/device-tokens` registration, PHI-free event payloads, notification-tap routing, and the existing polling fallback.

**Architecture:** The Central Backend remains the only clinical authority and persists every AttentionEvent. One Firebase Messaging project is active in any installed build because the FlutterFire `firebase_messaging` plugin does not support runtime multi-project messaging instances. The source supports two build-time Firebase slots (`primary` and `secondary`) so a disaster-recovery APK can be built against the second project; runtime resilience is provided by persistent server events plus polling. FCM carries only a generic notification plus `type=attention_event` and `event_id`.

**Tech Stack:** Flutter >=3.27.0, Dart >=3.5.0, firebase_core ^4.13.0, firebase_messaging ^16.5.0, flutter_local_notifications, existing ApiClient, SharedPreferences notification dedupe, GitHub Actions.

**Spec:** `R26-DS-012_System_Integration_Implementation_Handbook.pdf` Phase 7 — Device-token registry + push delivery + polling fallback.

## Global Constraints

- Do not use Firebase Realtime Database, Firestore, Storage, Functions, or Firebase Authentication.
- Do not put patient name, MRN, note text, scores, fusion values, model outputs, or clinical details in push payloads.
- Push delivery is never acknowledgement/resolution; only the Central Backend AttentionEvent lifecycle is authoritative.
- Keep the existing 30-second foreground polling fallback.
- Register device tokens only after clinician authentication.
- Revoke the current registration on explicit sign-out without allowing a network failure to block sign-out.
- Include the active Firebase `provider_project_id` in `/v1/device-tokens` registrations so primary and DR tokens cannot be mixed server-side.
- Do not claim seamless runtime switching between Firebase projects; secondary is a build-time disaster-recovery slot.
- Do not add this feature branch to workflow push triggers; let the full workflow run on the PR to `main` to avoid repeated failure-email noise.

## Acceptance Criteria

- An authenticated Android/iOS build with valid active-slot Firebase identifiers can obtain and register an FCM token through `/v1/device-tokens`.
- The backend receives the active Firebase project ID with every token registration.
- Foreground push produces the same generic local notification used by polling and is deduplicated by event ID.
- Background/terminated notification taps route only by event ID and fetch sensitive details from the authenticated Central Backend path.
- Push loss/throttling does not lose events because server persistence plus polling remains intact.
- Explicit sign-out performs best-effort token deactivation and still signs out when token revocation cannot reach the backend.
- Source supports a primary and secondary Firebase build slot, but does not make the unsupported claim that FlutterFire Messaging can switch projects at runtime.
- No Firebase database/storage/functions product is introduced.
