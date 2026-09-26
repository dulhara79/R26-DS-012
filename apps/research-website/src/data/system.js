// Integrated system. Sources: apps/anxiety_mobile_app/README.md (Aura) and
// apps/tcwpn-clinical-app/README.md (ClinAnx) in the R26-DS-012 repository.

export const architecture = [
  {
    layer: "Interfaces",
    items: [
      "Aura participant app (Flutter, Android)",
      "ClinAnx clinician console (Flutter)",
    ],
  },
  {
    layer: "Central backend",
    items: [
      "Single entry point for every model call",
      "Holds canonical state and eligibility",
    ],
  },
  {
    layer: "Component services",
    items: ["C1 physiological", "C2 behavioural", "C3 TC-WPN", "C4 DCAR prior"],
  },
  {
    layer: "Fusion",
    items: [
      "Reliability-weighted fusion",
      "Low / Medium / High or insufficient evidence",
    ],
  },
  {
    layer: "Evidence",
    items: [
      "CARE-AnxRAG guidance",
      "Abstention and insufficient-evidence states",
    ],
  },
];

export const aura = {
  name: "Aura",
  role: "Participant research app",
  summary:
    "Flutter Android application for longitudinal anxiety research. It integrates wearable physiological monitoring, passive smartphone sensing, study questionnaires, offline-first collection and participant-facing behavioural context.",
  flows: [
    [
      "ESP32-C3 chest strap",
      "Flutter physiological dashboard",
      "Component 1 physiological service",
    ],
    [
      "Android passive sensing",
      "Local offline queue",
      "Supabase sensor_events",
      "Component 2 daily processor",
      "Behavioural context UI",
    ],
  ],
  collection: [
    ["Screen events", "Event-driven screen on, unlock and off transitions"],
    [
      "Location grid (100 m)",
      "About every 15 minutes; coordinates rounded to 3 decimals on-device",
    ],
    ["App usage category", "About every 15 minutes; category totals only"],
    ["Movement window", "Five-minute accelerometer summary"],
    ["Call statistics", "Previous-day aggregate counts"],
    ["SMS activity", "Previous-day aggregate counts"],
    ["Battery status", "Hourly"],
    ["Service heartbeat", "Hourly"],
  ],
  privacy: [
    "Pseudonymous participant codes instead of names.",
    "Location coarsened on-device before queueing.",
    "App package names converted to broad categories on-device.",
    "Call and SMS content, phone numbers and contact names are not uploaded.",
    "Participant-facing behavioural observations are not a diagnosis or a calibrated risk probability.",
  ],
};

export const clinanx = {
  name: "ClinAnx",
  role: "Clinician console",
  summary:
    "Flutter app for psychiatry teams: enrol a patient, submit a clinical note, and read the multimodal assessment the central backend produces. It is a presentation layer; every model call, fusion step and piece of guidance happens server-side.",
  screens: [
    ["Caseload", "Roster, enrolment state, latest band per patient"],
    [
      "Patient chart",
      "Composite, tier, band, confidence, per-modality breakdown, trend",
    ],
    ["Note analysis", "Submit a clinical note; TC-WPN runs server-side"],
    ["Support set", "Manage the labelled notes that form TC-WPN's prototypes"],
    ["Fusion detail", "Gate decision, weights, contributions, conformal set"],
    [
      "Evidence",
      "CARE-AnxRAG guidance, with abstention and insufficient-evidence states",
    ],
    ["Alerts and settings", "Escalations; service health and session"],
  ],
  doesNot: [
    "Compute a composite score locally. If the backend is unreachable, the chart says so.",
    "Call component services, fusion or CARE-AnxRAG directly.",
    "Implement the retired GBDT / SHAP / DiCE intervention engine.",
    "Turn a missing score into 0, or a failed request into “Low risk”.",
  ],
};
