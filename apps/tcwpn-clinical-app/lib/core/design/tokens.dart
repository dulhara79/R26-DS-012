// lib/core/design/tokens.dart
//
// ONE token file for the entire application. Nothing in the app declares a raw
// Color, radius, or spacing value — everything resolves here. This is what makes
// the four research components read as a single product rather than three
// stitched-together prototypes.
//
// Direction: "clinical paper". Cool near-white ground, deep slate-teal ink,
// hairline structure, and a numeric register set in a monospaced face so risk
// scores read like instrument output rather than marketing copy.

import 'package:flutter/material.dart';

@immutable
class Ds {
  const Ds._();

  // ── Ground ───────────────────────────────────────────────────────────────
  static const Color canvas = Color(0xFFF7F9FA); // app background
  static const Color surface = Color(0xFFFFFFFF); // cards, sheets
  static const Color surfaceSunken = Color(0xFFF1F5F7); // wells, inputs
  static const Color hairline = Color(0xFFE3EAEE);
  static const Color hairlineStrong = Color(0xFFCFDBE1);

  // ── Ink ──────────────────────────────────────────────────────────────────
  static const Color ink = Color(0xFF0D1E26); // primary text
  static const Color inkMuted = Color(0xFF526973); // secondary text
  static const Color inkFaint = Color(0xFF8CA3AC); // captions, axis labels

  // ── Brand ────────────────────────────────────────────────────────────────
  static const Color brand = Color(0xFF0F5B6E); // deep slate-teal
  static const Color brandDeep = Color(0xFF0A4051);
  static const Color brandSoft = Color(0xFFE7F0F3);
  static const Color brandEdge = Color(0xFFB9D2DA);

  // ── Alert band (proposal §5.1: GREEN / AMBER / RED / DARK RED) ───────────
  static const Color green = Color(0xFF1F8A70);
  static const Color greenSoft = Color(0xFFE6F4F0);
  static const Color amber = Color(0xFFC08A1E);
  static const Color amberSoft = Color(0xFFFBF2DF);
  static const Color red = Color(0xFFD05B38);
  static const Color redSoft = Color(0xFFFAEBE5);
  static const Color darkRed = Color(0xFF98301C);
  static const Color darkRedSoft = Color(0xFFF4E0DC);

  // GREY is not a severity. It is the absence of an assessment — the backend
  // emits it when the fusion gate blocks, and it must never be coloured or
  // worded as though it were the low-risk end of the scale.
  static const Color grey = Color(0xFF6B7C85);
  static const Color greySoft = Color(0xFFEDF1F3);

  // ── Component identity colours (used only in fusion breakdowns) ──────────
  // Each modality keeps one colour everywhere it appears, so a clinician learns
  // "the sand band is the wearable" once and it holds across every screen.
  //
  // NAMING: these follow the CENTRAL BACKEND's wire vocabulary, where
  // c3_clinical_nlp is TC-WPN and c4_demographic is the DCAR prior. The paper
  // numbers those components the other way round (Component 4 = TC-WPN); the
  // human-readable labels in Modality.labels preserve the paper's numbering.
  // See the mapping table at the top of domain/models.dart.
  static const Color c1Physiological = Color(0xFF6E7FA8); // wearable
  static const Color c2Behavioral = Color(0xFF7FA88C); // phone sensing
  static const Color c3ClinicalNlp = Color(0xFF0F5B6E); // TC-WPN — the brand hue
  static const Color c4Demographic = Color(0xFFC0956B); // DCAR demographic prior

  // ── Radii ────────────────────────────────────────────────────────────────
  static const double rSm = 8;
  static const double rMd = 12;
  static const double rLg = 16;
  static const double rXl = 22;
  static const double rPill = 999;

  // ── Spacing scale (4pt) ──────────────────────────────────────────────────
  static const double s1 = 4;
  static const double s2 = 8;
  static const double s3 = 12;
  static const double s4 = 16;
  static const double s5 = 20;
  static const double s6 = 24;
  static const double s8 = 32;
  static const double s10 = 40;

  // ── Elevation: this product uses hairlines, not drop shadows. One shadow
  // exists, and only for surfaces that genuinely float (sheets, menus).
  static List<BoxShadow> get lift => const [
        BoxShadow(color: Color(0x0F0D1E26), blurRadius: 24, offset: Offset(0, 8)),
      ];

  static Duration get fast => const Duration(milliseconds: 160);
  static Duration get med => const Duration(milliseconds: 280);
}

/// Alert bands.
///
/// GREY IS NOT A BAND ON THE SEVERITY SCALE. The Central Backend returns
/// `band: "GREY"` with `composite: null` whenever the fusion gate blocks —
/// fewer than two usable modalities, no time-varying modality, or every reading
/// carrying a non-`ok` status. That is *missing evidence*, not low risk.
///
/// The previous `fromWire` had no GREY case and fell through to `green`, and
/// `composite_score` defaulted to 0, so a blocked assessment rendered as
/// "Stable · 0.000". Both defaults are fixed here and in FusionResult.fromJson.
enum AlertBand { grey, green, amber, red, darkRed }

extension AlertBandX on AlertBand {
  /// The four bands that actually express severity. Use this — not
  /// `AlertBand.values` — for legends, filters and caseload histograms, so
  /// "no assessment" is never offered as a risk level to filter by.
  static const List<AlertBand> scored = [
    AlertBand.green,
    AlertBand.amber,
    AlertBand.red,
    AlertBand.darkRed,
  ];

  bool get isScored => this != AlertBand.grey;

  /// Local banding from a score.
  ///
  /// Kept ONLY for numbers that never passed through the fusion service (e.g. a
  /// single modality's own score shown in isolation). It must NOT be used to
  /// re-derive the composite's band: the server bands at 0.33/0.66 into three
  /// tiers (fusion_service/fusion.py, BANDS) whereas this splits four ways at
  /// 0.25/0.50/0.75. Display the server's `tier`/`band` for the composite.
  static AlertBand fromScore(double s) {
    if (s >= 0.75) return AlertBand.darkRed;
    if (s >= 0.50) return AlertBand.red;
    if (s >= 0.25) return AlertBand.amber;
    return AlertBand.green;
  }

  /// Parses the backend's band vocabulary. An unrecognised value maps to GREY,
  /// not to GREEN: if we cannot tell what the server meant, the honest answer
  /// is "no assessment", never "stable".
  static AlertBand fromWire(String? s) {
    switch ((s ?? '').toUpperCase().replaceAll(' ', '_')) {
      case 'DARK_RED':
      case 'DARKRED':
        return AlertBand.darkRed;
      case 'RED':
        return AlertBand.red;
      case 'AMBER':
      case 'YELLOW':
        return AlertBand.amber;
      case 'GREEN':
        return AlertBand.green;
      case 'GREY':
      case 'GRAY':
      default:
        return AlertBand.grey;
    }
  }

  String get label => switch (this) {
        AlertBand.grey => 'No assessment',
        AlertBand.green => 'Stable',
        AlertBand.amber => 'Monitor',
        AlertBand.red => 'Review',
        AlertBand.darkRed => 'Urgent',
      };

  /// The band name as the proposal writes it — used in exports and PDFs where
  /// the protocol vocabulary matters more than the friendly label.
  String get protocolName => switch (this) {
        AlertBand.grey => 'GREY',
        AlertBand.green => 'GREEN',
        AlertBand.amber => 'AMBER',
        AlertBand.red => 'RED',
        AlertBand.darkRed => 'DARK RED',
      };

  Color get fg => switch (this) {
        AlertBand.grey => Ds.grey,
        AlertBand.green => Ds.green,
        AlertBand.amber => Ds.amber,
        AlertBand.red => Ds.red,
        AlertBand.darkRed => Ds.darkRed,
      };

  Color get bg => switch (this) {
        AlertBand.grey => Ds.greySoft,
        AlertBand.green => Ds.greenSoft,
        AlertBand.amber => Ds.amberSoft,
        AlertBand.red => Ds.redSoft,
        AlertBand.darkRed => Ds.darkRedSoft,
      };

  /// Guidance shown beside the band. Written from the clinician's side of the
  /// screen: what to do, not what the model computed.
  String get guidance => switch (this) {
        AlertBand.grey =>
          'No composite could be computed. This is missing evidence, not a low '
              'score — read the per-modality panel below to see which signals '
              'are absent or unusable.',
        AlertBand.green =>
          'No action indicated by the model. Continue the existing review interval.',
        AlertBand.amber =>
          'Elevated signal across one or more modalities. Consider bringing the next review forward.',
        AlertBand.red =>
          'Sustained elevated risk. Clinical review recommended before the next scheduled visit.',
        AlertBand.darkRed =>
          'Highest risk band. Same-day clinical review recommended.',
      };
}