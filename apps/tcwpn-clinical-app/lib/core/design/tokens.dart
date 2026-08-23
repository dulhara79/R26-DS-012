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

  // ── Component identity colours (used only in fusion breakdowns) ──────────
  // Each modality keeps one colour everywhere it appears, so a clinician learns
  // "the sand band is the wearable" once and it holds across every screen.
  static const Color c1Physiological = Color(0xFF6E7FA8); // wearable
  static const Color c2Behavioral = Color(0xFF7FA88C); // phone sensing
  static const Color c3Intervention = Color(0xFFC0956B); // intervention engine
  static const Color c4ClinicalNlp = Color(0xFF0F5B6E); // TC-WPN — the brand hue

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

/// Alert bands from proposal §5.1. Thresholds live here alone so the app and
/// the fusion service can be reconciled in one place.
enum AlertBand { green, amber, red, darkRed }

extension AlertBandX on AlertBand {
  static AlertBand fromScore(double s) {
    if (s >= 0.75) return AlertBand.darkRed;
    if (s >= 0.50) return AlertBand.red;
    if (s >= 0.25) return AlertBand.amber;
    return AlertBand.green;
  }

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
      default:
        return AlertBand.green;
    }
  }

  String get label => switch (this) {
        AlertBand.green => 'Stable',
        AlertBand.amber => 'Monitor',
        AlertBand.red => 'Review',
        AlertBand.darkRed => 'Urgent',
      };

  /// The band name as the proposal writes it — used in exports and PDFs where
  /// the protocol vocabulary matters more than the friendly label.
  String get protocolName => switch (this) {
        AlertBand.green => 'GREEN',
        AlertBand.amber => 'AMBER',
        AlertBand.red => 'RED',
        AlertBand.darkRed => 'DARK RED',
      };

  Color get fg => switch (this) {
        AlertBand.green => Ds.green,
        AlertBand.amber => Ds.amber,
        AlertBand.red => Ds.red,
        AlertBand.darkRed => Ds.darkRed,
      };

  Color get bg => switch (this) {
        AlertBand.green => Ds.greenSoft,
        AlertBand.amber => Ds.amberSoft,
        AlertBand.red => Ds.redSoft,
        AlertBand.darkRed => Ds.darkRedSoft,
      };

  /// Guidance shown beside the band. Written from the clinician's side of the
  /// screen: what to do, not what the model computed.
  String get guidance => switch (this) {
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
