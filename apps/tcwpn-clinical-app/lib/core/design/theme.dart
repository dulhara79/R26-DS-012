// lib/core/design/theme.dart
//
// The one ThemeData. Both research components render through this — there is no
// second theme, no per-module palette class, and no screen that hard-codes a
// colour.
//
// Type system (three roles, deliberately):
//   display  — Inter Tight, tight tracking. Screen titles and hero numbers.
//   body     — Inter. Everything a clinician reads as prose.
//   data     — IBM Plex Mono, tabular figures. Risk scores, thresholds, weights,
//              latency, patient MRNs. Anything that is a measurement.
//
// The mono register is the point: a risk score of 0.7134 is instrument output,
// and setting it in the same face as body copy makes it look like an opinion.

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

import 'tokens.dart';

class AppTheme {
  static TextStyle data({
    double size = 13,
    FontWeight weight = FontWeight.w500,
    Color color = Ds.ink,
    double? height,
    double letterSpacing = 0,
  }) =>
      GoogleFonts.ibmPlexMono(
        fontSize: size,
        fontWeight: weight,
        color: color,
        height: height,
        letterSpacing: letterSpacing,
        fontFeatures: const [FontFeature.tabularFigures()],
      );

  static TextStyle display({
    double size = 22,
    FontWeight weight = FontWeight.w700,
    Color color = Ds.ink,
    double letterSpacing = -0.4,
    double? height,
  }) =>
      GoogleFonts.interTight(
        fontSize: size,
        fontWeight: weight,
        color: color,
        letterSpacing: letterSpacing,
        height: height,
      );

  /// Small uppercase structural label. Used for section eyebrows where the
  /// label genuinely classifies the block below it.
  static TextStyle get eyebrow => GoogleFonts.inter(
        fontSize: 11,
        fontWeight: FontWeight.w600,
        color: Ds.inkFaint,
        letterSpacing: 0.9,
      );

  static ThemeData get light {
    final body = GoogleFonts.interTextTheme();

    return ThemeData(
      useMaterial3: true,
      scaffoldBackgroundColor: Ds.canvas,
      splashFactory: InkSparkle.splashFactory,
      colorScheme: ColorScheme.fromSeed(
        seedColor: Ds.brand,
        brightness: Brightness.light,
        primary: Ds.brand,
        onPrimary: Colors.white,
        surface: Ds.surface,
        onSurface: Ds.ink,
        error: Ds.red,
      ),
      textTheme: body.copyWith(
        displaySmall: display(size: 30, letterSpacing: -0.8),
        headlineSmall: display(size: 22),
        titleLarge: display(size: 17, weight: FontWeight.w700, letterSpacing: -0.2),
        titleMedium: GoogleFonts.inter(
            fontSize: 14, fontWeight: FontWeight.w600, color: Ds.ink),
        titleSmall: GoogleFonts.inter(
            fontSize: 12.5, fontWeight: FontWeight.w600, color: Ds.inkMuted),
        bodyLarge: GoogleFonts.inter(fontSize: 14.5, color: Ds.ink, height: 1.5),
        bodyMedium: GoogleFonts.inter(fontSize: 13.5, color: Ds.inkMuted, height: 1.5),
        bodySmall: GoogleFonts.inter(fontSize: 12, color: Ds.inkFaint, height: 1.4),
        labelLarge: GoogleFonts.inter(fontSize: 14, fontWeight: FontWeight.w600),
        labelSmall: eyebrow,
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: Ds.surface,
        surfaceTintColor: Colors.transparent,
        foregroundColor: Ds.ink,
        elevation: 0,
        scrolledUnderElevation: 0,
        centerTitle: false,
        titleTextStyle: display(size: 17, weight: FontWeight.w700, letterSpacing: -0.2),
        systemOverlayStyle: SystemUiOverlayStyle.dark,
      ),
      dividerTheme: const DividerThemeData(
          color: Ds.hairline, thickness: 1, space: 1),
      cardTheme: CardThemeData(
        color: Ds.surface,
        elevation: 0,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(Ds.rLg),
          side: const BorderSide(color: Ds.hairline),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: Ds.brand,
          foregroundColor: Colors.white,
          disabledBackgroundColor: Ds.surfaceSunken,
          disabledForegroundColor: Ds.inkFaint,
          minimumSize: const Size(double.infinity, 50),
          elevation: 0,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(Ds.rMd)),
          textStyle: GoogleFonts.inter(fontSize: 14.5, fontWeight: FontWeight.w600),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: Ds.brand,
          minimumSize: const Size(double.infinity, 50),
          side: const BorderSide(color: Ds.hairlineStrong),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(Ds.rMd)),
          textStyle: GoogleFonts.inter(fontSize: 14.5, fontWeight: FontWeight.w600),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: Ds.brand,
          textStyle: GoogleFonts.inter(fontSize: 13.5, fontWeight: FontWeight.w600),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: Ds.surface,
        contentPadding:
            const EdgeInsets.symmetric(horizontal: Ds.s4, vertical: Ds.s4),
        hintStyle: GoogleFonts.inter(fontSize: 13.5, color: Ds.inkFaint),
        labelStyle: GoogleFonts.inter(fontSize: 13.5, color: Ds.inkMuted),
        floatingLabelStyle: GoogleFonts.inter(
            fontSize: 12.5, color: Ds.brand, fontWeight: FontWeight.w600),
        border: _border(Ds.hairline),
        enabledBorder: _border(Ds.hairline),
        focusedBorder: _border(Ds.brand, width: 1.6),
        errorBorder: _border(Ds.red),
        focusedErrorBorder: _border(Ds.red, width: 1.6),
      ),
      navigationBarTheme: NavigationBarThemeData(
        backgroundColor: Ds.surface,
        surfaceTintColor: Colors.transparent,
        indicatorColor: Ds.brandSoft,
        height: 68,
        labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
        labelTextStyle: WidgetStateProperty.resolveWith(
          (s) => GoogleFonts.inter(
            fontSize: 11.5,
            fontWeight: s.contains(WidgetState.selected)
                ? FontWeight.w600
                : FontWeight.w500,
            color: s.contains(WidgetState.selected) ? Ds.brand : Ds.inkMuted,
          ),
        ),
        iconTheme: WidgetStateProperty.resolveWith(
          (s) => IconThemeData(
            size: 22,
            color: s.contains(WidgetState.selected) ? Ds.brand : Ds.inkMuted,
          ),
        ),
      ),
      snackBarTheme: SnackBarThemeData(
        behavior: SnackBarBehavior.floating,
        backgroundColor: Ds.ink,
        contentTextStyle: GoogleFonts.inter(fontSize: 13.5, color: Colors.white),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(Ds.rMd)),
      ),
      dialogTheme: DialogThemeData(
        backgroundColor: Ds.surface,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(Ds.rLg)),
        titleTextStyle: display(size: 17),
      ),
      bottomSheetTheme: const BottomSheetThemeData(
        backgroundColor: Ds.surface,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(Ds.rXl)),
        ),
      ),
      listTileTheme: const ListTileThemeData(
        contentPadding: EdgeInsets.symmetric(horizontal: Ds.s4, vertical: Ds.s1),
      ),
    );
  }

  static OutlineInputBorder _border(Color c, {double width = 1}) =>
      OutlineInputBorder(
        borderRadius: BorderRadius.circular(Ds.rMd),
        borderSide: BorderSide(color: c, width: width),
      );
}
