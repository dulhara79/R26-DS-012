import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppColors {
  // Primary teal palette — mental health / clinical standard
  static const Color primary        = Color(0xFF0F6E56);
  static const Color primaryLight   = Color(0xFF1D9E75);
  static const Color primaryLighter = Color(0xFF5DCAA5);
  static const Color primarySurface = Color(0xFFE1F5EE);

  // Surfaces
  static const Color surface        = Color(0xFFFFFFFF);
  static const Color surfaceSecond  = Color(0xFFF7FAFA);
  static const Color surfaceThird   = Color(0xFFEFF8F5);
  static const Color border         = Color(0xFFDCEDE7);
  static const Color borderStrong   = Color(0xFFB0D5C8);

  // Text
  static const Color textPrimary    = Color(0xFF0D2B22);
  static const Color textSecondary  = Color(0xFF4A7064);
  static const Color textHint       = Color(0xFF8AADA5);

  // Risk levels — semantic
  static const Color riskLow        = Color(0xFF27500A);  // green 800
  static const Color riskLowBg      = Color(0xFFEAF3DE);  // green 50
  static const Color riskModerate   = Color(0xFF633806);  // amber 800
  static const Color riskModerateBg = Color(0xFFFAEEDA);  // amber 50
  static const Color riskHigh       = Color(0xFF712B13);  // coral 800
  static const Color riskHighBg     = Color(0xFFFAECE7);  // coral 50
  static const Color riskVeryHigh   = Color(0xFF791F1F);  // red 800
  static const Color riskVeryHighBg = Color(0xFFFCEBEB);  // red 50

  // Accent
  static const Color warning        = Color(0xFFBA7517);
  static const Color warningBg      = Color(0xFFFAEEDA);
  static const Color info           = Color(0xFF185FA5);
  static const Color infoBg         = Color(0xFFE6F1FB);
}

class AppTheme {
  static ThemeData get light {
    return ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme(
        brightness: Brightness.light,
        primary:          AppColors.primary,
        onPrimary:        Colors.white,
        primaryContainer: AppColors.primarySurface,
        onPrimaryContainer: AppColors.primary,
        secondary:        AppColors.primaryLight,
        onSecondary:      Colors.white,
        secondaryContainer: AppColors.primarySurface,
        onSecondaryContainer: AppColors.primary,
        surface:          AppColors.surface,
        onSurface:        AppColors.textPrimary,
        surfaceContainerHighest: AppColors.surfaceSecond,
        error:            AppColors.riskVeryHigh,
        onError:          Colors.white,
        outline:          AppColors.border,
        outlineVariant:   AppColors.borderStrong,
      ),
      textTheme: GoogleFonts.interTextTheme().copyWith(
        displayLarge: GoogleFonts.inter(fontSize: 32, fontWeight: FontWeight.w600, color: AppColors.textPrimary, letterSpacing: -0.5),
        headlineMedium: GoogleFonts.inter(fontSize: 22, fontWeight: FontWeight.w600, color: AppColors.textPrimary, letterSpacing: -0.3),
        headlineSmall: GoogleFonts.inter(fontSize: 18, fontWeight: FontWeight.w600, color: AppColors.textPrimary),
        titleLarge: GoogleFonts.inter(fontSize: 16, fontWeight: FontWeight.w600, color: AppColors.textPrimary),
        titleMedium: GoogleFonts.inter(fontSize: 15, fontWeight: FontWeight.w500, color: AppColors.textPrimary),
        titleSmall: GoogleFonts.inter(fontSize: 13, fontWeight: FontWeight.w500, color: AppColors.textSecondary),
        bodyLarge: GoogleFonts.inter(fontSize: 15, fontWeight: FontWeight.w400, color: AppColors.textPrimary, height: 1.6),
        bodyMedium: GoogleFonts.inter(fontSize: 14, fontWeight: FontWeight.w400, color: AppColors.textPrimary, height: 1.5),
        bodySmall: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w400, color: AppColors.textSecondary, height: 1.4),
        labelLarge: GoogleFonts.inter(fontSize: 14, fontWeight: FontWeight.w500, color: AppColors.textPrimary, letterSpacing: 0.1),
        labelSmall: GoogleFonts.inter(fontSize: 11, fontWeight: FontWeight.w500, color: AppColors.textSecondary, letterSpacing: 0.5),
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: AppColors.surface,
        foregroundColor: AppColors.textPrimary,
        elevation: 0,
        surfaceTintColor: Colors.transparent,
        centerTitle: false,
        titleTextStyle: GoogleFonts.inter(
          fontSize: 18, fontWeight: FontWeight.w600,
          color: AppColors.textPrimary,
        ),
        iconTheme: const IconThemeData(color: AppColors.textPrimary, size: 22),
      ),
      cardTheme: CardThemeData(
        color: AppColors.surface,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: const BorderSide(color: AppColors.border, width: 0.8),
        ),
        margin: EdgeInsets.zero,
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: AppColors.primary,
          foregroundColor: Colors.white,
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          textStyle: GoogleFonts.inter(fontSize: 15, fontWeight: FontWeight.w600),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: AppColors.primary,
          side: const BorderSide(color: AppColors.primary, width: 1.2),
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          textStyle: GoogleFonts.inter(fontSize: 14, fontWeight: FontWeight.w500),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: AppColors.surfaceSecond,
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppColors.border, width: 0.8),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppColors.border, width: 0.8),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppColors.primary, width: 1.5),
        ),
        hintStyle: GoogleFonts.inter(fontSize: 14, color: AppColors.textHint),
        labelStyle: GoogleFonts.inter(fontSize: 14, color: AppColors.textSecondary),
      ),
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        backgroundColor: AppColors.surface,
        selectedItemColor: AppColors.primary,
        unselectedItemColor: AppColors.textHint,
        elevation: 0,
        type: BottomNavigationBarType.fixed,
      ),
      dividerTheme: const DividerThemeData(
        color: AppColors.border,
        thickness: 0.8,
        space: 0,
      ),
      scaffoldBackgroundColor: AppColors.surfaceSecond,
    );
  }
}
