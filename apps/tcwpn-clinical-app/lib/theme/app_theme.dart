import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppColors {
  // Primary teal palette — mental health / clinical standard
  static const Color primary        = Color(0xFF0D7B61);
  static const Color primaryLight   = Color(0xFF19B28D);
  static const Color primaryLighter = Color(0xFF5ED9B9);
  static const Color primarySurface = Color(0xFFF0FAF7);

  // Surfaces
  static const Color surface        = Color(0xFFFFFFFF);
  static const Color surfaceSecond  = Color(0xFFF9FBFB);
  static const Color surfaceThird   = Color(0xFFF2F7F6);
  static const Color border         = Color(0xFFE4F0EC);
  static const Color borderStrong   = Color(0xFFC8E0D8);

  // Text
  static const Color textPrimary    = Color(0xFF0A241E);
  static const Color textSecondary  = Color(0xFF45635B);
  static const Color textHint       = Color(0xFF90B0A8);

  // Risk levels — semantic
  static const Color riskLow        = Color(0xFF2E5B0C);
  static const Color riskLowBg      = Color(0xFFF0F7E9);
  static const Color riskModerate   = Color(0xFF7D4608);
  static const Color riskModerateBg = Color(0xFFFDF4E7);
  static const Color riskHigh       = Color(0xFF8C3418);
  static const Color riskHighBg     = Color(0xFFFDF1ED);
  static const Color riskVeryHigh   = Color(0xFF912424);
  static const Color riskVeryHighBg = Color(0xFFFEEFEF);

  // Accent
  static const Color warning        = Color(0xFFC77E1B);
  static const Color info           = Color(0xFF1B6DC7);
  static const Color infoBg         = Color(0xFFEDF4FD);

  // Glassmorphism simulation
  static BoxDecoration glass({double opacity = 0.8}) => BoxDecoration(
    color: Colors.white.withOpacity(opacity),
    borderRadius: BorderRadius.circular(16),
    border: Border.all(color: Colors.white.withOpacity(0.5), width: 1.5),
    boxShadow: [
      BoxShadow(
        color: Colors.black.withOpacity(0.05),
        blurRadius: 20,
        offset: const Offset(0, 10),
      ),
    ],
  );
}

class AppTheme {
  static ThemeData get light {
    final textTheme = GoogleFonts.outfitTextTheme();
    return ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(
        seedColor: AppColors.primary,
        primary: AppColors.primary,
        surface: AppColors.surface,
        surfaceContainerHighest: AppColors.surfaceSecond,
      ),
      textTheme: textTheme.copyWith(
        displayLarge: GoogleFonts.outfit(fontSize: 32, fontWeight: FontWeight.w700, color: AppColors.textPrimary),
        headlineMedium: GoogleFonts.outfit(fontSize: 24, fontWeight: FontWeight.w700, color: AppColors.textPrimary),
        headlineSmall: GoogleFonts.outfit(fontSize: 20, fontWeight: FontWeight.w600, color: AppColors.textPrimary),
        titleLarge: GoogleFonts.outfit(fontSize: 18, fontWeight: FontWeight.w600, color: AppColors.textPrimary),
        titleMedium: GoogleFonts.outfit(fontSize: 16, fontWeight: FontWeight.w500, color: AppColors.textPrimary),
        titleSmall: GoogleFonts.outfit(fontSize: 14, fontWeight: FontWeight.w500, color: AppColors.textSecondary),
        bodyLarge: GoogleFonts.outfit(fontSize: 16, fontWeight: FontWeight.w400, color: AppColors.textPrimary, height: 1.5),
        bodyMedium: GoogleFonts.outfit(fontSize: 14, fontWeight: FontWeight.w400, color: AppColors.textPrimary, height: 1.5),
        bodySmall: GoogleFonts.outfit(fontSize: 12, fontWeight: FontWeight.w400, color: AppColors.textSecondary),
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: AppColors.surface,
        foregroundColor: AppColors.textPrimary,
        elevation: 0,
        titleTextStyle: GoogleFonts.outfit(fontSize: 20, fontWeight: FontWeight.w600, color: AppColors.textPrimary),
      ),
      cardTheme: CardThemeData(
        color: AppColors.surface,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
          side: const BorderSide(color: AppColors.border, width: 1),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: AppColors.primary,
          foregroundColor: Colors.white,
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          textStyle: GoogleFonts.outfit(fontSize: 16, fontWeight: FontWeight.w600),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: Colors.white,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.primary, width: 2),
        ),
      ),
      bottomNavigationBarTheme: BottomNavigationBarThemeData(
        backgroundColor: AppColors.surface,
        selectedItemColor: AppColors.primary,
        unselectedItemColor: AppColors.textHint,
        selectedLabelStyle: GoogleFonts.outfit(fontSize: 12, fontWeight: FontWeight.w600),
        unselectedLabelStyle: GoogleFonts.outfit(fontSize: 12, fontWeight: FontWeight.w500),
        type: BottomNavigationBarType.fixed,
        elevation: 0,
      ),
    );
  }
}
