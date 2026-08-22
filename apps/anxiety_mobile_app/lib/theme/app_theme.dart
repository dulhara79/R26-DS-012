import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  // Calming Color Palette
  static const Color kBgTop = Color(0xFFFDFBFB);
  static const Color kBgBottom = Color(0xFFEBEDEE);
  static const Color kAccentBlue = Color(0xFF89CFF0); // Baby Blue
  static const Color kAccentLavender = Color(0xFFE0C3FC); // Soft Lavender
  static const Color kPrimaryDeep = Color(0xFF5E60CE); // Deep calming purple
  static const Color kTextDark = Color(0xFF2D3142);
  static const Color kTextLight = Color(0xFF9095A7);

  static ThemeData get lightTheme {
    final colorScheme = ColorScheme.fromSeed(seedColor: kPrimaryDeep);
    return ThemeData(
      useMaterial3: true,
      scaffoldBackgroundColor: kBgTop,
      textTheme: GoogleFonts.poppinsTextTheme(),
      colorScheme: colorScheme,
      appBarTheme: const AppBarTheme(
        backgroundColor: Colors.transparent,
        surfaceTintColor: Colors.transparent,
      ),
    );
  }

  static ThemeData get darkTheme {
    final colorScheme = ColorScheme.fromSeed(
      seedColor: kPrimaryDeep,
      brightness: Brightness.dark,
      surface: const Color(0xFF1A1B24),
    );
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: const Color(0xFF111218),
      colorScheme: colorScheme,
      textTheme: GoogleFonts.poppinsTextTheme(ThemeData.dark().textTheme),
      appBarTheme: const AppBarTheme(
        backgroundColor: Colors.transparent,
        surfaceTintColor: Colors.transparent,
      ),
      cardTheme: const CardThemeData(
        color: Color(0xFF1A1B24),
        surfaceTintColor: Colors.transparent,
      ),
      dialogTheme: const DialogThemeData(backgroundColor: Color(0xFF1A1B24)),
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        backgroundColor: Color(0xFF1A1B24),
      ),
    );
  }
}
