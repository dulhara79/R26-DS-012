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
    return ThemeData(
      useMaterial3: true,
      scaffoldBackgroundColor: kBgTop,
      textTheme: GoogleFonts.poppinsTextTheme(),
      colorScheme: ColorScheme.fromSeed(seedColor: kPrimaryDeep),
    );
  }
}
