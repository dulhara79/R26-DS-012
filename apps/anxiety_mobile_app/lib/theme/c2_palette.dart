import 'package:flutter/material.dart';

import 'theme_controller.dart';

/// Shared colour tokens for the Component 2 (behavioural context) pages.
///
/// Both the Activity tab and the sensing-details page read from here so the
/// two screens look identical in light, dark and scheduled theme modes. The
/// values follow the app-wide palette in [AppTheme] (`kPrimaryDeep` purple).
class C2Palette {
  static bool get _dark => ThemeController.instance.isDarkNow;

  static Color get scaffold =>
      _dark ? const Color(0xFF111218) : const Color(0xFFF5F3FF);
  static Color get cardBase =>
      _dark ? const Color(0xFF1A1B24) : const Color(0xFFFFFFFF);
  static Color get chip =>
      _dark ? const Color(0xFF29243B) : const Color(0xFFF0ECFF);

  static Color get p500 =>
      _dark ? const Color(0xFFB8B6FF) : const Color(0xFF5E60CE);
  static Color get p400 =>
      _dark ? const Color(0xFFC6B4FF) : const Color(0xFF7C5CBF);
  static Color get p200 =>
      _dark ? const Color(0xFF655A88) : const Color(0xFFC4B5FD);
  static Color get p100 =>
      _dark ? const Color(0xFF29243B) : const Color(0xFFF0ECFF);

  static Color get primary => p500;
  static Color get amber =>
      _dark ? const Color(0xFFFFB75D) : const Color(0xFFF59B24);
  static Color get amberBg =>
      _dark ? const Color(0xFF3A2B16) : const Color(0xFFFEF3DC);
  static Color get rose =>
      _dark ? const Color(0xFFFF8299) : const Color(0xFFEF5777);
  static Color get roseBg =>
      _dark ? const Color(0xFF3B2028) : const Color(0xFFFDEAEE);
  static Color get teal =>
      _dark ? const Color(0xFF5ED7C7) : const Color(0xFF0F9D8C);
  static Color get tealBg =>
      _dark ? const Color(0xFF173633) : const Color(0xFFE3F5F2);

  static Color get textPrimary =>
      _dark ? const Color(0xFFF3F1FA) : const Color(0xFF2D3142);
  static Color get textSecondary =>
      _dark ? const Color(0xFFC9C5D6) : const Color(0xFF5A607F);
  static Color get textMuted =>
      _dark ? const Color(0xFFA9A4B7) : const Color(0xFF9095A7);
  static Color get border =>
      _dark ? const Color(0xFF383643) : const Color(0xFFE8E5F4);
}
