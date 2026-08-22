import 'dart:async';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';

enum AppThemeMode { system, light, dark, scheduled }

extension AppThemeModeLabel on AppThemeMode {
  String get label {
    switch (this) {
      case AppThemeMode.system:
        return 'Use phone settings';
      case AppThemeMode.light:
        return 'Light';
      case AppThemeMode.dark:
        return 'Dark';
      case AppThemeMode.scheduled:
        return 'Scheduled';
    }
  }
}

class ThemeController extends ChangeNotifier with WidgetsBindingObserver {
  ThemeController._();

  static final ThemeController instance = ThemeController._();

  static const _modeKey = 'app_theme_mode';
  static const _darkStartKey = 'dark_theme_start_minutes';
  static const _darkEndKey = 'dark_theme_end_minutes';

  AppThemeMode _mode = AppThemeMode.system;
  int _darkStartMinutes = 20 * 60;
  int _darkEndMinutes = 7 * 60;
  bool _initialized = false;

  AppThemeMode get mode => _mode;
  TimeOfDay get darkStart => _fromMinutes(_darkStartMinutes);
  TimeOfDay get darkEnd => _fromMinutes(_darkEndMinutes);

  ThemeMode get themeMode {
    switch (_mode) {
      case AppThemeMode.system:
        return ThemeMode.system;
      case AppThemeMode.light:
        return ThemeMode.light;
      case AppThemeMode.dark:
        return ThemeMode.dark;
      case AppThemeMode.scheduled:
        return _isScheduledDark(DateTime.now())
            ? ThemeMode.dark
            : ThemeMode.light;
    }
  }

  bool get isDarkNow {
    if (_mode == AppThemeMode.system) {
      return PlatformDispatcher.instance.platformBrightness == Brightness.dark;
    }
    return themeMode == ThemeMode.dark;
  }

  Future<void> initialize() async {
    if (_initialized) return;
    final prefs = await SharedPreferences.getInstance();
    final savedMode = prefs.getString(_modeKey);
    _mode = AppThemeMode.values.firstWhere(
      (value) => value.name == savedMode,
      orElse: () => AppThemeMode.system,
    );
    _darkStartMinutes = prefs.getInt(_darkStartKey) ?? 20 * 60;
    _darkEndMinutes = prefs.getInt(_darkEndKey) ?? 7 * 60;
    WidgetsBinding.instance.addObserver(this);
    Timer.periodic(const Duration(minutes: 1), (_) => _refreshTheme());
    _initialized = true;
    _applySystemBars();
  }

  Future<void> setMode(AppThemeMode value) async {
    if (_mode == value) return;
    _mode = value;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_modeKey, value.name);
    _refreshTheme();
  }

  Future<void> setSchedule({
    required TimeOfDay start,
    required TimeOfDay end,
  }) async {
    _darkStartMinutes = _toMinutes(start);
    _darkEndMinutes = _toMinutes(end);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(_darkStartKey, _darkStartMinutes);
    await prefs.setInt(_darkEndKey, _darkEndMinutes);
    _refreshTheme();
  }

  @override
  void didChangePlatformBrightness() {
    if (_mode == AppThemeMode.system) _refreshTheme();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) _refreshTheme();
  }

  void _refreshTheme() {
    _applySystemBars();
    notifyListeners();
  }

  bool _isScheduledDark(DateTime now) {
    final current = now.hour * 60 + now.minute;
    return isDarkAtMinutes(
      current: current,
      start: _darkStartMinutes,
      end: _darkEndMinutes,
    );
  }

  static bool isDarkAtMinutes({
    required int current,
    required int start,
    required int end,
  }) {
    if (start == end) return true;
    if (start < end) {
      return current >= start && current < end;
    }
    return current >= start || current < end;
  }

  void _applySystemBars() {
    final dark = isDarkNow;
    SystemChrome.setSystemUIOverlayStyle(
      SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: dark ? Brightness.light : Brightness.dark,
        statusBarBrightness: dark ? Brightness.dark : Brightness.light,
        systemNavigationBarColor: dark ? const Color(0xFF111218) : Colors.white,
        systemNavigationBarIconBrightness: dark
            ? Brightness.light
            : Brightness.dark,
      ),
    );
  }

  static int _toMinutes(TimeOfDay time) => time.hour * 60 + time.minute;

  static TimeOfDay _fromMinutes(int minutes) =>
      TimeOfDay(hour: (minutes ~/ 60) % 24, minute: minutes % 60);
}
