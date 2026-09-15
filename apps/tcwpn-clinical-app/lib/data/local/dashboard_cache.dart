import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

import '../../domain/contracts/dashboard_snapshot.dart';

class DashboardCacheStore {
  static const _key = 'server_dashboard_v1';

  const DashboardCacheStore();

  Future<void> save(DashboardSnapshot snapshot) async {
    final preferences = await SharedPreferences.getInstance();
    await preferences.setString(_key, jsonEncode(snapshot.toJson()));
  }

  Future<DashboardSnapshot?> load() async {
    final preferences = await SharedPreferences.getInstance();
    final raw = preferences.getString(_key);
    if (raw == null || raw.isEmpty) return null;
    try {
      final decoded = jsonDecode(raw);
      if (decoded is! Map) return null;
      return DashboardSnapshot.fromJson(Map<String, dynamic>.from(decoded));
    } catch (_) {
      return null;
    }
  }

  Future<void> clear() async {
    final preferences = await SharedPreferences.getInstance();
    await preferences.remove(_key);
  }
}
