import 'package:shared_preferences/shared_preferences.dart';

import 'clinician_storage_scope.dart';

/// Local notification-delivery bookkeeping only.
///
/// This store never represents the server AttentionEvent lifecycle. In
/// particular, a delivered/dismissed notification is not an acknowledgement or
/// resolution. The Central Backend remains authoritative for clinical state.
abstract interface class AttentionNotificationStore {
  Future<bool> wasDelivered(String eventId);
  Future<void> markDelivered(String eventId);
  Future<void> savePendingOpen(String eventId);
  Future<String?> takePendingOpen();
}

class SharedPreferencesAttentionNotificationStore
    implements AttentionNotificationStore {
  static const _pendingOpenLogicalKey =
      'attention_notification_pending_open_v1';
  static const _deliveredLogicalKey =
      'attention_notification_delivered_v1';
  static const _maxDeliveredIds = 512;

  final String deliveryScope;

  const SharedPreferencesAttentionNotificationStore({
    required this.deliveryScope,
  });

  ClinicianStorageScope get _scope => ClinicianStorageScope(deliveryScope);

  String get _deliveredKey => _scope.key(_deliveredLogicalKey);
  String get _pendingOpenKey => _scope.key(_pendingOpenLogicalKey);

  Future<SharedPreferences> get _prefs => SharedPreferences.getInstance();

  @override
  Future<bool> wasDelivered(String eventId) async {
    if (eventId.isEmpty) return false;
    final ids = (await _prefs).getStringList(_deliveredKey) ?? const <String>[];
    return ids.contains(eventId);
  }

  @override
  Future<void> markDelivered(String eventId) async {
    if (eventId.isEmpty) return;
    final prefs = await _prefs;
    final existing = prefs.getStringList(_deliveredKey) ?? <String>[];
    if (existing.contains(eventId)) return;

    final updated = <String>[...existing, eventId];
    final bounded = updated.length <= _maxDeliveredIds
        ? updated
        : updated.sublist(updated.length - _maxDeliveredIds);
    await prefs.setStringList(_deliveredKey, bounded);
  }

  @override
  Future<void> savePendingOpen(String eventId) async {
    if (eventId.isEmpty) return;
    await (await _prefs).setString(_pendingOpenKey, eventId);
  }

  @override
  Future<String?> takePendingOpen() async {
    final prefs = await _prefs;
    final eventId = prefs.getString(_pendingOpenKey);
    if (eventId != null) {
      await prefs.remove(_pendingOpenKey);
    }
    return (eventId == null || eventId.isEmpty) ? null : eventId;
  }
}
