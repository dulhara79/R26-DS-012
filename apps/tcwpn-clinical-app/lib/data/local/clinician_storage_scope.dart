/// Namespaces every SharedPreferences-backed clinical cache by clinician.
///
/// The scope is bound by [Session] on sign-in / warm-start and cleared on
/// sign-out. Stores use [keyForCurrent] so legacy call sites cannot accidentally
/// read another clinician's data. When no clinician is bound, data goes into a
/// deliberately isolated non-clinical namespace; it is never treated as a
/// fallback for a real clinician account.
class ClinicianStorageScope {
  static const String _unboundScopeId = '__NO_CLINICIAN__';
  static String? _boundScopeId;

  final String scopeId;

  ClinicianStorageScope(String clinicianId)
      : scopeId = _normalizeRequired(clinicianId);

  static String _normalizeRequired(String clinicianId) {
    final normalized = clinicianId.trim().toUpperCase();
    if (normalized.isEmpty) {
      throw ArgumentError.value(
        clinicianId,
        'clinicianId',
        'Clinician storage scope must not be blank.',
      );
    }
    return Uri.encodeComponent(normalized);
  }

  static void bind(String clinicianId) {
    _boundScopeId = _normalizeRequired(clinicianId);
  }

  static void clear() {
    _boundScopeId = null;
  }

  static String? get boundScopeId => _boundScopeId;

  static String keyForCurrent(String logicalKey) =>
      _key(_boundScopeId ?? _unboundScopeId, logicalKey);

  String key(String logicalKey) => _key(scopeId, logicalKey);

  static String _key(String scopeId, String logicalKey) {
    final logical = logicalKey.trim();
    if (logical.isEmpty) {
      throw ArgumentError.value(
        logicalKey,
        'logicalKey',
        'Storage key must not be blank.',
      );
    }
    return 'clinanx_v1::$scopeId::$logical';
  }
}
