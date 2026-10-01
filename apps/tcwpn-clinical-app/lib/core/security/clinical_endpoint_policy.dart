class ClinicalEndpointPolicy {
  const ClinicalEndpointPolicy._();

  static bool isAllowed(String rawUrl) {
    final value = rawUrl.trim();
    if (value.isEmpty) return true;

    final uri = Uri.tryParse(value);
    if (uri == null || uri.host.isEmpty) return false;

    final scheme = uri.scheme.toLowerCase();
    if (scheme == 'https') return true;
    if (scheme != 'http') return false;

    final host = uri.host.toLowerCase();
    return host == 'localhost' || host == '127.0.0.1' || host == '::1';
  }

  static bool isLoopback(String rawUrl) {
    final uri = Uri.tryParse(rawUrl.trim());
    if (uri == null) return false;
    final host = uri.host.toLowerCase();
    return host == 'localhost' || host == '127.0.0.1' || host == '::1';
  }
}
