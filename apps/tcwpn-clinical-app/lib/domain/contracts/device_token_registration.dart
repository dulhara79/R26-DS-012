class DeviceTokenRegistration {
  const DeviceTokenRegistration({
    required this.provider,
    required this.providerProjectId,
    required this.platform,
    required this.pushToken,
    required this.active,
  });

  final String provider;
  final String providerProjectId;
  final String platform;
  final String pushToken;
  final bool active;

  Map<String, dynamic> toJson() => {
        'provider': provider,
        'provider_project_id': providerProjectId,
        'platform': platform,
        'push_token': pushToken,
        'active': active,
      };

  @override
  bool operator ==(Object other) =>
      other is DeviceTokenRegistration &&
      other.provider == provider &&
      other.providerProjectId == providerProjectId &&
      other.platform == platform &&
      other.pushToken == pushToken &&
      other.active == active;

  @override
  int get hashCode => Object.hash(
        provider,
        providerProjectId,
        platform,
        pushToken,
        active,
      );
}
