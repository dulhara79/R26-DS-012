import 'contract_parsing.dart';

class ClinicianPrincipal {
  final String clinicianId;
  final String? displayName;
  final String? role;
  final DateTime? expiresAt;

  const ClinicianPrincipal({
    required this.clinicianId,
    required this.displayName,
    required this.role,
    required this.expiresAt,
  });

  bool get isValid => clinicianId.trim().isNotEmpty;

  factory ClinicianPrincipal.fromJson(Map<String, dynamic> json) =>
      ClinicianPrincipal(
        clinicianId: contractString(json['clinician_id'] ?? json['sub']) ?? '',
        displayName: contractString(json['display_name'] ?? json['name']),
        role: contractString(json['role']),
        expiresAt: contractDateTime(json['expires_at'] ?? json['exp']),
      );
}
