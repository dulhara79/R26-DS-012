/// Classifies QR values without giving a participant identifier any authority.
enum PatientQrKind { invitation, participantId, unsupported }

class PatientQrPayload {
  const PatientQrPayload(this.kind, [this.value]);

  final PatientQrKind kind;
  final String? value;
}

/// The central backend issues invitation codes with secrets.token_urlsafe(32):
/// 43 base64url characters. Manual entry remains more permissive for existing
/// backend contracts; QR scanning intentionally accepts only this known format.
///
/// Aura's existing QR encodes P_<16 hex> (or a legacy clinanx://patient/
/// wrapper). That ID is NOT a patient-authorized clinician invitation.
PatientQrPayload parsePatientInvitationQr(String? raw) {
  final text = raw?.trim() ?? '';
  if (text.isEmpty || text.length > 512) {
    return const PatientQrPayload(PatientQrKind.unsupported);
  }

  final participant = RegExp(r'^P_[A-F0-9]{16}$');
  final invite = RegExp(r'^[A-Za-z0-9_-]{43}$');

  if (participant.hasMatch(text.toUpperCase())) {
    return PatientQrPayload(PatientQrKind.participantId, text.toUpperCase());
  }

  final uri = Uri.tryParse(text);
  if (uri != null && uri.scheme.toLowerCase() == 'clinanx') {
    // Explicit application QR wrappers only. Reject queries, fragments and
    // extra segments rather than accepting a token from an arbitrary URL.
    if (uri.hasQuery || uri.hasFragment || uri.userInfo.isNotEmpty ||
        uri.hasPort || uri.pathSegments.length != 1) {
      return const PatientQrPayload(PatientQrKind.unsupported);
    }
    final segment = uri.pathSegments.single;
    if (uri.host.toLowerCase() == 'patient' &&
        participant.hasMatch(segment.toUpperCase())) {
      return PatientQrPayload(PatientQrKind.participantId, segment.toUpperCase());
    }
    if (uri.host.toLowerCase() == 'invite' && invite.hasMatch(segment)) {
      return PatientQrPayload(PatientQrKind.invitation, segment);
    }
    return const PatientQrPayload(PatientQrKind.unsupported);
  }

  if (invite.hasMatch(text)) {
    return PatientQrPayload(PatientQrKind.invitation, text);
  }
  return const PatientQrPayload(PatientQrKind.unsupported);
}
