class ServiceConfig {
  static const String googleScriptUrl = String.fromEnvironment(
    'SCRIPT_URL',
    defaultValue:
        "https://script.google.com/macros/s/AKfycbzZlkyTLxoJgHbV1IqXi4ugXIC9GM5a_MIgkhWEMkA9b-_25wowCmNdOyJjylHONLnl/exec",
  );

  static const String authToken = String.fromEnvironment(
    'AUTH_TOKEN',
    defaultValue: "7c09db655b5f697a4faf0b18a517d5fb",
  );

  // Notification Channels
  static const String channelId = 'research_channel_01';
  static const String channelName = 'Data Collection Service';
  static const int notificationId = 888;

  // ── NHSL / PDPA Compliance Metadata ────────────────────────
  // TODO: Replace placeholder values before ERC submission

  /// Consent document version — increment when consent text changes
  static const String consentVersion = '1.0';

  /// Date the consent was last updated (YYYY-MM-DD)
  static const String consentDate = '2026-05-08';

  /// Study title as approved by ERC
  static const String studyTitle =
      'Digital Biomarkers of Anxiety: A Mobile Sensing Study';

  /// Principal Investigator
  static const String piName = '[Prof. Samantha Thelijjagoda]';
  static const String piAffiliation = 'Sri Lanka Institute of Information Technology (SLIIT)';
  static const String piEmail = '[samantha.t@sliit.lk]';

  /// Research Supervisor
  static const String supervisorName = '[Prof. Samantha Thelijjagoda]';
  static const String supervisorEmail = '[samantha.t@sliit.lk]';

  /// Ethics Review Committee
  static const String ercName = 'Ethics Review Committee, Faculty of Medicine, University of Colombo';
  static const String ercApprovalNumber = '[ERC-XXXX/XX/XXX — UPDATE AFTER APPROVAL]';
  static const String ercSecretaryEmail = '[erc-secretary@cmb.ac.lk — UPDATE]';

  /// Research team contact (for participant queries)
  static const String researchTeamEmail = '[it22130648@my.sliit.lk, it22171542@my.sliit.lk, it22107596@my.sliit.lk, it22093950@my.sliit.lk]';

  /// Data retention period
  static const String dataRetentionPeriod = '1 years after study completion and publication';

  /// Data controller (legal entity responsible for the data)
  static const String dataController = 'SLIIT Research Team, SLIIT Malabe Campus';
}
