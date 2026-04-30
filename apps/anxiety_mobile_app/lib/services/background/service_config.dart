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
}
