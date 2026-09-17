import 'env.dart';

class ResearchBuildInfo {
  const ResearchBuildInfo._();

  static Map<String, String> get values => <String, String>{
        'app_version': Env.appVersion,
        'build_environment': Env.buildEnvironment,
        'build_revision': Env.buildRevision,
        'backend_host': _hostOrState(Env.backendBase),
        'auth_mode': Env.hasRemoteAuth ? 'remote' : 'local-demo',
        'firebase_slot': Env.pushFirebaseSlot.toLowerCase(),
        'firebase_project_id': Env.activeFirebaseProjectId.isEmpty
            ? 'not-configured'
            : Env.activeFirebaseProjectId,
        'demo_data': Env.demoData ? 'enabled' : 'disabled',
      };

  static String _hostOrState(String rawUrl) {
    final value = rawUrl.trim();
    if (value.isEmpty) return 'not-configured';
    return Uri.tryParse(value)?.host.isNotEmpty == true
        ? Uri.parse(value).host
        : 'invalid';
  }
}
