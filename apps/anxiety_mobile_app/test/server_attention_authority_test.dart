import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  test(
    'patient app consumes server events and embeds no backend service token',
    () {
      final feedback = File(
        'lib/services/anxiety_feedback_service.dart',
      ).readAsStringSync();
      final api = File('lib/services/api_service.dart').readAsStringSync();

      expect(feedback, contains('ingestServerAttentionEvent'));
      expect(feedback, isNot(contains("eventId: 'anx:\${")));
      expect(api, isNot(contains('BACKEND_TOKEN')));
      expect(api, contains('/v1/patients/me/attention-events?status=OPEN'));
    },
  );
}
