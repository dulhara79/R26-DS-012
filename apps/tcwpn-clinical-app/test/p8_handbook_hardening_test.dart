import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/core/config/research_build_info.dart';
import 'package:r26_ds012_app/core/notifications/push_attention_message.dart';
import 'package:r26_ds012_app/core/security/clinical_endpoint_policy.dart';

void main() {
  group('Handbook Phase 8 security boundary', () {
    test('active mobile Dart contains no reusable privileged backend token', () {
      final dartFiles = Directory('lib')
          .listSync(recursive: true)
          .whereType<File>()
          .where((file) => file.path.endsWith('.dart'));

      for (final file in dartFiles) {
        final source = file.readAsStringSync();
        expect(
          source,
          isNot(contains('BACKEND_TOKEN')),
          reason: '${file.path} must not compile a reusable backend token into the app.',
        );
        expect(
          source,
          isNot(contains('Env.backendToken')),
          reason: '${file.path} must use the authenticated clinician Session bearer.',
        );
      }
    });

    test('clinical endpoints require HTTPS except loopback development', () {
      expect(ClinicalEndpointPolicy.isAllowed('https://backend.example.org'), isTrue);
      expect(ClinicalEndpointPolicy.isAllowed('https://example.org:8443'), isTrue);
      expect(ClinicalEndpointPolicy.isAllowed('http://localhost:8000'), isTrue);
      expect(ClinicalEndpointPolicy.isAllowed('http://127.0.0.1:8000'), isTrue);
      expect(ClinicalEndpointPolicy.isAllowed('http://[::1]:8000'), isTrue);

      expect(ClinicalEndpointPolicy.isAllowed('http://backend.example.org'), isFalse);
      expect(ClinicalEndpointPolicy.isAllowed('http://10.0.2.2:8000'), isFalse);
      expect(ClinicalEndpointPolicy.isAllowed('ftp://backend.example.org'), isFalse);
      expect(ClinicalEndpointPolicy.isAllowed('backend.example.org'), isFalse);
    });

    test('research build summary exposes routing identity but never secrets', () {
      final values = ResearchBuildInfo.values;

      expect(values.keys, containsAll(<String>[
        'app_version',
        'build_environment',
        'backend_host',
        'auth_mode',
        'firebase_slot',
        'firebase_project_id',
        'demo_data',
      ]));

      final rendered = values.toString().toLowerCase();
      expect(rendered, isNot(contains('token')));
      expect(rendered, isNot(contains('password')));
      expect(rendered, isNot(contains('api_key')));
      expect(rendered, isNot(contains('salt')));
    });
  });

  group('Handbook Phase 8 push privacy', () {
    test('accepts event-identity-only attention push', () {
      final message = PushAttentionMessage.tryParse(const {
        'type': 'attention_event',
        'event_id': 'evt-001',
      });
      expect(message?.eventId, 'evt-001');
    });

    test('rejects malformed or wrong-type push', () {
      expect(PushAttentionMessage.tryParse(const {}), isNull);
      expect(
        PushAttentionMessage.tryParse(const {
          'type': 'other',
          'event_id': 'evt-001',
        }),
        isNull,
      );
      expect(
        PushAttentionMessage.tryParse(const {
          'type': 'attention_event',
          'event_id': '   ',
        }),
        isNull,
      );
    });

    test('rejects push payload containing prohibited clinical/identity detail', () {
      const prohibited = <String, Object>{
        'patient_name': 'Synthetic Patient',
        'mrn': 'MRN-001',
        'note_text': 'clinical note',
        'current_score': 0.7,
        'forecast_score': 0.8,
        'fusion_score': 0.75,
      };

      for (final entry in prohibited.entries) {
        expect(
          PushAttentionMessage.tryParse({
            'type': 'attention_event',
            'event_id': 'evt-001',
            entry.key: entry.value,
          }),
          isNull,
          reason: 'Push must reject prohibited field ${entry.key}.',
        );
      }
    });
  });

  test('polling fallback is not structurally inside the permission-only branch', () {
    final source = File('lib/features/shell.dart').readAsStringSync();
    expect(
      source,
      contains('_startNotificationPolling();'),
    );
    expect(
      source,
      isNot(contains('if (enabled) {\n        try {\n          await _pushService.activateAuthenticatedSession();\n        } catch (_) {\n          // Push is an acceleration path. Polling remains the recovery path.\n        }\n        _startNotificationPolling();\n      }')),
      reason: 'Polling fallback must not disappear when OS notification permission is denied.',
    );
  });
}
