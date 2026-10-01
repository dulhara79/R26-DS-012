import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  group('Handbook Phase 9 research-release documentation', () {
    test('release documentation set exists', () {
      const required = <String>[
        'docs/release/PHASE9_RESEARCH_RELEASE.md',
        'docs/release/SOP.md',
        'docs/release/DEPLOYMENT_RUNBOOK.md',
        'docs/release/TEST_EVIDENCE.md',
        'docs/release/KNOWN_LIMITATIONS.md',
      ];

      for (final path in required) {
        expect(File(path).existsSync(), isTrue, reason: path);
      }
    });

    test('README points to the Phase 9 release evidence and runbook', () {
      final readme = File('README.md').readAsStringSync();
      expect(readme, contains('PHASE9_RESEARCH_RELEASE.md'));
      expect(readme, contains('DEPLOYMENT_RUNBOOK.md'));
      expect(readme, contains('KNOWN_LIMITATIONS.md'));
    });

    test('active release docs do not reintroduce shared mobile backend token', () {
      const paths = <String>[
        'README.md',
        'ARCHITECTURE.md',
        'APK_BUILD.md',
        'docs/release/PHASE9_RESEARCH_RELEASE.md',
        'docs/release/SOP.md',
        'docs/release/DEPLOYMENT_RUNBOOK.md',
        'docs/release/TEST_EVIDENCE.md',
        'docs/release/KNOWN_LIMITATIONS.md',
      ];

      for (final path in paths) {
        final file = File(path);
        if (!file.existsSync()) continue;
        final source = file.readAsStringSync();
        expect(
          source,
          isNot(contains('BACKEND_TOKEN')),
          reason: '$path must not instruct users to embed a shared backend token',
        );
      }
    });

    test('research-use boundary is explicit in the public entry point', () {
      final readme = File('README.md').readAsStringSync().toLowerCase();
      expect(readme, contains('research prototype'));
      expect(readme, contains('not a diagnostic device'));
    });
  });
}
