import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  test('ChartController resolves Aura ids without attach compatibility', () {
    final source = File('lib/state/controllers.dart').readAsStringSync();

    expect(source, contains('resolveAppUserId('));
    expect(source, isNot(contains('_backend.attach(')));
  });

  test('ChartController never mints authoritative escalation alerts locally', () {
    final source = File('lib/state/controllers.dart').readAsStringSync();

    expect(source, isNot(contains('_raiseIfEscalated(')));
    expect(source, isNot(contains('roster.raiseAlert(ClinicalAlert(')));
  });

  test('CentralBackendGateway exposes no deprecated attach shim', () {
    final source = File('lib/data/api/gateways.dart').readAsStringSync();

    expect(source, contains('resolveAppUserId'));
    expect(source, isNot(contains('Future<String?> attach(')));
  });
}
