import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  test('server dashboard is the single primary worklist destination', () {
    final shell = File('lib/features/shell.dart').readAsStringSync();

    expect(shell, contains('ServerDashboardScreen('));
    expect(shell, isNot(contains('KpiDashboardScreen()')));
    expect(shell, isNot(contains("label: 'Caseload'")));
    expect(shell, contains("label: 'Dashboard'"));

    final indexedStack = RegExp(
      r'body:\s*IndexedStack\([\s\S]*?children:\s*\[([\s\S]*?)\]\s*,\s*\)',
    ).firstMatch(shell)?.group(1);

    expect(indexedStack, isNotNull);
    expect(indexedStack!.trimLeft(), startsWith('ServerDashboardScreen('));
    expect(indexedStack, contains('AskCareScreen()'));
  });

  test('P4 shell uses server Activity instead of legacy local Alerts', () {
    final shell = File('lib/features/shell.dart').readAsStringSync();

    expect(shell, contains('ActivityScreen.production()'));
    expect(shell, contains("label: 'Activity'"));
    expect(shell, isNot(contains('AlertsScreen()')));
    expect(shell, isNot(contains("label: 'Alerts'")));
    expect(shell, isNot(contains('roster.unacknowledgedCount')));
  });
}
