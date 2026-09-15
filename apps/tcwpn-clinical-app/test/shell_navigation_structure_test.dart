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
}
