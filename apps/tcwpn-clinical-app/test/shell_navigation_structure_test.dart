import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  test('every visible shell destination has a registered screen', () {
    final shell = File('lib/features/shell.dart').readAsStringSync();
    final indexedStack = RegExp(
      r'body:\s*IndexedStack\([\s\S]*?children:\s*const\s*\[([\s\S]*?)\]\s*,\s*\)',
    ).firstMatch(shell)?.group(1);

    expect(indexedStack, isNotNull);
    expect(indexedStack, contains('AskCareScreen()'));
    expect(indexedStack, contains('KpiDashboardScreen()'));
  });
}
