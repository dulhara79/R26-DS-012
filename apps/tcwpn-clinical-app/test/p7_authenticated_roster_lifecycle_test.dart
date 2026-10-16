import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  test('root app does not initialize clinical roster before authentication', () {
    final mainSource = File('lib/main.dart').readAsStringSync();

    expect(
      mainSource,
      isNot(contains('create: (_) => RosterController()..init()')),
      reason:
          'Clinical SharedPreferences must not be loaded before a clinician session is active.',
    );
  });

  test('authenticated shell owns the roster provider lifecycle', () {
    final shellSource = File('lib/features/shell.dart').readAsStringSync();

    expect(shellSource, contains('ChangeNotifierProvider'));
    expect(shellSource, contains('RosterController()..init()'));
  });
}
