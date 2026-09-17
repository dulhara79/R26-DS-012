import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  test('AuthService consumes centralized Env auth configuration', () {
    final source = File('lib/data/api/auth_service.dart').readAsStringSync();

    expect(source, isNot(contains("String.fromEnvironment('AUTH_BASE'")));
    expect(source, isNot(contains("String.fromEnvironment('AUTH_SALT'")));
    expect(source, isNot(contains("String.fromEnvironment('AUTH_LOCAL'")));
    expect(source, contains('Env.authBase'));
    expect(source, contains('Env.authSalt'));
    expect(source, contains('Env.authLocalAccounts'));
  });

  test('demo clinical fixtures require explicit opt-in', () {
    final source = File('lib/core/config/env.dart').readAsStringSync();

    expect(
      source,
      contains("'DEMO_DATA',\n    defaultValue: false"),
      reason: 'A release build must not seed demo patients when a define is omitted.',
    );
  });
}
