import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  test('Settings states that note text goes through Central Backend', () {
    final source =
        File('lib/features/settings/settings_screen.dart').readAsStringSync();

    expect(source, isNot(contains('Sent to the model service for analysis.')));
    expect(source, contains('Central Backend'));
    expect(source, contains('Clinical NLP'));
  });

  test('Settings does not claim every ClinAnx host is certificate-pinned', () {
    final source =
        File('lib/features/settings/settings_screen.dart').readAsStringSync();

    expect(
      source,
      isNot(contains('Traffic is restricted to pinned certificate authorities')),
    );
    expect(source.toLowerCase(), contains('platform tls'));
  });
}
