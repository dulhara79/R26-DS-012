import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  test('explain screen keeps real fusion first and has labelled low-risk demo fallback', () async {
    final source = await File('lib/features/explain/explain_screen.dart').readAsString();

    expect(source, contains('demoLowRiskScore = 0.2033'));
    expect(source, contains('Demo low-risk example'));
    expect(source, contains('fusion.hasComposite'));
    expect(source, contains('Real backend fusion result'));
  });
}
