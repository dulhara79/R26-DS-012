import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:r26_ds012_app/domain/models.dart';
import 'package:r26_ds012_app/features/chart/patient_chart_screen.dart';

void main() {
  testWidgets('shows the generated pairing code and its expiry', (tester) async {
    final enrolment = EnrolmentResult(
      subjectId: 'subject-1',
      pairingCode: '419-330',
      expiresAt: DateTime.utc(2026, 8, 30, 18),
    );

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: PairingCodeNotice(
            enrolment: enrolment,
          ),
        ),
      ),
    );

    expect(find.textContaining('PAIRING CODE: 419-330'), findsOneWidget);
    expect(
      find.textContaining('Ask the patient to enter this code'),
      findsOneWidget,
    );
    expect(
      find.textContaining('Expires 30 Aug 2026, 18:00 UTC'),
      findsOneWidget,
    );
  });
}
