import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:r26_ds012_app/domain/repositories/assignment_invite_repository.dart';
import 'package:r26_ds012_app/features/patients/link_patient_screen.dart';

class _Invites implements AssignmentInviteRepository {
  String? submitted;
  int calls = 0;

  @override
  Future<AssignedSubject> redeem(String inviteCode) async {
    calls++;
    submitted = inviteCode;
    return const AssignedSubject('subject-a');
  }
}

void main() {
  const token = 'A1234567890123456789012345678901234567890_Z';

  testWidgets('scanned invitation uses existing authenticated redemption', (tester) async {
    final invites = _Invites();
    String? linked;
    await tester.pumpWidget(MaterialApp(
      home: Builder(builder: (context) => Scaffold(
        body: TextButton(
          onPressed: () async {
            linked = await Navigator.of(context).push<String>(
              MaterialPageRoute(builder: (_) => LinkPatientScreen(
                repository: invites,
                scanInvitation: (_) async => token,
              )),
            );
          },
          child: const Text('Open'),
        ),
      )),
    ));
    await tester.tap(find.text('Open'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Scan invitation QR'));
    await tester.pumpAndSettle();
    expect(find.widgetWithText(TextFormField, 'Patient invitation code'), findsOneWidget);
    await tester.tap(find.text('Accept invitation'));
    await tester.pumpAndSettle();
    expect(invites.calls, 1);
    expect(invites.submitted, token);
    expect(linked, 'subject-a');
  });

  testWidgets('scanning an Aura Participant ID cannot call assignment API', (tester) async {
    final invites = _Invites();
    await tester.pumpWidget(MaterialApp(
      home: LinkPatientScreen(
        repository: invites,
        scanInvitation: (_) async => 'P_0123456789ABCDEF',
      ),
    ));
    await tester.tap(find.text('Scan invitation QR'));
    await tester.pumpAndSettle();
    expect(find.textContaining('Participant ID, not a clinician invitation'), findsOneWidget);
    expect(invites.calls, 0);
  });

  testWidgets('manual Participant ID is rejected before redemption', (tester) async {
    final invites = _Invites();
    await tester.pumpWidget(MaterialApp(
      home: LinkPatientScreen(repository: invites),
    ));
    await tester.enterText(find.byType(TextFormField), 'P_0123456789ABCDEF');
    await tester.tap(find.text('Accept invitation'));
    await tester.pumpAndSettle();
    expect(find.textContaining('Participant ID cannot authorize linking'), findsOneWidget);
    expect(invites.calls, 0);
  });

  testWidgets('cancelled QR scan keeps manual entry usable', (tester) async {
    final invites = _Invites();
    await tester.pumpWidget(MaterialApp(
      home: LinkPatientScreen(
        repository: invites,
        scanInvitation: (_) async => null,
      ),
    ));
    await tester.tap(find.text('Scan invitation QR'));
    await tester.pumpAndSettle();
    await tester.enterText(find.byType(TextFormField), 'legacy-manual-code');
    await tester.tap(find.text('Accept invitation'));
    await tester.pumpAndSettle();
    expect(invites.submitted, 'legacy-manual-code');
  });
}
