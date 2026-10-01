import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/domain/contracts/dashboard_snapshot.dart';
import 'package:r26_ds012_app/domain/repositories/assignment_invite_repository.dart';
import 'package:r26_ds012_app/domain/repositories/dashboard_repository.dart';
import 'package:r26_ds012_app/features/patients/patients_screen.dart';
import 'package:r26_ds012_app/state/dashboard_controller.dart';

class _Roster implements DashboardRepository {
  int calls = 0;
  @override
  Future<DashboardSnapshot> loadDashboard() async {
    calls++;
    return DashboardSnapshot(
      openEvents: const [],
      assignedPatients: const [],
      fetchedAt: DateTime.utc(2026, 9, 30),
    );
  }
}

class _Invites implements AssignmentInviteRepository {
  String? submitted;
  ApiException? failure;
  @override
  Future<AssignedSubject> redeem(String code) async {
    submitted = code;
    if (failure != null) throw failure!;
    return const AssignedSubject('subject-linked');
  }
}

void main() {
  setUp(() => SharedPreferences.setMockInitialValues({}));

  testWidgets(
    'patient invite redemption refreshes roster without local insertion',
    (tester) async {
      final roster = _Roster();
      final invites = _Invites();
      final controller = DashboardController(repository: roster);
      await controller.load();
      await tester.pumpWidget(
        MaterialApp(
          home: PatientsScreen(
            controller: controller,
            inviteRepository: invites,
          ),
        ),
      );
      await tester.pumpAndSettle();

      await tester.tap(find.byTooltip('Link patient with invite'));
      await tester.pumpAndSettle();
      expect(
        find.textContaining('patient ID alone cannot grant access'),
        findsOneWidget,
      );
      await tester.enterText(find.byType(TextFormField), 'aB_123-Z');
      await tester.tap(find.text('Accept invitation'));
      await tester.pumpAndSettle();

      expect(invites.submitted, 'aB_123-Z');
      expect(roster.calls, 2);
      expect(
        find.textContaining('roster could not be refreshed yet'),
        findsOneWidget,
      );
      expect(find.text('subject-linked'), findsNothing);
    },
  );

  testWidgets('used invite shows 409 error and keeps clinician in the form', (
    tester,
  ) async {
    final roster = _Roster();
    final invites = _Invites()
      ..failure = const ApiException(
        kind: ApiFailure.conflict,
        statusCode: 409,
      );
    final controller = DashboardController(repository: roster);
    await controller.load();
    await tester.pumpWidget(
      MaterialApp(
        home: PatientsScreen(controller: controller, inviteRepository: invites),
      ),
    );
    await tester.pumpAndSettle();

    await tester.tap(find.byTooltip('Link patient with invite'));
    await tester.pumpAndSettle();
    await tester.enterText(find.byType(TextFormField), 'used-code');
    await tester.tap(find.text('Accept invitation'));
    await tester.pumpAndSettle();

    expect(find.textContaining('already been used'), findsOneWidget);
    expect(roster.calls, 1);
    expect(find.text('Link patient'), findsOneWidget);
  });
}
