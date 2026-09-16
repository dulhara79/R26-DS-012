import 'package:flutter_test/flutter_test.dart';
import 'package:http/testing.dart';
import 'package:r26_ds012_app/data/api/api_client.dart';
import 'package:r26_ds012_app/data/repositories/central_backend_repositories.dart';

Matcher _notConfigured() => throwsA(
      isA<ApiException>().having(
        (e) => e.kind,
        'kind',
        ApiFailure.notConfigured,
      ),
    );

void main() {
  ApiClient noNetworkApi() => ApiClient(
        'https://backend.test',
        client: MockClient((request) async {
          throw StateError(
            'Unverified target adapter attempted network call: ${request.url}',
          );
        }),
        bearer: () => 'clinician-jwt',
      );

  test('auth target adapter remains blocked until backend contract is verified',
      () async {
    await expectLater(
      CentralBackendAuthRepository(noNetworkApi()).validateCurrentSession(),
      _notConfigured(),
    );
  });

  test('dashboard target adapter remains blocked until backend contract is verified',
      () async {
    await expectLater(
      CentralBackendDashboardRepository(noNetworkApi()).loadDashboard(),
      _notConfigured(),
    );
  });

  test('latest assessment target adapter remains blocked until backend contract is verified',
      () async {
    await expectLater(
      CentralBackendAssessmentRepository(noNetworkApi())
          .latestAssessment('subject-001'),
      _notConfigured(),
    );
  });

  test('attention-event target adapter remains blocked without verified routes',
      () async {
    final repo = CentralBackendAttentionEventRepository(noNetworkApi());

    await expectLater(repo.openEvents(), _notConfigured());
    await expectLater(repo.activity(), _notConfigured());
    await expectLater(
      repo.activity(subjectId: 'subject-001'),
      _notConfigured(),
    );
    await expectLater(repo.eventById('evt-001'), _notConfigured());
    await expectLater(repo.acknowledge('evt-001'), _notConfigured());
    await expectLater(repo.resolve('evt-001'), _notConfigured());
  });
}
