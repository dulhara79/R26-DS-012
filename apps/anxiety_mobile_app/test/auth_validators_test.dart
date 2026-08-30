import 'package:anxiety_mobile_app/services/auth_validators.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('AuthValidators', () {
    test('accepts valid signup values', () {
      expect(AuthValidators.displayName('Senuvi Layathma'), isNull);
      expect(AuthValidators.email('senuvi@example.com'), isNull);
      expect(AuthValidators.age('22'), isNull);
      expect(AuthValidators.password('AuraDemo9'), isNull);
      expect(
        AuthValidators.confirmPassword('AuraDemo9', 'AuraDemo9'),
        isNull,
      );
    });

    test('rejects invalid email and age', () {
      expect(AuthValidators.email('not-an-email'), isNotNull);
      expect(AuthValidators.age('17'), isNotNull);
      expect(AuthValidators.age('31'), isNotNull);
    });

    test('rejects weak or mismatched passwords', () {
      expect(AuthValidators.password('short'), isNotNull);
      expect(AuthValidators.password('alllowercase9'), isNotNull);
      expect(AuthValidators.password('ALLUPPERCASE9'), isNotNull);
      expect(AuthValidators.password('NoNumberHere'), isNotNull);
      expect(
        AuthValidators.confirmPassword('AuraDemo8', 'AuraDemo9'),
        isNotNull,
      );
    });
  });
}
