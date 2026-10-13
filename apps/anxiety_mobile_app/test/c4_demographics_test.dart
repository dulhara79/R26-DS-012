import 'package:anxiety_mobile_app/services/c4_demographics.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final validProfile = <String, dynamic>{
    'age': '23',
    'gender': 'Male',
    'education_level': 'Undergraduate',
    'c4_education': "bachelor's degree",
    'c4_smoke': 'never smokes',
    'c4_drink': 'never drinks',
  };

  test('C4 sends exactly five supported demographic features', () {
    final result = C4Demographics.fromSavedProfile(validProfile);

    expect(result, isNotNull);
    expect(result!.toJson(), <String, dynamic>{
      'age': 23,
      'gender': 'male',
      'edu': "bachelor's degree",
      'smoke': 'never smokes',
      'drink': 'never drinks',
    });
  });

  test(
    'C4 does not guess a completed degree from ongoing undergraduate study',
    () {
      final profile = Map<String, dynamic>.from(validProfile)
        ..remove('c4_education');
      expect(C4Demographics.fromSavedProfile(profile), isNull);
    },
  );

  test('C4 never invents missing smoking or drinking responses', () {
    final noSmoke = Map<String, dynamic>.from(validProfile)..remove('c4_smoke');
    final noDrink = Map<String, dynamic>.from(validProfile)..remove('c4_drink');

    expect(C4Demographics.fromSavedProfile(noSmoke), isNull);
    expect(C4Demographics.fromSavedProfile(noDrink), isNull);
  });

  test('C4 rejects categories outside the frozen model vocabulary', () {
    final noDegree = Map<String, dynamic>.from(validProfile)
      ..['c4_education'] = 'tertiary';
    final noSmokingCategory = Map<String, dynamic>.from(validProfile)
      ..['c4_smoke'] = 'none';
    final noDrinkingCategory = Map<String, dynamic>.from(validProfile)
      ..['c4_drink'] = 'never';

    expect(C4Demographics.fromSavedProfile(noDegree), isNull);
    expect(C4Demographics.fromSavedProfile(noSmokingCategory), isNull);
    expect(C4Demographics.fromSavedProfile(noDrinkingCategory), isNull);
  });

  test('C4 does not score ages or gender categories outside model scope', () {
    final older = Map<String, dynamic>.from(validProfile)..['age'] = '28';
    final unsupportedGender = Map<String, dynamic>.from(validProfile)
      ..['gender'] = 'Prefer not to say';

    expect(C4Demographics.fromSavedProfile(older), isNull);
    expect(C4Demographics.fromSavedProfile(unsupportedGender), isNull);
  });
}
