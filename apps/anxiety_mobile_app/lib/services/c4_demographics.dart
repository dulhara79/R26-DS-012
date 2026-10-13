/// Frozen DCAR demographic input vocabulary. Never infer or default missing
/// smoking, drinking, degree, age, or gender information.
class C4Demographics {
  const C4Demographics({
    required this.age,
    required this.gender,
    required this.education,
    required this.smoking,
    required this.drinking,
  });

  final int age;
  final String gender;
  final String education;
  final String smoking;
  final String drinking;

  static const educationOptions = <String>[
    "bachelor's degree",
    'associate degree',
    "master's degree",
    'doctorate degree',
  ];

  static const smokingOptions = <String>[
    'never smokes',
    'occasional smoker (cumulative smoking <10 packs)',
    'current smoker (cumulative smoking >10 packs)',
    'former smoker (cumulative smoking >10 packs), but not in the past year',
  ];

  static const drinkingOptions = <String>[
    'never drinks',
    'drinks occasionally (less than once a week)',
    'current regular drinker (more than once a week)',
    'drank in the past (more than once a week), but not in the past year',
  ];

  static C4Demographics? fromSavedProfile(Map<String, dynamic> profile) {
    final rawAge = profile['age']?.toString();
    final age = int.tryParse(rawAge ?? '');
    final gender = profile['gender']?.toString().trim().toLowerCase();
    final education = profile['c4_education'];
    final smoking = profile['c4_smoke'];
    final drinking = profile['c4_drink'];
    if (age == null ||
        age < 18 ||
        age > 27 ||
        (gender != 'female' && gender != 'male') ||
        !educationOptions.contains(education) ||
        !smokingOptions.contains(smoking) ||
        !drinkingOptions.contains(drinking)) {
      return null;
    }
    return C4Demographics(
      age: age,
      gender: gender!,
      education: education as String,
      smoking: smoking as String,
      drinking: drinking as String,
    );
  }

  Map<String, dynamic> toJson() => <String, dynamic>{
    'age': age,
    'gender': gender,
    'edu': education,
    'smoke': smoking,
    'drink': drinking,
  };
}
