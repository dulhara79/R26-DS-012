class AuthValidators {
  static final RegExp _emailPattern = RegExp(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$",
  );

  static String? displayName(String? value) {
    final name = value?.trim() ?? '';
    if (name.isEmpty) return 'Display name is required.';
    if (name.length < 2) return 'Enter at least 2 characters.';
    if (name.length > 80) return 'Use 80 characters or fewer.';
    if (!RegExp(r"^[A-Za-z][A-Za-z .'-]*$").hasMatch(name)) {
      return 'Use letters, spaces, apostrophes, hyphens or periods only.';
    }
    return null;
  }

  static String? email(String? value) {
    final email = value?.trim() ?? '';
    if (email.isEmpty) return 'Email is required.';
    if (email.length > 254 || !_emailPattern.hasMatch(email)) {
      return 'Enter a valid email address.';
    }
    return null;
  }

  static String? age(String? value) {
    final raw = value?.trim() ?? '';
    if (raw.isEmpty) return 'Age is required.';
    final age = int.tryParse(raw);
    if (age == null) return 'Enter a valid age.';
    if (age < 18 || age > 30) {
      return 'This study is for participants aged 18 to 30.';
    }
    return null;
  }

  static String? password(String? value) {
    final password = value ?? '';
    if (password.isEmpty) return 'Password is required.';
    if (password.length < 8) return 'Use at least 8 characters.';
    if (!RegExp(r'[A-Z]').hasMatch(password)) {
      return 'Add at least one uppercase letter.';
    }
    if (!RegExp(r'[a-z]').hasMatch(password)) {
      return 'Add at least one lowercase letter.';
    }
    if (!RegExp(r'\d').hasMatch(password)) {
      return 'Add at least one number.';
    }
    return null;
  }

  static String? confirmPassword(String? value, String password) {
    if ((value ?? '').isEmpty) return 'Confirm your password.';
    if (value != password) return 'Passwords do not match.';
    return null;
  }
}
