// tool/hash_password.dart
//
// Generates the salted digest for a local-mode account.
//
//   dart run tool/hash_password.dart <password> [salt]
//
// Example:
//   dart run tool/hash_password.dart "Kx9#mQ2vLp" my-project-salt
//
// Paste the output into your --dart-define=AUTH_LOCAL string. The plaintext
// password is never stored anywhere in the repository or the build.

import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';

void main(List<String> args) {
  if (args.isEmpty) {
    stderr.writeln('usage: dart run tool/hash_password.dart <password> [salt]');
    exit(64);
  }

  final password = args[0];
  final salt = args.length > 1 ? args[1] : 'r26-ds012-local-salt';
  final digest = sha256.convert(utf8.encode('$salt$password')).toString();

  stdout.writeln('salt   : $salt');
  stdout.writeln('digest : $digest');
  stdout.writeln();
  stdout.writeln('Entry format — id|Display Name|digest');
  stdout.writeln('DR001|Dr Your Name|$digest');
  stdout.writeln();
  stdout.writeln('Then build with:');
  stdout.writeln('  --dart-define=AUTH_SALT=$salt \\');
  stdout.writeln('  --dart-define=AUTH_LOCAL="DR001|Dr Your Name|$digest"');
}
