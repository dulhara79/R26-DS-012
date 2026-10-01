import 'dart:convert';
import 'dart:io';

Map<String, dynamic> loadContractFixture(String name) {
  final raw = File('test/fixtures/contracts/$name').readAsStringSync();
  return jsonDecode(raw) as Map<String, dynamic>;
}
