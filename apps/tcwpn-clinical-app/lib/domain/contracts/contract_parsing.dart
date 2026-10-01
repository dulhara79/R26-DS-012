String? contractString(Object? value) {
  if (value == null) return null;
  final text = value.toString().trim();
  return text.isEmpty ? null : text;
}

int? contractInt(Object? value) {
  if (value is int) return value;
  if (value is num) return value.toInt();
  return int.tryParse(value?.toString() ?? '');
}

double? contractDouble(Object? value) {
  if (value is num) return value.toDouble();
  return double.tryParse(value?.toString() ?? '');
}

bool? contractBool(Object? value) => value is bool ? value : null;

DateTime? contractDateTime(Object? value) {
  final raw = contractString(value);
  if (raw == null) return null;
  final hasOffset =
      raw.endsWith('Z') || RegExp(r'[+-]\d{2}:?\d{2}$').hasMatch(raw);
  return DateTime.tryParse(hasOffset ? raw : '${raw}Z')?.toLocal();
}

Map<String, dynamic>? contractMap(Object? value) {
  if (value is! Map) return null;
  return Map<String, dynamic>.from(value);
}

List<Map<String, dynamic>> contractMapList(Object? value) {
  if (value is! List) return const [];
  return value
      .whereType<Map>()
      .map((item) => Map<String, dynamic>.from(item))
      .toList(growable: false);
}
