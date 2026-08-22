import 'package:anxiety_mobile_app/theme/theme_controller.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('overnight dark schedule crosses midnight', () {
    expect(
      ThemeController.isDarkAtMinutes(
        current: 23 * 60,
        start: 20 * 60,
        end: 7 * 60,
      ),
      isTrue,
    );
    expect(
      ThemeController.isDarkAtMinutes(
        current: 6 * 60 + 59,
        start: 20 * 60,
        end: 7 * 60,
      ),
      isTrue,
    );
    expect(
      ThemeController.isDarkAtMinutes(
        current: 12 * 60,
        start: 20 * 60,
        end: 7 * 60,
      ),
      isFalse,
    );
  });

  test('same-day schedule includes start and excludes end', () {
    expect(
      ThemeController.isDarkAtMinutes(
        current: 18 * 60,
        start: 18 * 60,
        end: 22 * 60,
      ),
      isTrue,
    );
    expect(
      ThemeController.isDarkAtMinutes(
        current: 22 * 60,
        start: 18 * 60,
        end: 22 * 60,
      ),
      isFalse,
    );
  });
}
