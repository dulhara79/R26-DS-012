import 'package:flutter/material.dart';

import '../theme/theme_controller.dart';

class AppearanceSettingsPage extends StatelessWidget {
  const AppearanceSettingsPage({super.key});

  Future<void> _pickTime(
    BuildContext context,
    ThemeController controller, {
    required bool isStart,
  }) async {
    final selected = await showTimePicker(
      context: context,
      initialTime: isStart ? controller.darkStart : controller.darkEnd,
    );
    if (selected == null) return;
    await controller.setSchedule(
      start: isStart ? selected : controller.darkStart,
      end: isStart ? controller.darkEnd : selected,
    );
  }

  @override
  Widget build(BuildContext context) {
    final controller = ThemeController.instance;
    return AnimatedBuilder(
      animation: controller,
      builder: (context, _) {
        return Scaffold(
          appBar: AppBar(title: const Text('Appearance')),
          body: ListView(
            padding: const EdgeInsets.all(20),
            children: [
              Text(
                'Theme',
                style: Theme.of(
                  context,
                ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 8),
              Text(
                'Choose a fixed theme, follow your phone, or switch automatically at set times.',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const SizedBox(height: 16),
              Card(
                clipBehavior: Clip.antiAlias,
                child: RadioGroup<AppThemeMode>(
                  groupValue: controller.mode,
                  onChanged: (value) {
                    if (value != null) controller.setMode(value);
                  },
                  child: Column(
                    children: AppThemeMode.values
                        .map(
                          (mode) => RadioListTile<AppThemeMode>(
                            value: mode,
                            title: Text(mode.label),
                            subtitle: Text(_description(mode)),
                          ),
                        )
                        .toList(),
                  ),
                ),
              ),
              if (controller.mode == AppThemeMode.scheduled) ...[
                const SizedBox(height: 18),
                Text(
                  'Dark theme schedule',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 8),
                Card(
                  child: Column(
                    children: [
                      ListTile(
                        leading: const Icon(Icons.dark_mode_outlined),
                        title: const Text('Turn dark theme on'),
                        trailing: Text(controller.darkStart.format(context)),
                        onTap: () =>
                            _pickTime(context, controller, isStart: true),
                      ),
                      const Divider(height: 1),
                      ListTile(
                        leading: const Icon(Icons.light_mode_outlined),
                        title: const Text('Turn dark theme off'),
                        trailing: Text(controller.darkEnd.format(context)),
                        onTap: () =>
                            _pickTime(context, controller, isStart: false),
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
        );
      },
    );
  }

  String _description(AppThemeMode mode) {
    switch (mode) {
      case AppThemeMode.system:
        return 'Match the light or dark setting on your phone.';
      case AppThemeMode.light:
        return 'Always use the light theme.';
      case AppThemeMode.dark:
        return 'Always use the dark theme.';
      case AppThemeMode.scheduled:
        return 'Use dark theme between the times you choose.';
    }
  }
}
