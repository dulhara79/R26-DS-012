import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../theme/app_theme.dart';

class RatingSettingsPage extends StatefulWidget {
  const RatingSettingsPage({super.key});

  @override
  State<RatingSettingsPage> createState() => _RatingSettingsPageState();
}

class _RatingSettingsPageState extends State<RatingSettingsPage> {
  bool _enabled = true;

  TimeOfDay _morningTime = const TimeOfDay(hour: 9, minute: 0);
  TimeOfDay _afternoonTime = const TimeOfDay(hour: 14, minute: 0);
  TimeOfDay _eveningTime = const TimeOfDay(hour: 20, minute: 0);

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _enabled = prefs.getBool('rating_enabled') ?? true;
      _morningTime = TimeOfDay(
        hour: prefs.getInt('ema_morning_hour') ?? 9,
        minute: prefs.getInt('ema_morning_minute') ?? 0,
      );
      _afternoonTime = TimeOfDay(
        hour: prefs.getInt('ema_afternoon_hour') ?? 14,
        minute: prefs.getInt('ema_afternoon_minute') ?? 0,
      );
      _eveningTime = TimeOfDay(
        hour: prefs.getInt('ema_evening_hour') ?? 20,
        minute: prefs.getInt('ema_evening_minute') ?? 0,
      );
    });
  }

  Future<void> _save() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('rating_enabled', _enabled);
    await prefs.setInt('ema_morning_hour', _morningTime.hour);
    await prefs.setInt('ema_morning_minute', _morningTime.minute);
    await prefs.setInt('ema_afternoon_hour', _afternoonTime.hour);
    await prefs.setInt('ema_afternoon_minute', _afternoonTime.minute);
    await prefs.setInt('ema_evening_hour', _eveningTime.hour);
    await prefs.setInt('ema_evening_minute', _eveningTime.minute);
  }

  Future<void> _pickTime(
    String label,
    TimeOfDay current,
    ValueChanged<TimeOfDay> onPicked,
  ) async {
    final picked = await showTimePicker(context: context, initialTime: current);
    if (picked != null) {
      onPicked(picked);
      await _save();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Check-in Settings')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Card(
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              child: SwitchListTile(
                title: const Text(
                  'Enable daily check-ins',
                  style: TextStyle(fontWeight: FontWeight.w600),
                ),
                subtitle: const Text(
                  '3 check-ins per day (morning, afternoon, evening)',
                ),
                value: _enabled,
                activeTrackColor: AppTheme.kPrimaryDeep,
                onChanged: (v) async {
                  setState(() => _enabled = v);
                  await _save();
                },
              ),
            ),

            const SizedBox(height: 16),

            Opacity(
              opacity: _enabled ? 1.0 : 0.4,
              child: Card(
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(
                  children: [
                    _timeTile(
                      icon: '☀️',
                      label: 'Morning check-in',
                      time: _morningTime,
                      onTap: _enabled
                          ? () => _pickTime(
                              'Morning',
                              _morningTime,
                              (t) => setState(() => _morningTime = t),
                            )
                          : null,
                    ),
                    const Divider(height: 1, indent: 16, endIndent: 16),
                    _timeTile(
                      icon: '🌤️',
                      label: 'Afternoon check-in',
                      time: _afternoonTime,
                      onTap: _enabled
                          ? () => _pickTime(
                              'Afternoon',
                              _afternoonTime,
                              (t) => setState(() => _afternoonTime = t),
                            )
                          : null,
                    ),
                    const Divider(height: 1, indent: 16, endIndent: 16),
                    _timeTile(
                      icon: '🌙',
                      label: 'Evening check-in',
                      time: _eveningTime,
                      onTap: _enabled
                          ? () => _pickTime(
                              'Evening',
                              _eveningTime,
                              (t) => setState(() => _eveningTime = t),
                            )
                          : null,
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 20),

            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: AppTheme.kPrimaryDeep.withValues(alpha: 0.07),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(
                    Icons.info_outline,
                    color: AppTheme.kPrimaryDeep,
                    size: 18,
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      'You will also receive a weekly GAD-7 questionnaire every Monday morning. '
                      'All check-ins are required for the research.',
                      style: TextStyle(
                        fontSize: 13,
                        color: Colors.teal.shade800,
                      ),
                    ),
                  ),
                ],
              ),
            ),

            const Spacer(),

            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                // FIX: capture ScaffoldMessenger and Navigator BEFORE the await
                // to avoid using BuildContext across async gaps
                onPressed: () async {
                  final messenger = ScaffoldMessenger.of(context);
                  final navigator = Navigator.of(context);
                  await _save();
                  messenger.showSnackBar(
                    const SnackBar(content: Text('Settings saved')),
                  );
                  navigator.pop();
                },
                child: const Text('Save Settings'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _timeTile({
    required String icon,
    required String label,
    required TimeOfDay time,
    VoidCallback? onTap,
  }) {
    return ListTile(
      leading: Text(icon, style: const TextStyle(fontSize: 22)),
      title: Text(label, style: const TextStyle(fontWeight: FontWeight.w500)),
      trailing: Text(
        time.format(context),
        style: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.bold,
          color: onTap != null ? AppTheme.kPrimaryDeep : Colors.grey,
        ),
      ),
      onTap: onTap,
    );
  }
}
