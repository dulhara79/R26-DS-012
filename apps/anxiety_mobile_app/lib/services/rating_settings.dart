import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../theme/app_theme.dart';
import '../services/background/daily_reminder.dart';
import '../services/notification_helper.dart';

class RatingSettingsPage extends StatefulWidget {
  const RatingSettingsPage({super.key});

  @override
  State<RatingSettingsPage> createState() => _RatingSettingsPageState();
}

class _RatingSettingsPageState extends State<RatingSettingsPage> {
  bool _enabled = true;
  bool _weeklyEnabled = true;

  TimeOfDay _morningTime = const TimeOfDay(hour: 9, minute: 0);
  TimeOfDay _afternoonTime = const TimeOfDay(hour: 14, minute: 0);
  TimeOfDay _eveningTime = const TimeOfDay(hour: 20, minute: 0);
  int _weeklyWeekday = DateTime.sunday;
  TimeOfDay _gad7Time = const TimeOfDay(hour: 20, minute: 0);
  TimeOfDay _pss10Time = const TimeOfDay(hour: 21, minute: 0);

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final prefs = await SharedPreferences.getInstance();
    if (!mounted) return;
    setState(() {
      _enabled = prefs.getBool('rating_enabled') ?? true;
      _weeklyEnabled = prefs.getBool('weekly_checkins_enabled') ?? true;
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
      _weeklyWeekday =
          prefs.getInt('weekly_checkin_weekday') ?? DateTime.sunday;
      _gad7Time = TimeOfDay(
        hour: prefs.getInt('gad7_hour') ?? 20,
        minute: prefs.getInt('gad7_minute') ?? 0,
      );
      _pss10Time = TimeOfDay(
        hour: prefs.getInt('pss10_hour') ?? 21,
        minute: prefs.getInt('pss10_minute') ?? 0,
      );
    });
  }

  Future<void> _save({bool resetDailyThrottles = false}) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('rating_enabled', _enabled);
    await prefs.setBool('weekly_checkins_enabled', _weeklyEnabled);
    await prefs.setInt('ema_morning_hour', _morningTime.hour);
    await prefs.setInt('ema_morning_minute', _morningTime.minute);
    await prefs.setInt('ema_afternoon_hour', _afternoonTime.hour);
    await prefs.setInt('ema_afternoon_minute', _afternoonTime.minute);
    await prefs.setInt('ema_evening_hour', _eveningTime.hour);
    await prefs.setInt('ema_evening_minute', _eveningTime.minute);
    await prefs.setInt('weekly_checkin_weekday', _weeklyWeekday);
    await prefs.setInt('gad7_hour', _gad7Time.hour);
    await prefs.setInt('gad7_minute', _gad7Time.minute);
    await prefs.setInt('pss10_hour', _pss10Time.hour);
    await prefs.setInt('pss10_minute', _pss10Time.minute);

    if (resetDailyThrottles) {
      await DailyReminder.clearThrottleTimestamps();
    }
    if (!_enabled) await NotificationHelper.cancelAllDailyCheckIns();
    if (!_weeklyEnabled) await NotificationHelper.cancelWeeklyCheckIns();
  }

  Future<void> _useDailyDefaults() async {
    setState(() {
      _enabled = true;
      _morningTime = const TimeOfDay(hour: 9, minute: 0);
      _afternoonTime = const TimeOfDay(hour: 14, minute: 0);
      _eveningTime = const TimeOfDay(hour: 20, minute: 0);
    });
    await _save(resetDailyThrottles: true);
  }

  Future<void> _useWeeklyDefaults() async {
    setState(() {
      _weeklyEnabled = true;
      _weeklyWeekday = DateTime.sunday;
      _gad7Time = const TimeOfDay(hour: 20, minute: 0);
      _pss10Time = const TimeOfDay(hour: 21, minute: 0);
    });
    await _save();
  }

  Future<void> _pickTime(
    TimeOfDay current,
    ValueChanged<TimeOfDay> onPicked, {
    required bool daily,
  }) async {
    final picked = await showTimePicker(context: context, initialTime: current);
    if (picked != null) {
      onPicked(picked);
      await _save(resetDailyThrottles: daily);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Check-in Settings')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Card(
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            child: SwitchListTile(
              title: const Text(
                'Daily check-in reminders',
                style: TextStyle(fontWeight: FontWeight.w600),
              ),
              subtitle: const Text(
                'Turn off to stop morning, afternoon, and evening reminders',
              ),
              value: _enabled,
              activeTrackColor: AppTheme.kPrimaryDeep,
              onChanged: (v) async {
                setState(() => _enabled = v);
                await _save(resetDailyThrottles: true);
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
                  const Padding(
                    padding: EdgeInsets.fromLTRB(16, 14, 16, 6),
                    child: Text(
                      'Use the default times or tap any time to choose your own.',
                      style: TextStyle(fontSize: 12),
                    ),
                  ),
                  _timeTile(
                    icon: '☀️',
                    label: 'Morning check-in',
                    time: _morningTime,
                    onTap: _enabled
                        ? () => _pickTime(
                            _morningTime,
                            (t) => setState(() => _morningTime = t),
                            daily: true,
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
                            _afternoonTime,
                            (t) => setState(() => _afternoonTime = t),
                            daily: true,
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
                            _eveningTime,
                            (t) => setState(() => _eveningTime = t),
                            daily: true,
                          )
                        : null,
                  ),
                  TextButton.icon(
                    onPressed: _enabled ? _useDailyDefaults : null,
                    icon: const Icon(Icons.restore_rounded),
                    label: const Text('Use daily defaults'),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),

          Card(
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            child: SwitchListTile(
              title: const Text(
                'Weekly check reminders',
                style: TextStyle(fontWeight: FontWeight.w600),
              ),
              subtitle: const Text(
                'One anxiety check and one stress check each week',
              ),
              value: _weeklyEnabled,
              activeTrackColor: AppTheme.kPrimaryDeep,
              onChanged: (value) async {
                setState(() => _weeklyEnabled = value);
                await _save();
              },
            ),
          ),

          const SizedBox(height: 16),

          Opacity(
            opacity: _weeklyEnabled ? 1.0 : 0.4,
            child: Card(
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                children: [
                  const Padding(
                    padding: EdgeInsets.fromLTRB(16, 14, 16, 6),
                    child: Text(
                      'Default: Sunday at 8:00 PM and 9:00 PM. Change the day or either time if you prefer.',
                      style: TextStyle(fontSize: 12),
                    ),
                  ),
                  ListTile(
                    leading: const Text('📅', style: TextStyle(fontSize: 22)),
                    title: const Text(
                      'Weekly day',
                      style: TextStyle(fontWeight: FontWeight.w500),
                    ),
                    trailing: DropdownButton<int>(
                      value: _weeklyWeekday,
                      onChanged: _weeklyEnabled
                          ? (value) async {
                              if (value == null) return;
                              setState(() => _weeklyWeekday = value);
                              await _save();
                            }
                          : null,
                      items: const [
                        DropdownMenuItem(value: 1, child: Text('Monday')),
                        DropdownMenuItem(value: 2, child: Text('Tuesday')),
                        DropdownMenuItem(value: 3, child: Text('Wednesday')),
                        DropdownMenuItem(value: 4, child: Text('Thursday')),
                        DropdownMenuItem(value: 5, child: Text('Friday')),
                        DropdownMenuItem(value: 6, child: Text('Saturday')),
                        DropdownMenuItem(value: 7, child: Text('Sunday')),
                      ],
                    ),
                  ),
                  const Divider(height: 1, indent: 16, endIndent: 16),
                  _timeTile(
                    icon: '🧠',
                    label: 'Weekly anxiety check',
                    time: _gad7Time,
                    onTap: _weeklyEnabled
                        ? () => _pickTime(
                            _gad7Time,
                            (time) => setState(() => _gad7Time = time),
                            daily: false,
                          )
                        : null,
                  ),
                  const Divider(height: 1, indent: 16, endIndent: 16),
                  _timeTile(
                    icon: '📊',
                    label: 'Weekly stress check',
                    time: _pss10Time,
                    onTap: _weeklyEnabled
                        ? () => _pickTime(
                            _pss10Time,
                            (time) => setState(() => _pss10Time = time),
                            daily: false,
                          )
                        : null,
                  ),
                  TextButton.icon(
                    onPressed: _weeklyEnabled ? _useWeeklyDefaults : null,
                    icon: const Icon(Icons.restore_rounded),
                    label: const Text('Use weekly defaults'),
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
                    'Completing a check-in stops that reminder until its next scheduled day or week.',
                    style: TextStyle(fontSize: 13, color: Colors.teal.shade800),
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 24),

          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: () async {
                // Capture context-dependent objects before the async gap.
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
          const SizedBox(height: 24),
        ],
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
