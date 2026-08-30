import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:image_picker/image_picker.dart';

import 'background_service_helper.dart';
import 'theme/app_theme.dart';
import 'pages/data_rights_page.dart';
import 'pages/share_participant_id_page.dart';
import 'pages/baseline_calibration_page.dart';
import 'pages/appearance_settings_page.dart';
import 'services/background/service_config.dart';
import 'services/background/daily_reminder.dart';
import 'services/participant_identity_service.dart';
import 'services/rating_settings.dart';
import 'theme/theme_controller.dart';

class ProfilePage extends StatefulWidget {
  final bool isTab;
  const ProfilePage({super.key, this.isTab = false});

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  bool _isEditing = false;
  String _displayName = '';
  String? _participantId;
  String? _profileImagePath;
  final ImagePicker _picker = ImagePicker();
  final TextEditingController _displayNameController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _isEditing =
        !widget.isTab; // Tab mode starts in view; standalone starts in edit
    _loadProfile();
  }

  Future<void> _loadProfile() async {
    final prefs = await SharedPreferences.getInstance();
    final displayName = prefs.getString('display_name') ?? '';
    final participantId = prefs.getString(
      ParticipantIdentityService.participantIdKey,
    );
    final savedProfileImagePath = prefs.getString('profile_image_path');
    final validProfileImagePath =
        savedProfileImagePath != null &&
            File(savedProfileImagePath).existsSync()
        ? savedProfileImagePath
        : null;
    final profileJson = prefs.getString('user_profile_data');
    if (!mounted) return;
    if (profileJson != null) {
      Map<String, dynamic> data;
      try {
        data = Map<String, dynamic>.from(jsonDecode(profileJson) as Map);
      } catch (error) {
        debugPrint('Could not read saved profile: $error');
        data = <String, dynamic>{};
      }
      setState(() {
        _displayName = displayName;
        _participantId = participantId;
        _displayNameController.text = displayName;
        _ageController.text = data['age'] ?? '';
        _gender = data['gender'];
        _maritalStatus = data['marital_status'];
        _employmentStatus = data['employment_status'];
        _financialStatus = data['financial_status'];
        _educationLevel = data['education_level'];
        _livingSituation = data['living_situation'];
        _anxietyDiagnosis = data['anxiety_diagnosis'];
        _onMedication = data['on_medication'];
        _sleepQuality =
            double.tryParse(data['sleep_quality_rating'] ?? '3') ?? 3;

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
        _profileImagePath = validProfileImagePath;
      });
    } else {
      setState(() {
        _displayName = displayName;
        _participantId = participantId;
        _displayNameController.text = displayName;
        _profileImagePath = validProfileImagePath;
      });
    }

    if (savedProfileImagePath != null && validProfileImagePath == null) {
      await prefs.remove('profile_image_path');
    }
  }

  @override
  void dispose() {
    _displayNameController.dispose();
    _ageController.dispose();
    super.dispose();
  }

  Future<void> _pickProfileImage() async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      if (!mounted) return;
      if (image != null) {
        setState(() {
          _profileImagePath = image.path;
        });
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString('profile_image_path', image.path);
      }
    } catch (e) {
      debugPrint('Error picking image: $e');
    }
  }

  Future<void> _removeProfileImage() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('profile_image_path');
    if (!mounted) return;
    setState(() => _profileImagePath = null);
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Profile picture removed from Aura.')),
    );
  }

  Future<void> _showProfileImageOptions() async {
    final action = await showModalBottomSheet<String>(
      context: context,
      showDragHandle: true,
      builder: (context) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.photo_library_outlined),
              title: Text(
                _profileImagePath == null
                    ? 'Choose profile picture'
                    : 'Change profile picture',
              ),
              onTap: () => Navigator.pop(context, 'choose'),
            ),
            if (_profileImagePath != null)
              ListTile(
                leading: const Icon(Icons.delete_outline, color: Colors.red),
                title: const Text(
                  'Remove profile picture',
                  style: TextStyle(color: Colors.red),
                ),
                subtitle: const Text(
                  'This removes it from Aura, not from your phone gallery.',
                ),
                onTap: () => Navigator.pop(context, 'remove'),
              ),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );

    if (action == 'choose') {
      await _pickProfileImage();
    } else if (action == 'remove') {
      await _removeProfileImage();
    }
  }

  Future<void> _saveTime(String period, TimeOfDay time) async {
    final prefs = await SharedPreferences.getInstance();
    if (!mounted) return;
    setState(() {
      if (period == 'morning') {
        _morningTime = time;
      } else if (period == 'afternoon') {
        _afternoonTime = time;
      } else if (period == 'evening') {
        _eveningTime = time;
      }
    });
    await prefs.setInt('ema_${period}_hour', time.hour);
    await prefs.setInt('ema_${period}_minute', time.minute);
    await DailyReminder.clearThrottleTimestamps();
  }

  Future<void> _openCheckInSettings() async {
    await Navigator.of(
      context,
    ).push(MaterialPageRoute(builder: (_) => const RatingSettingsPage()));
    await _loadProfile();
  }

  final _formKey = GlobalKey<FormState>();
  bool _isSaving = false;

  // --- Form fields ---
  final _ageController = TextEditingController();
  String? _gender;
  String? _maritalStatus;
  String? _employmentStatus;
  String? _financialStatus;
  String? _educationLevel;
  String? _livingSituation;
  String? _anxietyDiagnosis;
  String? _onMedication;
  double _sleepQuality = 3;

  // --- Notification Times ---
  TimeOfDay _morningTime = const TimeOfDay(hour: 9, minute: 0);
  TimeOfDay _afternoonTime = const TimeOfDay(hour: 14, minute: 0);
  TimeOfDay _eveningTime = const TimeOfDay(hour: 20, minute: 0);

  static const _genders = ['Male', 'Female', 'Non-binary', 'Prefer not to say'];
  static const _maritalStatuses = [
    'Single',
    'Married',
    'Divorced',
    'Widowed',
    'Other',
  ];
  static const _employmentStatuses = [
    'Student',
    'Employed (Full-time)',
    'Employed (Part-time)',
    'Unemployed',
    'Self-employed',
  ];
  static const _financialStatuses = [
    'Low income',
    'Lower-middle income',
    'Middle income',
    'Upper-middle income',
    'High income',
  ];
  static const _educationLevels = [
    'O/L or below',
    'A/L',
    'Undergraduate',
    'Postgraduate',
    'Other',
  ];
  static const _livingSituations = [
    'Alone',
    'With family',
    'With partner/spouse',
    'With friends/roommates',
    'University hostel',
    'Other',
  ];

  Future<void> _saveProfile() async {
    if (_isSaving) return;
    if (!_formKey.currentState!.validate()) return;
    if (_gender == null ||
        _maritalStatus == null ||
        _employmentStatus == null ||
        _financialStatus == null ||
        _educationLevel == null ||
        _livingSituation == null ||
        _anxietyDiagnosis == null ||
        _onMedication == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please fill in all fields.'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    setState(() => _isSaving = true);

    final prefs = await SharedPreferences.getInstance();
    final uid = prefs.getString('user_id') ?? 'Unknown';
    final displayName = _displayNameController.text.trim();

    await ParticipantIdentityService.updateDisplayName(displayName);

    final profile = {
      'age': _ageController.text.trim(),
      'gender': _gender,
      'marital_status': _maritalStatus,
      'employment_status': _employmentStatus,
      'financial_status': _financialStatus,
      'education_level': _educationLevel,
      'living_situation': _livingSituation,
      'anxiety_diagnosis': _anxietyDiagnosis,
      'on_medication': _onMedication,
      'sleep_quality_rating': _sleepQuality.round().toString(),
    };

    await BackgroundServiceHelper.sendToSheet(
      uid,
      'Demographics',
      jsonEncode(profile),
    );

    await prefs.setBool('profile_complete', true);
    await prefs.setString('user_profile_data', jsonEncode(profile));

    if (mounted) {
      setState(() {
        _displayName = displayName;
        _isSaving = false;
      });
      if (widget.isTab) {
        // In tab mode, go back to view mode after save
        setState(() => _isEditing = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Profile updated successfully!',
              style: GoogleFonts.poppins(fontSize: 13),
            ),
            backgroundColor: const Color(0xFF5E60CE),
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        );
      } else {
        // First-time setup: go to calibration screen next
        Navigator.pushReplacement(
          context,
          PageRouteBuilder(
            pageBuilder: (ctx, animation, secondaryAnimation) =>
                BaselineCalibrationPage(userId: uid),
            transitionsBuilder: (ctx, a, secondaryAnimation, c) =>
                FadeTransition(opacity: a, child: c),
            transitionDuration: const Duration(milliseconds: 800),
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    // Tab mode with data loaded: show attractive profile view or edit form
    if (widget.isTab && !_isEditing) {
      return _buildProfileView();
    }
    return _buildEditForm();
  }

  // ══════════════════════════════════════════════════════════════
  // ATTRACTIVE PROFILE VIEW (Tab Mode)
  // ══════════════════════════════════════════════════════════════

  Widget _buildProfileView() {
    const labels = ['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent'];
    final sleepLabel = labels[(_sleepQuality.round() - 1).clamp(0, 4)];

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        child: Column(
          children: [
            // ── Gradient Header with Avatar ──
            Container(
              width: double.infinity,
              padding: const EdgeInsets.only(top: 60, bottom: 30),
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [Color(0xFF667eea), Color(0xFF764ba2)],
                ),
                borderRadius: BorderRadius.only(
                  bottomLeft: Radius.circular(36),
                  bottomRight: Radius.circular(36),
                ),
              ),
              child: Column(
                children: [
                  // Avatar
                  GestureDetector(
                    onTap: _showProfileImageOptions,
                    child: Stack(
                      alignment: Alignment.bottomRight,
                      children: [
                        Container(
                          width: 90,
                          height: 90,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: Colors.white.withValues(alpha: 0.2),
                            border: Border.all(
                              color: Colors.white.withValues(alpha: 0.5),
                              width: 3,
                            ),
                            image: _profileImagePath != null
                                ? DecorationImage(
                                    image: FileImage(File(_profileImagePath!)),
                                    fit: BoxFit.cover,
                                  )
                                : null,
                          ),
                          child: _profileImagePath == null
                              ? Icon(
                                  _gender == 'Male'
                                      ? Icons.face_rounded
                                      : _gender == 'Female'
                                      ? Icons.face_3_rounded
                                      : Icons.person_rounded,
                                  color: Colors.white,
                                  size: 48,
                                )
                              : null,
                        ),
                        Container(
                          padding: const EdgeInsets.all(4),
                          decoration: const BoxDecoration(
                            color: Colors.white,
                            shape: BoxShape.circle,
                          ),
                          child: const Icon(
                            Icons.camera_alt_rounded,
                            color: Color(0xFF667eea),
                            size: 16,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 14),
                  Text(
                    _displayName.isNotEmpty ? _displayName : 'Aura user',
                    style: GoogleFonts.poppins(
                      fontSize: 22,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 16),
                  // Edit button
                  ElevatedButton.icon(
                    onPressed: () => setState(() => _isEditing = true),
                    icon: const Icon(Icons.edit_rounded, size: 18),
                    label: Text(
                      'Edit Profile',
                      style: GoogleFonts.poppins(fontWeight: FontWeight.w600),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: const Color(0xFF5E60CE),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(14),
                      ),
                      elevation: 0,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 24,
                        vertical: 10,
                      ),
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 20),

            // ── Info Cards ──
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Column(
                children: [
                  _infoCard('Personal Info', Icons.person_outline_rounded, [
                    _infoRow(
                      Icons.cake_rounded,
                      'Age',
                      _ageController.text.isNotEmpty
                          ? '${_ageController.text} years'
                          : 'Not set',
                    ),
                    _infoRow(Icons.wc_rounded, 'Gender', _gender ?? 'Not set'),
                    _infoRow(
                      Icons.favorite_rounded,
                      'Marital Status',
                      _maritalStatus ?? 'Not set',
                    ),
                  ]),
                  const SizedBox(height: 14),
                  _infoCard('Professional', Icons.work_outline_rounded, [
                    _infoRow(
                      Icons.business_center_rounded,
                      'Employment',
                      _employmentStatus ?? 'Not set',
                    ),
                    _infoRow(
                      Icons.account_balance_wallet_rounded,
                      'Financial Status',
                      _financialStatus ?? 'Not set',
                    ),
                    _infoRow(
                      Icons.school_rounded,
                      'Education',
                      _educationLevel ?? 'Not set',
                    ),
                    _infoRow(
                      Icons.home_rounded,
                      'Living Situation',
                      _livingSituation ?? 'Not set',
                    ),
                  ]),
                  const SizedBox(height: 14),
                  _infoCard('Health', Icons.health_and_safety_outlined, [
                    _infoRow(
                      Icons.psychology_rounded,
                      'Anxiety Diagnosis',
                      _anxietyDiagnosis ?? 'Not set',
                    ),
                    _infoRow(
                      Icons.medication_rounded,
                      'On Medication',
                      _onMedication ?? 'Not set',
                    ),
                    _infoRow(
                      Icons.bedtime_rounded,
                      'Sleep Quality',
                      sleepLabel,
                    ),
                  ]),
                  const SizedBox(height: 14),
                  _infoCard('Check-in Schedule', Icons.schedule_rounded, [
                    _timeTile(
                      'Morning',
                      _morningTime,
                      (t) => _saveTime('morning', t),
                      Icons.wb_sunny_rounded,
                    ),
                    _timeTile(
                      'Afternoon',
                      _afternoonTime,
                      (t) => _saveTime('afternoon', t),
                      Icons.wb_cloudy_rounded,
                    ),
                    _timeTile(
                      'Evening',
                      _eveningTime,
                      (t) => _saveTime('evening', t),
                      Icons.nightlight_round,
                    ),
                    SizedBox(
                      width: double.infinity,
                      child: OutlinedButton.icon(
                        onPressed: _openCheckInSettings,
                        icon: const Icon(Icons.tune_rounded),
                        label: const Text(
                          'Manage or turn off check-in reminders',
                        ),
                      ),
                    ),
                  ]),
                  const SizedBox(height: 14),
                  _buildDoctorConnectionCard(),
                  const SizedBox(height: 14),
                  _buildAppearanceCard(),
                  const SizedBox(height: 14),
                  _buildFaqSection(),
                  const SizedBox(height: 14),
                  _buildPrivacyCard(),
                  const SizedBox(height: 30),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _infoCard(String title, IconData icon, List<Widget> children) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 15,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: AppTheme.kPrimaryDeep.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(icon, color: AppTheme.kPrimaryDeep, size: 20),
              ),
              const SizedBox(width: 12),
              Text(
                title,
                style: GoogleFonts.poppins(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                  color: Theme.of(context).colorScheme.onSurface,
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          ...children,
        ],
      ),
    );
  }

  Widget _infoRow(IconData icon, String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        children: [
          Icon(
            icon,
            size: 18,
            color: Theme.of(context).colorScheme.onSurfaceVariant,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              label,
              style: GoogleFonts.poppins(
                fontSize: 13,
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
          ),
          Text(
            value,
            style: GoogleFonts.poppins(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: Theme.of(context).colorScheme.onSurface,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDoctorConnectionCard() {
    final participantId = _participantId;
    final canShare =
        participantId != null &&
        ParticipantIdentityService.isParticipantId(participantId);

    return _infoCard(
      'Doctor Connection',
      Icons.qr_code_2_rounded,
      [
        ListTile(
          contentPadding: EdgeInsets.zero,
          leading: const Icon(
            Icons.medical_information_outlined,
            color: AppTheme.kPrimaryDeep,
          ),
          title: Text(
            canShare ? 'Show my Patient ID QR' : 'Patient ID unavailable',
            style: GoogleFonts.poppins(fontWeight: FontWeight.w600),
          ),
          subtitle: Text(
            canShare
                ? 'Let your doctor scan this while you are together.'
                : 'Complete your Aura setup before connecting to a doctor.',
            style: GoogleFonts.poppins(fontSize: 12),
          ),
          trailing: canShare
              ? const Icon(Icons.chevron_right_rounded)
              : null,
          onTap: canShare
              ? () => Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) => ShareParticipantIdPage(
                        participantId: participantId!,
                      ),
                    ),
                  )
              : null,
        ),
      ],
    );
  }

  Widget _buildAppearanceCard() {
    final controller = ThemeController.instance;
    return AnimatedBuilder(
      animation: controller,
      builder: (context, _) => _infoCard('Appearance', Icons.palette_outlined, [
        ListTile(
          contentPadding: EdgeInsets.zero,
          leading: Icon(
            controller.isDarkNow
                ? Icons.dark_mode_rounded
                : Icons.light_mode_rounded,
            color: AppTheme.kPrimaryDeep,
          ),
          title: Text(
            controller.mode.label,
            style: GoogleFonts.poppins(fontWeight: FontWeight.w600),
          ),
          subtitle: Text(
            controller.mode == AppThemeMode.scheduled
                ? '${controller.darkStart.format(context)} to ${controller.darkEnd.format(context)}'
                : 'Tap to change light and dark theme settings',
            style: GoogleFonts.poppins(fontSize: 12),
          ),
          trailing: const Icon(Icons.chevron_right_rounded),
          onTap: () => Navigator.push(
            context,
            MaterialPageRoute(builder: (_) => const AppearanceSettingsPage()),
          ),
        ),
      ]),
    );
  }

  Widget _buildFaqSection() {
    const faqs = [
      (
        'What do my readings mean?',
        'It is an estimate based on your recent body signals and personal baseline. It is not a medical diagnosis.',
      ),
      (
        'How certain is the 10-minute outlook?',
        'The outlook is a model estimate, not a guarantee. Movement, poor sensor contact, illness, caffeine, and other factors can affect it.',
      ),
      (
        'Why are my body readings unavailable?',
        'Check that the chest strap is worn correctly, Bluetooth is on, and Aura has Bluetooth permission. Then return to the Body tab and reconnect.',
      ),
      (
        'What happens if I miss a check-in?',
        'Nothing bad. You can continue with the next check-in. Regular answers simply help the research data make more sense.',
      ),
      (
        'What data does Aura store?',
        'Aura stores study responses and approved sensor or activity information under a random Participant ID, not your display name.',
      ),
      (
        'Can I stop data collection?',
        'Yes. Open Manage My Data and Privacy below to request deletion or withdraw from the study.',
      ),
      (
        'Is Aura an emergency or treatment service?',
        'No. Aura does not replace professional care or emergency help. If you may be in immediate danger, contact local emergency services or a trusted person now.',
      ),
    ];

    return _infoCard('Help & FAQ', Icons.help_outline_rounded, [
      ...faqs.map(
        (faq) => ExpansionTile(
          tilePadding: EdgeInsets.zero,
          childrenPadding: const EdgeInsets.only(bottom: 12),
          title: Text(
            faq.$1,
            style: GoogleFonts.poppins(
              fontSize: 13,
              fontWeight: FontWeight.w600,
            ),
          ),
          children: [
            Align(
              alignment: Alignment.centerLeft,
              child: Text(
                faq.$2,
                style: GoogleFonts.poppins(fontSize: 12, height: 1.5),
              ),
            ),
          ],
        ),
      ),
    ]);
  }

  // ══════════════════════════════════════════════════════════════
  // EDIT FORM (original form, used in both standalone and edit mode)
  // ══════════════════════════════════════════════════════════════

  Widget _buildEditForm() {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: SafeArea(
        child: Form(
          key: _formKey,
          child: ListView(
            padding: const EdgeInsets.all(24),
            children: [
              // Back to view button in tab mode
              if (widget.isTab) ...[
                Align(
                  alignment: Alignment.centerLeft,
                  child: TextButton(
                    onPressed: () => setState(() => _isEditing = false),
                    child: Text(
                      'Back to Profile',
                      style: GoogleFonts.poppins(fontWeight: FontWeight.w500),
                    ),
                    style: TextButton.styleFrom(
                      foregroundColor: AppTheme.kPrimaryDeep,
                    ),
                  ),
                ),
                const SizedBox(height: 8),
              ],
              const SizedBox(height: 8),
              const Icon(
                Icons.person_pin,
                size: 52,
                color: AppTheme.kPrimaryDeep,
              ),
              const SizedBox(height: 12),
              Text(
                'User Profile',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Theme.of(context).colorScheme.onSurface,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                'This information is collected once and kept strictly confidential for research purposes.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 13,
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 28),

              _sectionLabel('Display Name'),
              TextFormField(
                controller: _displayNameController,
                textCapitalization: TextCapitalization.words,
                decoration: _inputDec('What should Aura call you?').copyWith(
                  helperText:
                      'You can change this later. It stays on this phone and does not change your Participant ID.',
                  helperMaxLines: 3,
                ),
                validator: (v) {
                  final name = v?.trim() ?? '';
                  if (name.isEmpty) return 'Required';
                  if (name.length > 80) {
                    return 'Use 80 characters or fewer';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 16),

              _sectionLabel('Age'),
              TextFormField(
                controller: _ageController,
                keyboardType: TextInputType.number,
                decoration: _inputDec('Enter your age'),
                validator: (v) {
                  if (v == null || v.isEmpty) return 'Required';
                  final n = int.tryParse(v);
                  if (n == null || n < 18 || n > 30) {
                    return 'This study is for ages 18 to 30';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 16),

              _sectionLabel('Gender'),
              _dropdown(_genders, _gender, (v) => setState(() => _gender = v)),
              const SizedBox(height: 16),

              _sectionLabel('Marital Status'),
              _dropdown(
                _maritalStatuses,
                _maritalStatus,
                (v) => setState(() => _maritalStatus = v),
              ),
              const SizedBox(height: 16),

              _sectionLabel('Employment Status'),
              _dropdown(
                _employmentStatuses,
                _employmentStatus,
                (v) => setState(() => _employmentStatus = v),
              ),
              const SizedBox(height: 16),

              _sectionLabel('Financial Status'),
              _dropdown(
                _financialStatuses,
                _financialStatus,
                (v) => setState(() => _financialStatus = v),
              ),
              const SizedBox(height: 16),

              _sectionLabel('Highest Education Level'),
              _dropdown(
                _educationLevels,
                _educationLevel,
                (v) => setState(() => _educationLevel = v),
              ),
              const SizedBox(height: 16),

              _sectionLabel('Living Situation'),
              _dropdown(
                _livingSituations,
                _livingSituation,
                (v) => setState(() => _livingSituation = v),
              ),
              const SizedBox(height: 16),

              _sectionLabel(
                'Has a health professional ever told you that you have an anxiety disorder?',
              ),
              _radioGroup(
                ['Yes', 'No', 'Unsure'],
                _anxietyDiagnosis,
                (v) => setState(() => _anxietyDiagnosis = v),
              ),
              const SizedBox(height: 16),

              _sectionLabel(
                'Do you currently take medicine for anxiety or another mental health condition?',
              ),
              _radioGroup(
                ['Yes', 'No', 'Prefer not to say'],
                _onMedication,
                (v) => setState(() => _onMedication = v),
              ),
              const SizedBox(height: 16),

              _sectionLabel(
                'How would you rate your typical sleep quality? (1 = Very Poor, 5 = Excellent)',
              ),
              const SizedBox(height: 8),
              _buildSlider(),
              const SizedBox(height: 32),

              const Divider(),
              const SizedBox(height: 24),
              const Text(
                'Privacy & Data Rights',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 12),
              _buildPrivacyCard(),
              const SizedBox(height: 32),

              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _isSaving ? null : _saveProfile,
                  child: _isSaving
                      ? const SizedBox(
                          height: 20,
                          width: 20,
                          child: CircularProgressIndicator(
                            color: Colors.white,
                            strokeWidth: 2,
                          ),
                        )
                      : const Text('Save & Continue'),
                ),
              ),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }

  Widget _sectionLabel(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Text(
        text,
        style: TextStyle(
          fontSize: 14,
          fontWeight: FontWeight.w600,
          color: Theme.of(context).colorScheme.onSurface,
        ),
      ),
    );
  }

  InputDecoration _inputDec(String hint) {
    return InputDecoration(
      hintText: hint,
      filled: true,
      fillColor: Theme.of(context).colorScheme.surface,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide.none,
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide(
          color: Theme.of(context).colorScheme.outlineVariant,
        ),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: AppTheme.kPrimaryDeep, width: 2),
      ),
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
    );
  }

  Widget _dropdown(
    List<String> items,
    String? value,
    ValueChanged<String?> onChanged,
  ) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: value,
          isExpanded: true,
          hint: const Text('Select an option'),
          items: items
              .map((e) => DropdownMenuItem(value: e, child: Text(e)))
              .toList(),
          onChanged: onChanged,
        ),
      ),
    );
  }

  /// FIX: replaced deprecated RadioListTile groupValue/onChanged
  /// with RadioGroup ancestor pattern (Flutter 3.32+)
  Widget _radioGroup(
    List<String> options,
    String? groupValue,
    ValueChanged<String?> onChanged,
  ) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
      ),
      child: RadioGroup<String>(
        groupValue: groupValue,
        onChanged: onChanged,
        child: Column(
          children: options
              .map(
                (o) => RadioListTile<String>(
                  title: Text(o),
                  value: o,
                  dense: true,
                  contentPadding: EdgeInsets.zero,
                ),
              )
              .toList(),
        ),
      ),
    );
  }

  Widget _buildSlider() {
    const labels = ['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent'];
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
      ),
      child: Column(
        children: [
          Slider(
            value: _sleepQuality,
            min: 1,
            max: 5,
            divisions: 4,
            activeColor: AppTheme.kPrimaryDeep,
            label: labels[_sleepQuality.round() - 1],
            onChanged: (v) => setState(() => _sleepQuality = v),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: labels
                .map(
                  (l) => Text(
                    l,
                    style: TextStyle(
                      fontSize: 10,
                      color: Theme.of(context).colorScheme.onSurfaceVariant,
                    ),
                  ),
                )
                .toList(),
          ),
        ],
      ),
    );
  }

  Widget _timeTile(
    String label,
    TimeOfDay time,
    ValueChanged<TimeOfDay> onChanged,
    IconData icon,
  ) {
    return Card(
      elevation: 0,
      margin: const EdgeInsets.only(bottom: 12),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: Theme.of(context).colorScheme.outlineVariant),
      ),
      child: ListTile(
        leading: Icon(icon, color: AppTheme.kPrimaryDeep),
        title: Text(label, style: const TextStyle(fontWeight: FontWeight.w600)),
        trailing: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            color: AppTheme.kPrimaryDeep.withValues(alpha: 0.1),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Text(
            time.format(context),
            style: const TextStyle(
              color: AppTheme.kPrimaryDeep,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
        onTap: () async {
          final picked = await showTimePicker(
            context: context,
            initialTime: time,
          );
          if (picked != null) onChanged(picked);
        },
      ),
    );
  }

  Widget _buildPrivacyCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.verified_user_outlined, color: Colors.green, size: 20),
              SizedBox(width: 8),
              Text(
                "Consent Status: Active",
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  color: Colors.green,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            "Your data is stored under a Participant ID instead of your name. "
            "Anything that could identify you is kept separate from your health information.",
            style: TextStyle(
              fontSize: 12,
              color: Theme.of(context).colorScheme.onSurfaceVariant,
            ),
          ),
          const SizedBox(height: 16),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: () => Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const DataRightsPage()),
              ),
              icon: const Icon(Icons.shield_outlined, size: 18),
              label: const Text('Manage My Data and Privacy'),
              style: OutlinedButton.styleFrom(
                foregroundColor: AppTheme.kPrimaryDeep,
                side: const BorderSide(color: AppTheme.kPrimaryDeep),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            "Contact: ${ServiceConfig.researchTeamEmail}",
            style: const TextStyle(fontSize: 11, color: Colors.grey),
          ),
        ],
      ),
    );
  }
}
