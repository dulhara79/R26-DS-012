import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_background_service/flutter_background_service.dart';
import '../theme/app_theme.dart';
import '../services/background/service_config.dart';
import '../services/background_service_helper.dart';
import 'informed_consent_page.dart';
import 'privacy_policy_page.dart';
import '../main.dart';

class DataRightsPage extends StatefulWidget {
  const DataRightsPage({super.key});

  @override
  State<DataRightsPage> createState() => _DataRightsPageState();
}

class _DataRightsPageState extends State<DataRightsPage> {
  String _participantId = '';
  String _consentTimestamp = '';
  String _consentVersion = '';
  bool _isWithdrawn = false;

  @override
  void initState() {
    super.initState();
    _loadStatus();
  }

  Future<void> _loadStatus() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _participantId = prefs.getString('user_id') ?? 'Not set';
      _consentTimestamp = prefs.getString('consent_timestamp') ?? 'Unknown';
      _consentVersion = prefs.getString('consent_version') ?? '1.0';
      _isWithdrawn = prefs.getBool('consent_withdrawn') ?? false;
    });
  }

  // ── Request Data Export (Right to Access) ──
  Future<void> _requestDataExport() async {
    final confirm = await _showConfirmDialog(
      title: 'Request Data Export',
      content: 'This will send a request to the research team to provide you with '
          'a copy of all data collected about you.\n\n'
          'You will be contacted at the email associated with your research enrollment.',
      confirmText: 'Request Export',
      confirmColor: AppTheme.kPrimaryDeep,
    );
    if (confirm != true) return;

    final prefs = await SharedPreferences.getInstance();
    final uid = prefs.getString('user_id') ?? 'Unknown';
    
    await BackgroundServiceHelper.sendToSheet(
      uid,
      'Data_Export_Request',
      jsonEncode({
        'participant_id': uid,
        'timestamp': DateTime.now().toIso8601String(),
        'consent_version': _consentVersion,
        'request_type': 'access',
      }),
    );

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Data export request submitted. The research team will process your request.'),
          backgroundColor: Colors.green,
          duration: Duration(seconds: 4),
        ),
      );
    }
  }

  // ── Request Data Deletion (Right to Erasure) ──
  Future<void> _requestDataDeletion() async {
    final confirm = await _showConfirmDialog(
      title: '⚠️ Request Data Deletion',
      content: 'This will send a formal request to the research team to permanently '
          'delete all data associated with your Participant ID.\n\n'
          'This action is logged and cannot be undone. The research team will '
          'process your request within 21 business days as required by the PDPA.',
      confirmText: 'Request Deletion',
      confirmColor: Colors.red,
    );
    if (confirm != true) return;

    final prefs = await SharedPreferences.getInstance();
    final uid = prefs.getString('user_id') ?? 'Unknown';
    
    await BackgroundServiceHelper.sendToSheet(
      uid,
      'Data_Deletion_Request',
      jsonEncode({
        'participant_id': uid,
        'timestamp': DateTime.now().toIso8601String(),
        'consent_version': _consentVersion,
        'request_type': 'erasure',
      }),
    );

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Data deletion request submitted. Processing within 21 business days.'),
          backgroundColor: Colors.orange,
          duration: Duration(seconds: 4),
        ),
      );
    }
  }

  // ── Withdraw from Study ──
  Future<void> _withdrawFromStudy() async {
    final confirm = await _showConfirmDialog(
      title: '⚠️ Withdraw from Study',
      content: 'This will:\n\n'
          '• Stop all data collection immediately\n'
          '• Stop the background service\n'
          '• Clear all locally stored data\n'
          '• Send a withdrawal record to the research team\n\n'
          'You can also request deletion of previously collected data.\n\n'
          'This action cannot be undone. Are you sure?',
      confirmText: 'Yes, Withdraw',
      confirmColor: Colors.red,
    );
    if (confirm != true) return;

    // Double confirmation
    final doubleConfirm = await _showConfirmDialog(
      title: 'Final Confirmation',
      content: 'Please confirm one more time that you wish to withdraw from '
          'the research study. All local data will be erased.',
      confirmText: 'Confirm Withdrawal',
      confirmColor: Colors.red.shade700,
    );
    if (doubleConfirm != true) return;

    final prefs = await SharedPreferences.getInstance();
    final uid = prefs.getString('user_id') ?? 'Unknown';
    
    // 1. Send withdrawal record to Google Sheet
    await BackgroundServiceHelper.sendToSheet(
      uid,
      'Consent_Withdrawal',
      jsonEncode({
        'participant_id': uid,
        'timestamp': DateTime.now().toIso8601String(),
        'consent_version': _consentVersion,
        'original_consent_timestamp': _consentTimestamp,
        'reason': 'Participant voluntary withdrawal via app',
      }),
      immediate: true,
    );

    // 2. Stop background service
    try {
      final service = FlutterBackgroundService();
      service.invoke('stopService');
    } catch (_) {}

    // 3. Mark as withdrawn and clear local data
    await prefs.setBool('consent_withdrawn', true);
    await prefs.setString('withdrawal_timestamp', DateTime.now().toIso8601String());
    
    // Clear all study data but keep withdrawal record
    await prefs.remove('user_id');
    await prefs.remove('consent_accepted');
    await prefs.remove('profile_complete');
    await prefs.remove('user_profile_data');
    await prefs.remove('offline_queue');
    await prefs.remove('last_battery_level');
    await prefs.remove('rating_enabled');

    if (mounted) {
      // Navigate to the beginning
      Navigator.of(context).pushAndRemoveUntil(
        MaterialPageRoute(builder: (_) => const SplashRouter()),
        (route) => false,
      );
    }
  }

  Future<bool?> _showConfirmDialog({
    required String title,
    required String content,
    required String confirmText,
    required Color confirmColor,
  }) {
    return showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
        content: Text(content, style: const TextStyle(fontSize: 14, height: 1.5)),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: ElevatedButton.styleFrom(backgroundColor: confirmColor),
            child: Text(confirmText, style: const TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.kBgTop,
      appBar: AppBar(
        title: const Text('Data Rights & Privacy'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          // ── Consent Status Card ──
          _buildStatusCard(),
          const SizedBox(height: 20),

          // ── Your Rights ──
          const Text(
            'Your Rights Under the PDPA',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 12),

          // Right to Access
          _buildActionCard(
            icon: Icons.download_outlined,
            iconColor: Colors.blue,
            title: 'Request My Data',
            subtitle: 'Get a copy of all data collected about you (Right to Access)',
            onTap: _isWithdrawn ? null : _requestDataExport,
          ),

          // Right to Erasure
          _buildActionCard(
            icon: Icons.delete_forever_outlined,
            iconColor: Colors.orange,
            title: 'Request Data Deletion',
            subtitle: 'Request permanent deletion of your data (Right to Erasure)',
            onTap: _isWithdrawn ? null : _requestDataDeletion,
          ),

          // Withdraw from Study
          _buildActionCard(
            icon: Icons.exit_to_app_rounded,
            iconColor: Colors.red,
            title: 'Withdraw from Study',
            subtitle: 'Stop all data collection and clear local data',
            onTap: _isWithdrawn ? null : _withdrawFromStudy,
          ),

          const SizedBox(height: 24),

          // ── Documents ──
          const Text(
            'Documents',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 12),

          _buildActionCard(
            icon: Icons.description_outlined,
            iconColor: AppTheme.kPrimaryDeep,
            title: 'Privacy Policy',
            subtitle: 'Full PDPA-compliant privacy notice',
            onTap: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const PrivacyPolicyPage()),
            ),
          ),

          _buildActionCard(
            icon: Icons.gavel_outlined,
            iconColor: AppTheme.kPrimaryDeep,
            title: 'View Informed Consent',
            subtitle: 'Review the consent document you agreed to',
            onTap: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const InformedConsentPage()),
            ),
          ),

          const SizedBox(height: 24),

          // ── Contact ──
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.grey.shade50,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: Colors.grey.shade200),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Contact the Research Team',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
                const SizedBox(height: 8),
                Text(
                  'For any queries about your data or rights:\n'
                  '${ServiceConfig.researchTeamEmail}\n\n'
                  'Ethics Review Committee:\n'
                  '${ServiceConfig.ercSecretaryEmail}',
                  style: TextStyle(fontSize: 13, color: Colors.grey.shade700, height: 1.5),
                ),
              ],
            ),
          ),
          const SizedBox(height: 30),
        ],
      ),
    );
  }

  Widget _buildStatusCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: _isWithdrawn
              ? [Colors.red.shade50, Colors.red.shade100]
              : [Colors.green.shade50, Colors.green.shade100],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: _isWithdrawn ? Colors.red.shade300 : Colors.green.shade300,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                _isWithdrawn ? Icons.cancel_outlined : Icons.verified_user_outlined,
                color: _isWithdrawn ? Colors.red : Colors.green,
                size: 22,
              ),
              const SizedBox(width: 8),
              Text(
                _isWithdrawn ? 'Consent Status: Withdrawn' : 'Consent Status: Active',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  color: _isWithdrawn ? Colors.red.shade800 : Colors.green.shade800,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          _statusRow('Participant ID', _participantId),
          _statusRow('Consent Given', _formatTimestamp(_consentTimestamp)),
          _statusRow('Consent Version', _consentVersion),
          _statusRow('Study', ServiceConfig.studyTitle),
        ],
      ),
    );
  }

  Widget _statusRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 120,
            child: Text(label, style: TextStyle(fontSize: 12, color: Colors.grey.shade600)),
          ),
          Expanded(
            child: Text(value, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500)),
          ),
        ],
      ),
    );
  }

  Widget _buildActionCard({
    required IconData icon,
    required Color iconColor,
    required String title,
    required String subtitle,
    required VoidCallback? onTap,
  }) {
    return Card(
      elevation: 0,
      margin: const EdgeInsets.only(bottom: 10),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(14),
        side: BorderSide(color: Colors.grey.shade200),
      ),
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: iconColor.withValues(alpha: 0.1),
          child: Icon(icon, color: iconColor, size: 22),
        ),
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14)),
        subtitle: Text(subtitle, style: TextStyle(fontSize: 12, color: Colors.grey.shade600)),
        trailing: Icon(Icons.chevron_right, color: onTap != null ? Colors.grey : Colors.grey.shade300),
        onTap: onTap,
        enabled: onTap != null,
      ),
    );
  }

  String _formatTimestamp(String isoTimestamp) {
    try {
      final dt = DateTime.parse(isoTimestamp);
      return '${dt.year}-${dt.month.toString().padLeft(2, '0')}-${dt.day.toString().padLeft(2, '0')} '
          '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
    } catch (_) {
      return isoTimestamp;
    }
  }
}
