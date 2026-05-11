import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.surfaceSecond,
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Profile card
          _buildProfileCard(),
          const SizedBox(height: 24),

          _SettingsSection('Notification settings', [
            _SettingsTile(
              icon: Icons.notifications_active_rounded,
              title: 'Assessment alerts',
              subtitle: 'Sound and vibration for new assessments',
              trailing: Switch(value: true, onChanged: (v) {}),
            ),
            _SettingsTile(
              icon: Icons.volume_up_rounded,
              title: 'Alert sound',
              subtitle: 'System default (Medical Alert)',
            ),
          ]),
          const SizedBox(height: 16),

          _SettingsSection('Model configuration', [
            _SettingsTile(
              icon: Icons.link_rounded,
              title: 'API endpoint',
              subtitle: 'dulharakaushalya-tc-wpn-demo.hf.space',
            ),
            _SettingsTile(
              icon: Icons.tune_rounded,
              title: 'Decision threshold',
              subtitle: '0.4036 (locked from validation)',
            ),
            _SettingsTile(
              icon: Icons.analytics_rounded,
              title: 'Model version',
              subtitle: 'TC-WPN v1.0 · Val AUROC 0.9671',
            ),
          ]),
          const SizedBox(height: 16),

          _SettingsSection('Security', [
             _SettingsTile(
              icon: Icons.fingerprint_rounded,
              title: 'Biometric login',
              subtitle: 'Enabled for enhanced patient privacy',
              trailing: Switch(value: true, onChanged: (v) {}),
            ),
          ]),
          const SizedBox(height: 16),

          _SettingsSection('About', [
            _SettingsTile(
              icon: Icons.info_outline_rounded,
              title: 'Research component',
              subtitle: 'TC-WPN for Few-Shot Clinical Anxiety Detection',
            ),
            _SettingsTile(
              icon: Icons.school_rounded,
              title: 'Institution',
              subtitle: 'SLIIT · Faculty of Computing',
            ),
          ]),
          const SizedBox(height: 24),

          _buildSignOutButton(context),
          const SizedBox(height: 24),
        ],
      ),
    );
  }

  Widget _buildProfileCard() {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppColors.border, width: 0.8),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.03),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      padding: const EdgeInsets.all(20),
      child: Row(
        children: [
          Container(
            width: 64, height: 64,
            decoration: const BoxDecoration(
              color: AppColors.primarySurface,
              shape: BoxShape.circle,
            ),
            alignment: Alignment.center,
            child: const Text('DK',
                style: TextStyle(
                    fontSize: 22, fontWeight: FontWeight.bold,
                    color: AppColors.primary)),
          ),
          const SizedBox(width: 16),
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Dr. Kaushalya',
                    style: TextStyle(fontSize: 18,
                        fontWeight: FontWeight.bold)),
                Text('Consultant Psychiatrist',
                    style: TextStyle(fontSize: 13,
                        color: AppColors.textSecondary)),
                Text('NHSL · Ward 04',
                    style: TextStyle(fontSize: 12,
                        color: AppColors.textHint)),
              ],
            ),
          ),
          IconButton(
            icon: const Icon(Icons.edit_outlined, size: 20),
            onPressed: () {},
          ),
        ],
      ),
    );
  }

  Widget _buildSignOutButton(BuildContext context) {
    return OutlinedButton.icon(
      onPressed: () => Navigator.of(context).popUntil((r) => r.isFirst),
      icon: const Icon(Icons.logout_rounded, size: 18),
      label: const Text('Sign out'),
      style: OutlinedButton.styleFrom(
        foregroundColor: AppColors.riskHigh,
        side: const BorderSide(color: AppColors.riskHigh),
        padding: const EdgeInsets.symmetric(vertical: 14),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
  }
}

class _SettingsSection extends StatelessWidget {
  final String title;
  final List<Widget> children;
  const _SettingsSection(this.title, this.children);

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(left: 4, bottom: 8),
          child: Text(title.toUpperCase(),
              style: const TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                  color: AppColors.textHint,
                  letterSpacing: 1.1)),
        ),
        Container(
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: AppColors.border, width: 0.8),
          ),
          child: Column(
            children: children.asMap().entries.map((e) => Column(
              children: [
                e.value,
                if (e.key < children.length - 1)
                  const Divider(height: 0, indent: 56),
              ],
            )).toList(),
          ),
        ),
      ],
    );
  }
}

class _SettingsTile extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final Widget? trailing;

  const _SettingsTile({
    required this.icon, required this.title,
    required this.subtitle, this.trailing,
  });

  @override
  Widget build(BuildContext context) {
    return ListTile(
      leading: Container(
        width: 38, height: 38,
        decoration: BoxDecoration(
          color: AppColors.primarySurface,
          borderRadius: BorderRadius.circular(10),
        ),
        alignment: Alignment.center,
        child: Icon(icon, size: 20, color: AppColors.primary),
      ),
      title: Text(title,
          style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600)),
      subtitle: Text(subtitle,
          style: const TextStyle(fontSize: 12, color: AppColors.textSecondary)),
      trailing: trailing,
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
    );
  }
}
