// ─── support_set_screen.dart ─────────────────────────────────────────────────
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/models.dart';
import '../theme/app_theme.dart';
import '../services/patient_provider.dart';

class SupportSetScreen extends StatefulWidget {
  const SupportSetScreen({super.key});

  @override
  State<SupportSetScreen> createState() => _SupportSetScreenState();
}

class _SupportSetScreenState extends State<SupportSetScreen> {
  final _ctrl  = TextEditingController();
  String _label = 'anxiety';

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<PatientProvider>();

    return Scaffold(
      backgroundColor: AppColors.surfaceSecond,
      appBar: AppBar(title: const Text('Support set manager')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Explainer card
            Container(
              decoration: BoxDecoration(
                color: AppColors.primarySurface,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppColors.primaryLighter.withOpacity(0.4)),
              ),
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(children: [
                    const Icon(Icons.info_outline_rounded,
                        size: 18, color: AppColors.primary),
                    const SizedBox(width: 8),
                    Text('How few-shot adaptation works',
                        style: Theme.of(context).textTheme.titleMedium
                            ?.copyWith(color: AppColors.primary)),
                  ]),
                  const SizedBox(height: 8),
                  const Text(
                    'TC-WPN adapts to your clinical site without retraining. '
                    'Add labeled notes here — the model builds prototypes from '
                    'these examples and uses them to classify new notes. '
                    'More diverse examples → better adaptation.',
                    style: TextStyle(
                        fontSize: 13, color: AppColors.textSecondary, height: 1.5),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // Stats row
            Row(
              children: [
                _StatCard(
                  label: 'Anxiety notes',
                  value: '${provider.anxietySupport.length}',
                  color: AppColors.riskHigh,
                  bg:    AppColors.riskHighBg,
                ),
                const SizedBox(width: 10),
                _StatCard(
                  label: 'Control notes',
                  value: '${provider.controlSupport.length}',
                  color: AppColors.riskLow,
                  bg:    AppColors.riskLowBg,
                ),
              ],
            ),
            const SizedBox(height: 16),

            // Add note form
            Text('Add labeled note', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 10),

            // Label selector
            Row(
              children: ['anxiety', 'control'].map((l) {
                final sel = _label == l;
                return Expanded(
                  child: GestureDetector(
                    onTap: () => setState(() => _label = l),
                    child: Container(
                      margin: EdgeInsets.only(right: l == 'anxiety' ? 8 : 0),
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      decoration: BoxDecoration(
                        color: sel
                            ? (l == 'anxiety'
                                ? AppColors.riskHighBg
                                : AppColors.riskLowBg)
                            : AppColors.surface,
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(
                          color: sel
                              ? (l == 'anxiety'
                                  ? AppColors.riskHigh
                                  : AppColors.riskLow)
                              : AppColors.border,
                          width: sel ? 1.5 : 0.8,
                        ),
                      ),
                      alignment: Alignment.center,
                      child: Text(
                        l[0].toUpperCase() + l.substring(1),
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          color: sel
                              ? (l == 'anxiety'
                                  ? AppColors.riskHigh
                                  : AppColors.riskLow)
                              : AppColors.textSecondary,
                        ),
                      ),
                    ),
                  ),
                );
              }).toList(),
            ),
            const SizedBox(height: 10),

            TextField(
              controller: _ctrl,
              maxLines: 6,
              decoration: const InputDecoration(
                hintText: 'Paste clinical note text here...',
                labelText: 'Clinical note',
                alignLabelWithHint: true,
              ),
            ),
            const SizedBox(height: 10),

            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () {
                      final t = _ctrl.text.trim();
                      if (t.isEmpty) return;
                      provider.addSupportNote(t, _label);
                      _ctrl.clear();
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text('Added to $_label support set'),
                          backgroundColor: AppColors.primary,
                        ),
                      );
                    },
                    icon: const Icon(Icons.add_rounded, size: 18),
                    label: const Text('Add note'),
                  ),
                ),
                const SizedBox(width: 10),
                OutlinedButton(
                  onPressed: provider.supportNotes.isEmpty
                      ? null
                      : () => _confirmClear(context, provider),
                  child: const Text('Clear all'),
                ),
              ],
            ),
            const SizedBox(height: 20),

            // Current support notes
            if (provider.supportNotes.isNotEmpty) ...[
              Text('Current support set',
                  style: Theme.of(context).textTheme.titleLarge),
              const SizedBox(height: 10),
              ...provider.supportNotes.map((n) => _SupportNoteItem(
                note: n,
                onDelete: () => provider.removeSupportNote(n.id),
              )),
            ],
          ],
        ),
      ),
    );
  }

  Future<void> _confirmClear(
      BuildContext context, PatientProvider provider) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Clear support set?'),
        content: const Text(
          'This will remove all labeled notes. '
          'Default examples will be used for inference.',
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, true),
            style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.riskHigh),
            child: const Text('Clear all'),
          ),
        ],
      ),
    );
    if (ok == true) provider.clearSupportNotes();
  }
}

class _StatCard extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  final Color bg;
  const _StatCard({required this.label, required this.value,
      required this.color, required this.bg});

  @override
  Widget build(BuildContext context) => Expanded(
    child: Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(value,
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.w700,
                  color: color)),
          Text(label,
              style: TextStyle(fontSize: 12, color: color.withOpacity(0.8))),
        ],
      ),
    ),
  );
}

class _SupportNoteItem extends StatelessWidget {
  final SupportNote note;
  final VoidCallback onDelete;
  const _SupportNoteItem({required this.note, required this.onDelete});

  @override
  Widget build(BuildContext context) {
    final isAnx = note.label == 'anxiety';
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.border, width: 0.8),
      ),
      padding: const EdgeInsets.all(12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 8, height: 8,
            margin: const EdgeInsets.only(top: 4),
            decoration: BoxDecoration(
              color: isAnx ? AppColors.riskHigh : AppColors.riskLow,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  isAnx ? 'Anxiety' : 'Control',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    color: isAnx ? AppColors.riskHigh : AppColors.riskLow,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  note.text.length > 120
                      ? '${note.text.substring(0, 120)}...'
                      : note.text,
                  style: const TextStyle(
                      fontSize: 12, color: AppColors.textSecondary,
                      height: 1.4),
                ),
              ],
            ),
          ),
          IconButton(
            icon: const Icon(Icons.delete_outline_rounded,
                size: 18, color: AppColors.textHint),
            onPressed: onDelete,
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(),
          ),
        ],
      ),
    );
  }
}


// ─── settings_screen.dart ─────────────────────────────────────────────────────
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
          Container(
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppColors.border, width: 0.8),
            ),
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Container(
                  width: 52, height: 52,
                  decoration: const BoxDecoration(
                    color: AppColors.primarySurface,
                    shape: BoxShape.circle,
                  ),
                  alignment: Alignment.center,
                  child: const Text('DK',
                      style: TextStyle(
                          fontSize: 18, fontWeight: FontWeight.w700,
                          color: AppColors.primary)),
                ),
                const SizedBox(width: 14),
                const Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Dr. Kaushalya',
                          style: TextStyle(fontSize: 16,
                              fontWeight: FontWeight.w600)),
                      Text('Consultant Psychiatrist · NHSL',
                          style: TextStyle(fontSize: 12,
                              color: AppColors.textSecondary)),
                      Text('ID: DR001',
                          style: TextStyle(fontSize: 12,
                              color: AppColors.textHint)),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 20),

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
            _SettingsTile(
              icon: Icons.model_training_rounded,
              title: 'NHSL adaptation',
              subtitle: 'Pending — finetuning in progress',
              valueColor: AppColors.riskModerate,
            ),
          ]),
          const SizedBox(height: 16),

          _SettingsSection('Performance metrics', [
            _SettingsTile(
              icon: Icons.show_chart_rounded,
              title: 'Test AUROC (K=5)',
              subtitle: '0.9635  [95% CI: 0.959–0.968]',
            ),
            _SettingsTile(
              icon: Icons.bar_chart_rounded,
              title: 'Macro F1 (K=5)',
              subtitle: '0.9037',
            ),
            _SettingsTile(
              icon: Icons.data_usage_rounded,
              title: 'Training data',
              subtitle: 'MIMIC-IV + MIMIC-III · 4,640 records',
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
            _SettingsTile(
              icon: Icons.privacy_tip_outlined,
              title: 'Data privacy',
              subtitle: 'No raw note text is stored after embedding',
            ),
          ]),
          const SizedBox(height: 24),

          OutlinedButton.icon(
            onPressed: () => Navigator.of(context).popUntil((r) => r.isFirst),
            icon: const Icon(Icons.logout_rounded, size: 18),
            label: const Text('Sign out'),
            style: OutlinedButton.styleFrom(
              foregroundColor: AppColors.riskHigh,
              side: const BorderSide(color: AppColors.riskHigh),
            ),
          ),
          const SizedBox(height: 24),
        ],
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
          padding: const EdgeInsets.only(bottom: 8),
          child: Text(title,
              style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: AppColors.textSecondary,
                  letterSpacing: 0.5)),
        ),
        Container(
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: AppColors.border, width: 0.8),
          ),
          child: Column(
            children: children.asMap().entries.map((e) => Column(
              children: [
                e.value,
                if (e.key < children.length - 1)
                  const Divider(height: 0, indent: 52),
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
  final Color? valueColor;
  const _SettingsTile({
    required this.icon, required this.title,
    required this.subtitle, this.valueColor,
  });

  @override
  Widget build(BuildContext context) {
    return ListTile(
      leading: Container(
        width: 36, height: 36,
        decoration: BoxDecoration(
          color: AppColors.primarySurface,
          borderRadius: BorderRadius.circular(8),
        ),
        alignment: Alignment.center,
        child: Icon(icon, size: 18, color: AppColors.primary),
      ),
      title: Text(title,
          style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500)),
      subtitle: Text(subtitle,
          style: TextStyle(
              fontSize: 12,
              color: valueColor ?? AppColors.textSecondary)),
      contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 4),
    );
  }
}
