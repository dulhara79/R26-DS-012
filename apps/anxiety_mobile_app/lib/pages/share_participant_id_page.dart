import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:qr_flutter/qr_flutter.dart';

import '../services/api_service.dart';
import '../services/participant_identity_service.dart';

class ShareParticipantIdPage extends StatelessWidget {
  final String participantId;

  const ShareParticipantIdPage({super.key, required this.participantId});

  String get _qrData => 'clinanx://patient/$participantId';

  Future<void> _connectWithPairingCode(BuildContext context) async {
    final formKey = GlobalKey<FormState>();
    final controller = TextEditingController();
    final pairingCode = await showDialog<String>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Enter pairing code'),
        content: Form(
          key: formKey,
          child: TextFormField(
            controller: controller,
            autofocus: true,
            textCapitalization: TextCapitalization.characters,
            inputFormatters: [
              FilteringTextInputFormatter.allow(RegExp(r'[A-Za-z0-9-]')),
              LengthLimitingTextInputFormatter(9),
            ],
            decoration: const InputDecoration(
              hintText: 'XXXX-XXXX',
              helperText: 'Use the code your doctor gives you.',
            ),
            validator: (value) {
              final code = value?.trim().toUpperCase() ?? '';
              if (!RegExp(r'^[A-Z0-9]{4}-[A-Z0-9]{4}$').hasMatch(code)) {
                return 'Enter the code in XXXX-XXXX format.';
              }
              return null;
            },
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () {
              if (formKey.currentState?.validate() != true) return;
              Navigator.pop(
                dialogContext,
                controller.text.trim().toUpperCase(),
              );
            },
            child: const Text('Connect'),
          ),
        ],
      ),
    );
    controller.dispose();

    if (pairingCode == null || !context.mounted) return;

    final messenger = ScaffoldMessenger.of(context);
    messenger.showSnackBar(
      const SnackBar(
        duration: Duration(seconds: 15),
        content: Text('Connecting to your doctor...'),
      ),
    );

    final result = await ApiService.pairWithCentralBackend(
      participantId: participantId,
      pairingCode: pairingCode,
    );
    if (result['success'] == true) {
      await ParticipantIdentityService.saveCentralSubjectId(
        result['subject_id'].toString(),
      );
    }

    if (!context.mounted) return;
    messenger.hideCurrentSnackBar();
    messenger.showSnackBar(
      SnackBar(
        content: Text(
          result['success'] == true
              ? 'Aura is now connected to your doctor.'
              : result['message']?.toString() ?? 'Could not connect Aura.',
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Connect to Doctor')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            children: [
              Text(
                'Ask your doctor to scan this code',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 10),
              Text(
                'The code contains only your Aura Participant ID. '
                'It does not contain your name, readings, or diagnosis.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  height: 1.5,
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 28),
              Container(
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color: Theme.of(context).colorScheme.outlineVariant,
                  ),
                ),
                child: QrImageView(
                  data: _qrData,
                  version: QrVersions.auto,
                  size: 240,
                  backgroundColor: Colors.white,
                  semanticsLabel: 'Aura Participant ID QR code',
                ),
              ),
              const SizedBox(height: 24),
              Text(
                'Participant ID',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 6),
              SelectableText(
                participantId,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 17,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.8,
                ),
              ),
              const SizedBox(height: 18),
              OutlinedButton.icon(
                onPressed: () async {
                  await Clipboard.setData(ClipboardData(text: participantId));
                  if (!context.mounted) return;
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Participant ID copied.')),
                  );
                },
                icon: const Icon(Icons.copy_rounded),
                label: const Text('Copy ID'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: Theme.of(context).colorScheme.primary,
                  side: BorderSide(
                    color: Theme.of(context).colorScheme.primary,
                  ),
                ),
              ),
              const SizedBox(height: 24),
              const Divider(),
              const SizedBox(height: 20),
              Text(
                'Have a pairing code?',
                style: Theme.of(
                  context,
                ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 8),
              Text(
                'Enter the code from your doctor to link this Participant ID '
                'to the correct clinical record.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  height: 1.45,
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 14),
              FilledButton.icon(
                onPressed: () => _connectWithPairingCode(context),
                icon: const Icon(Icons.link_rounded),
                label: const Text('Enter Pairing Code'),
              ),
              const SizedBox(height: 22),
              Text(
                'Share this only with a healthcare professional involved '
                'in your care or this study.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 12,
                  height: 1.45,
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
