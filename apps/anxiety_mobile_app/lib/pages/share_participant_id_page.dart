import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:qr_flutter/qr_flutter.dart';

import '../services/api_service.dart';
import '../services/participant_identity_service.dart';
import '../services/patient_session_service.dart';

class ShareParticipantIdPage extends StatelessWidget {
  final String participantId;

  const ShareParticipantIdPage({super.key, required this.participantId});

  String get _qrData => participantId;

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

    final expectedSubject =
        await ParticipantIdentityService.getCentralSubjectId();
    final subjectId = await ApiService.selfEnrol(
      participantId,
      pairingCode: pairingCode,
      expectedSubjectId: expectedSubject,
    );
    if (subjectId != null) {
      await ParticipantIdentityService.saveCentralSubjectId(subjectId);
    }

    if (!context.mounted) return;
    messenger.hideCurrentSnackBar();
    messenger.showSnackBar(
      SnackBar(
        content: Text(
          subjectId != null
              ? 'Aura is now connected to your doctor.'
              : 'Could not confirm this code. Ask your doctor for a fresh pairing code.',
        ),
      ),
    );
  }

  Future<void> _inviteClinician(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    var session = await PatientSessionService.instance.currentSession();
    if (session == null) {
      final expected = await ParticipantIdentityService.getCentralSubjectId();
      final subject = await ApiService.selfEnrol(
        participantId,
        expectedSubjectId: expected,
      );
      if (subject != null) {
        await ParticipantIdentityService.saveCentralSubjectId(subject);
        session = await PatientSessionService.instance.currentSession();
      }
    }
    if (session == null) {
      messenger.showSnackBar(
        const SnackBar(
          content: Text(
            'Patient session unavailable. If your doctor enrolled first, enter their pairing code.',
          ),
        ),
      );
      return;
    }
    final invite = await ApiService.createAssignmentInvite();
    if (!context.mounted) return;
    if (invite == null) {
      messenger.showSnackBar(
        const SnackBar(
          content: Text(
            'Could not create an invitation. Check your connection and try again.',
          ),
        ),
      );
      return;
    }
    await showDialog<void>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Clinician invitation'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Give this one-use code to your doctor. It does not contain your name or readings.',
            ),
            const SizedBox(height: 12),
            SelectableText(invite.code),
            const SizedBox(height: 8),
            Text(
              'Expires at ${invite.expiresAt.toLocal().hour.toString().padLeft(2, '0')}:${invite.expiresAt.toLocal().minute.toString().padLeft(2, '0')}',
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () async {
              await Clipboard.setData(ClipboardData(text: invite.code));
              if (dialogContext.mounted) Navigator.pop(dialogContext);
            },
            child: const Text('Copy code'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(dialogContext),
            child: const Text('Close'),
          ),
        ],
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
                'Share your participant ID',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 10),
              Text(
                'This QR contains only your Aura Participant ID. It does not give a clinician access to your record. '
                'To link a clinician, use “Give clinician a code” below. It does not contain your name, readings, or diagnosis.',
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
              OutlinedButton.icon(
                onPressed: () => _inviteClinician(context),
                icon: const Icon(Icons.person_add_alt_1_outlined),
                label: const Text('Give clinician a code'),
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
