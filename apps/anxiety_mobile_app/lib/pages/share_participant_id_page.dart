import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:qr_flutter/qr_flutter.dart';

import '../theme/app_theme.dart';

class ShareParticipantIdPage extends StatelessWidget {
  final String participantId;

  const ShareParticipantIdPage({
    super.key,
    required this.participantId,
  });

  String get _qrData => 'clinanx://patient/$participantId';

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
                  await Clipboard.setData(
                    ClipboardData(text: participantId),
                  );
                  if (!context.mounted) return;
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Participant ID copied.')),
                  );
                },
                icon: const Icon(Icons.copy_rounded),
                label: const Text('Copy ID'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppTheme.kPrimaryDeep,
                  side: const BorderSide(color: AppTheme.kPrimaryDeep),
                ),
              ),
              const SizedBox(height: 18),
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
