import 'package:flutter/material.dart';
import 'package:mobile_scanner/mobile_scanner.dart';

import 'patient_invitation_qr.dart';

/// Scans patient-issued assignment invitations. A participant-ID-only QR
/// cannot grant access; it is deliberately never returned to the caller.
class ScanPatientInviteScreen extends StatefulWidget {
  const ScanPatientInviteScreen({super.key});

  @override
  State<ScanPatientInviteScreen> createState() => _ScanPatientInviteScreenState();
}

class _ScanPatientInviteScreenState extends State<ScanPatientInviteScreen> {
  final MobileScannerController _controller = MobileScannerController(
    detectionSpeed: DetectionSpeed.noDuplicates,
    formats: const [BarcodeFormat.qrCode],
  );
  bool _handled = false;
  String? _guidance;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _onDetect(BarcodeCapture capture) {
    if (_handled) return;
    for (final barcode in capture.barcodes) {
      final payload = parsePatientInvitationQr(barcode.rawValue);
      if (payload.kind == PatientQrKind.invitation) {
        _handled = true;
        Navigator.of(context).pop(payload.value);
        return;
      }
      if (barcode.rawValue?.trim().isEmpty ?? true) continue;
      final message = payload.kind == PatientQrKind.participantId
          ? 'This QR contains only an Aura Participant ID. It cannot grant access. Ask the patient to choose "Give clinician a code" and enter that invitation manually. Aura must be updated before it can display an invitation QR.'
          : 'This is not a supported clinician invitation QR. Ask the patient for their one-use invitation code.';
      if (mounted && _guidance != message) {
        setState(() => _guidance = message);
      }
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(
      title: const Text('Scan invitation QR'),
      actions: [
        IconButton(
          tooltip: 'Torch',
          icon: const Icon(Icons.flashlight_on_outlined),
          onPressed: () => _controller.toggleTorch(),
        ),
        IconButton(
          tooltip: 'Switch camera',
          icon: const Icon(Icons.cameraswitch_outlined),
          onPressed: () => _controller.switchCamera(),
        ),
      ],
    ),
    body: Stack(
      fit: StackFit.expand,
      children: [
        MobileScanner(
          controller: _controller,
          onDetect: _onDetect,
          errorBuilder: (context, error) => Center(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.no_photography_outlined, size: 40),
                  const SizedBox(height: 12),
                  const Text('Camera unavailable'),
                  const SizedBox(height: 8),
                  Text(
                    error.errorDetails?.message ??
                        'Allow camera access in device settings, or enter the invitation manually.',
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 16),
                  FilledButton(
                    onPressed: () => Navigator.of(context).pop(),
                    child: const Text('Enter code manually'),
                  ),
                ],
              ),
            ),
          ),
        ),
        IgnorePointer(
          child: Center(
            child: Container(
              width: 240,
              height: 240,
              decoration: BoxDecoration(
                border: Border.all(color: Theme.of(context).colorScheme.primary, width: 3),
                borderRadius: BorderRadius.circular(16),
              ),
            ),
          ),
        ),
        Positioned(
          left: 16,
          right: 16,
          bottom: 24,
          child: SafeArea(
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.82),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    _guidance ??
                        'Scan a patient-issued clinician invitation QR, not the Participant ID QR.',
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Colors.white),
                  ),
                  const SizedBox(height: 8),
                  TextButton(
                    onPressed: () => Navigator.of(context).pop(),
                    child: const Text('Enter invitation manually'),
                  ),
                ],
              ),
            ),
          ),
        ),
      ],
    ),
  );
}
