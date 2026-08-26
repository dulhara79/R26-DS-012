// lib/features/patients/scan_patient_id_screen.dart
//
// RESTORED. This path previously held a byte-for-byte copy of
// lib/state/controllers.dart — its first line still read
// `// lib/state/controllers.dart`. Two consequences followed:
//
//   • shell.dart imports BOTH this file and ../state/controllers.dart, and both
//     declared RosterController and ChartController, so every reference to
//     RosterController in shell.dart was an `ambiguous_import` error.
//   • The real ScanPatientIdScreen widget did not exist anywhere, so
//     shell.dart:655 `const ScanPatientIdScreen()` was `creation_with_non_type`.
//
// This is the widget that route always meant to reach.
//
// WHAT IT DOES
// ------------
// Reads the QR the patient's Aura app displays and pops the decoded string back
// to the caller. The caller (_AddPatientSheet in shell.dart) drops it into the
// MRN field, which validates the same pattern again — so this screen is a
// convenience, never the only line of defence.
//
// WHY THE PATTERN CHECK IS HERE AND NOT ONLY IN THE FORM
// -----------------------------------------------------
// A camera pointed at a ward will find barcodes: wristbands, drug packaging,
// equipment asset tags. Accepting the first thing it decodes would silently
// enrol a patient under a medication's GTIN. Only `P_` + 16 uppercase hex is
// accepted — the same shape as C2_TEST_SUBJECT=P_65DC4002E7863773 in the
// backend's env.example — and anything else is reported as a wrong code rather
// than returned.

import 'package:flutter/material.dart';
import 'package:mobile_scanner/mobile_scanner.dart';

import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';

/// Returns the scanned Aura Participant ID, or null if the clinician backed out.
class ScanPatientIdScreen extends StatefulWidget {
  const ScanPatientIdScreen({super.key});

  @override
  State<ScanPatientIdScreen> createState() => _ScanPatientIdScreenState();
}

class _ScanPatientIdScreenState extends State<ScanPatientIdScreen> {
  /// Same shape the roster form validates and the same shape
  /// RosterController._participantIdPattern registers as the C2 alias. Keep the
  /// three in step: a value that scans but does not validate is a worse
  /// experience than one that never scanned.
  static final RegExp _participantId = RegExp(r'^P_[A-F0-9]{16}$');

  final MobileScannerController _controller = MobileScannerController(
    detectionSpeed: DetectionSpeed.noDuplicates,
    formats: const [BarcodeFormat.qrCode],
  );

  /// Guards against the callback firing again while the pop is in flight.
  /// onDetect can deliver several frames before the route is actually gone, and
  /// popping twice takes the caller's screen with it.
  bool _handled = false;

  String? _rejected;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _onDetect(BarcodeCapture capture) {
    if (_handled) return;

    for (final barcode in capture.barcodes) {
      final value = (barcode.rawValue ?? '').trim().toUpperCase();
      if (value.isEmpty) continue;

      if (_participantId.hasMatch(value)) {
        _handled = true;
        Navigator.of(context).pop(value);
        return;
      }

      // Decoded something, but not an Aura ID. Say so — a scanner that silently
      // ignores a code the clinician can plainly see is worse than one that
      // explains itself.
      if (mounted && _rejected != value) {
        setState(() => _rejected = value);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        elevation: 0,
        title: Text(
          'Scan Aura QR',
          style: AppTheme.display(size: 18).copyWith(color: Colors.white),
        ),
        actions: [
          IconButton(
            tooltip: 'Torch',
            icon: const Icon(Icons.flashlight_on_rounded),
            onPressed: () => _controller.toggleTorch(),
          ),
          IconButton(
            tooltip: 'Switch camera',
            icon: const Icon(Icons.cameraswitch_rounded),
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
            errorBuilder: (context, error) => _CameraUnavailable(
              message: error.errorDetails?.message ??
                  'The camera could not be started.',
              onCancel: () => Navigator.of(context).pop(),
            ),
          ),

          // Reticle. Purely a framing aid — decoding is full-frame, so a code
          // slightly outside the box still reads.
          IgnorePointer(
            child: Center(
              child: Container(
                width: 240,
                height: 240,
                decoration: BoxDecoration(
                  border: Border.all(color: Ds.brandEdge, width: 2),
                  borderRadius: BorderRadius.circular(Ds.s3),
                ),
              ),
            ),
          ),

          Positioned(
            left: Ds.s5,
            right: Ds.s5,
            bottom: Ds.s6,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: Ds.s4,
                    vertical: Ds.s3,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.62),
                    borderRadius: BorderRadius.circular(Ds.s3),
                  ),
                  child: Text(
                    _rejected == null
                        ? 'Point the camera at the QR code in the patient\u2019s '
                            'Aura app.'
                        : 'That code is not an Aura Participant ID.\n'
                            'Expected P_ followed by 16 characters.',
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 13,
                      height: 1.45,
                    ),
                  ),
                ),
                const SizedBox(height: Ds.s3),
                TextButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: const Text(
                    'Enter the ID by hand instead',
                    style: TextStyle(color: Colors.white),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// Shown when the camera cannot start — permission denied, no camera, or the
/// platform channel failing. The caller keeps its manual-entry field, so this
/// is a dead end for the scanner, not for adding a patient.
class _CameraUnavailable extends StatelessWidget {
  final String message;
  final VoidCallback onCancel;

  const _CameraUnavailable({required this.message, required this.onCancel});

  @override
  Widget build(BuildContext context) => ColoredBox(
        color: Ds.canvas,
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(Ds.s6),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.no_photography_rounded,
                    size: 40, color: Ds.inkFaint),
                const SizedBox(height: Ds.s4),
                Text(
                  'Camera unavailable',
                  style: AppTheme.display(size: 17),
                ),
                const SizedBox(height: Ds.s2),
                Text(
                  message,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 12.5,
                    color: Ds.inkMuted,
                    height: 1.45,
                  ),
                ),
                const SizedBox(height: Ds.s5),
                FilledButton(
                  onPressed: onCancel,
                  child: const Text('Enter the ID by hand'),
                ),
              ],
            ),
          ),
        ),
      );
}
