import 'package:flutter/material.dart';
import 'package:mobile_scanner/mobile_scanner.dart';

import '../../core/design/tokens.dart';

class ScanPatientIdScreen extends StatefulWidget {
  const ScanPatientIdScreen({super.key});

  @override
  State<ScanPatientIdScreen> createState() => _ScanPatientIdScreenState();
}

class _ScanPatientIdScreenState extends State<ScanPatientIdScreen> {
  static final RegExp _participantIdPattern =
      RegExp(r'^P_[A-F0-9]{16}$');

  bool _handled = false;
  String? _message;

  String? _participantIdFrom(String? rawValue) {
    final raw = rawValue?.trim();
    if (raw == null || raw.isEmpty) return null;

    final directId = raw.toUpperCase();
    if (_participantIdPattern.hasMatch(directId)) return directId;

    final uri = Uri.tryParse(raw);
    if (uri == null ||
        uri.scheme.toLowerCase() != 'clinanx' ||
        uri.host.toLowerCase() != 'patient' ||
        uri.pathSegments.length != 1) {
      return null;
    }

    final participantId = uri.pathSegments.first.toUpperCase();
    return _participantIdPattern.hasMatch(participantId)
        ? participantId
        : null;
  }

  void _handleCapture(BarcodeCapture capture) {
    if (_handled) return;

    for (final barcode in capture.barcodes) {
      final participantId = _participantIdFrom(barcode.rawValue);
      if (participantId == null) continue;

      _handled = true;
      Navigator.pop(context, participantId);
      return;
    }

    if (mounted && _message == null) {
      setState(() => _message = 'This is not an Aura Patient ID QR code.');
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        backgroundColor: Colors.black,
        appBar: AppBar(
          title: const Text('Scan Aura QR'),
          backgroundColor: Colors.black,
          foregroundColor: Colors.white,
        ),
        body: Stack(
          fit: StackFit.expand,
          children: [
            MobileScanner(onDetect: _handleCapture),
            Center(
              child: IgnorePointer(
                child: Container(
                  width: 260,
                  height: 260,
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.white, width: 3),
                    borderRadius: BorderRadius.circular(Ds.rLg),
                  ),
                ),
              ),
            ),
            Positioned(
              left: Ds.s5,
              right: Ds.s5,
              bottom: Ds.s8,
              child: Container(
                padding: const EdgeInsets.all(Ds.s4),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.72),
                  borderRadius: BorderRadius.circular(Ds.rMd),
                ),
                child: Text(
                  _message ??
                      'Point the camera at the QR shown in the patient\'s '
                          'Aura app.',
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 13,
                    height: 1.45,
                  ),
                ),
              ),
            ),
          ],
        ),
      );
}
