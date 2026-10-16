import 'package:flutter/material.dart';

import '../../data/api/api_client.dart';
import '../../data/repositories/assignment_invite_repository.dart';
import '../../domain/repositories/assignment_invite_repository.dart';
import 'patient_invitation_qr.dart';
import 'scan_patient_invite_screen.dart';

typedef InvitationQrScanner = Future<String?> Function(BuildContext context);

/// Accepts patient consent through a server-issued, single-use invite code.
/// Both manual entry and QR scanning use the same authenticated backend API.
/// The response never becomes a local roster entry; callers reload the roster.
class LinkPatientScreen extends StatefulWidget {
  final AssignmentInviteRepository? repository;
  final InvitationQrScanner? scanInvitation;

  const LinkPatientScreen({
    super.key,
    this.repository,
    this.scanInvitation,
  });

  @override
  State<LinkPatientScreen> createState() => _LinkPatientScreenState();
}

class _LinkPatientScreenState extends State<LinkPatientScreen> {
  final _form = GlobalKey<FormState>();
  final _code = TextEditingController();
  late final AssignmentInviteRepository _repository =
      widget.repository ?? CentralBackendAssignmentInviteRepository();
  bool _busy = false;
  String? _error;

  @override
  void dispose() {
    _code.dispose();
    super.dispose();
  }

  Future<void> _scan() async {
    if (_busy) return;
    try {
      final scanner = widget.scanInvitation;
      final raw = scanner != null
          ? await scanner(context)
          : await Navigator.of(context).push<String>(
              MaterialPageRoute(
                builder: (_) => const ScanPatientInviteScreen(),
              ),
            );
      if (!mounted || raw == null) return;
      final scanned = parsePatientInvitationQr(raw);
      if (scanned.kind != PatientQrKind.invitation) {
        setState(() {
          _error = scanned.kind == PatientQrKind.participantId
              ? 'The scanned QR is a Participant ID, not a clinician invitation. Ask the patient to choose "Give clinician a code" in Aura.'
              : 'The QR is not a valid clinician invitation. Enter the code supplied by the patient.';
        });
        return;
      }
      setState(() {
        _code.text = scanned.value!;
        _error = null;
      });
    } catch (_) {
      if (mounted) {
        setState(() {
          _error = 'Unable to open the scanner. Enter the invitation code manually.';
        });
      }
    }
  }

  String? _validateCode(String? input) {
    final value = input?.trim() ?? '';
    if (value.isEmpty) return 'Enter the invitation code.';
    if (parsePatientInvitationQr(value).kind == PatientQrKind.participantId) {
      return 'A Participant ID cannot authorize linking. Ask for a clinician invitation code.';
    }
    if (value.startsWith('clinanx://') &&
        parsePatientInvitationQr(value).kind != PatientQrKind.invitation) {
      return 'This QR link is not a clinician invitation.';
    }
    if (value.startsWith('https://') || value.startsWith('http://')) {
      return 'Enter a patient-issued invitation code, not a web link.';
    }
    return null;
  }

  Future<void> _redeem() async {
    if (_form.currentState?.validate() != true || _busy) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    final value = _code.text.trim();
    final parsed = parsePatientInvitationQr(value);
    final invite = parsed.kind == PatientQrKind.invitation ? parsed.value! : value;
    try {
      final result = await _repository.redeem(invite);
      if (mounted) Navigator.of(context).pop(result.subjectId);
    } on ApiException catch (error) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _error = switch (error.statusCode) {
          401 => 'Your clinician session has expired. Sign out and sign in again.',
          403 => 'Your account cannot accept this invitation.',
          404 => 'Invitation not found. Ask the patient for a fresh code.',
          409 => 'This invitation has already been used or the patient record changed.',
          410 => 'This invitation has expired. Ask the patient for a fresh code.',
          422 => 'The server rejected this invitation code.',
          _ => error.message,
        };
      });
    } catch (_) {
      if (mounted) {
        setState(() {
          _busy = false;
          _error = 'Could not link this patient. Check the code and connection, then try again.';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('Link patient')),
    body: SafeArea(
      child: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          const Text(
            'Ask the patient to open “Connect to Doctor” in Aura and choose “Give clinician a code”. '
            'The patient ID alone cannot grant access.',
          ),
          const SizedBox(height: 20),
          OutlinedButton.icon(
            onPressed: _busy ? null : _scan,
            icon: const Icon(Icons.qr_code_scanner_outlined),
            label: const Text('Scan invitation QR'),
          ),
          const SizedBox(height: 8),
          Text(
            'Aura currently displays a Participant ID QR, not an invitation QR. '
            'Use the manual code below until Aura supports invitation QR sharing.',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 20),
          const Row(
            children: [
              Expanded(child: Divider()),
              Padding(
                padding: EdgeInsets.symmetric(horizontal: 12),
                child: Text('OR'),
              ),
              Expanded(child: Divider()),
            ],
          ),
          const SizedBox(height: 16),
          Form(
            key: _form,
            child: TextFormField(
              controller: _code,
              enabled: !_busy,
              maxLength: 128,
              autocorrect: false,
              textCapitalization: TextCapitalization.none,
              decoration: const InputDecoration(
                labelText: 'Patient invitation code',
              ),
              validator: _validateCode,
            ),
          ),
          if (_error != null) ...[
            const SizedBox(height: 8),
            Text(
              _error!,
              style: TextStyle(color: Theme.of(context).colorScheme.error),
            ),
          ],
          const SizedBox(height: 16),
          FilledButton(
            onPressed: _busy ? null : _redeem,
            child: _busy
                ? const CircularProgressIndicator()
                : const Text('Accept invitation'),
          ),
        ],
      ),
    ),
  );
}
