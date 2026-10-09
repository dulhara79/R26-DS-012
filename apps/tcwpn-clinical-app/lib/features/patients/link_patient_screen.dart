import 'package:flutter/material.dart';

import '../../data/api/api_client.dart';
import '../../data/repositories/assignment_invite_repository.dart';
import '../../domain/repositories/assignment_invite_repository.dart';

/// Accepts patient consent through a server-issued, single-use invite code.
/// The response never becomes a local roster entry; callers reload the roster.
class LinkPatientScreen extends StatefulWidget {
  final AssignmentInviteRepository? repository;

  const LinkPatientScreen({super.key, this.repository});

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

  Future<void> _redeem() async {
    if (_form.currentState?.validate() != true || _busy) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final result = await _repository.redeem(_code.text);
      if (mounted) Navigator.of(context).pop(result.subjectId);
    } on ApiException catch (error) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _error = switch (error.statusCode) {
          401 =>
            'Your clinician session has expired. Sign out and sign in again.',
          403 => 'Your account cannot accept this invitation.',
          404 => 'Invitation not found. Ask the patient for a fresh code.',
          409 => 'This invitation has already been used or the patient record changed.',
          410 =>
            'This invitation has expired. Ask the patient for a fresh code.',
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
            'Ask the patient to open “Connect to Doctor” in Aura and choose “Give clinician a code”. Enter that one-use invitation exactly as shown. The patient ID alone cannot grant access.',
          ),
          const SizedBox(height: 20),
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
              validator: (value) => value == null || value.trim().isEmpty
                  ? 'Enter the invitation code.'
                  : null,
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
