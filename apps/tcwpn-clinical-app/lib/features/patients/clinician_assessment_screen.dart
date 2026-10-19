import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../../core/design/components.dart';
import '../../core/design/tokens.dart';
import '../../data/repositories/p5c_repositories.dart';
import '../../domain/contracts/assessment_summary.dart';
import '../../domain/contracts/contract_enums.dart';
import '../../state/clinician_assessment_controller.dart';

class ClinicianAssessmentScreen extends StatefulWidget {
  final ClinicianAssessmentController? controller;
  final AssessmentSummary? assessment;
  final String? clinicianId;
  final String? displayId;

  const ClinicianAssessmentScreen({
    super.key,
    required this.controller,
    this.displayId,
  })  : assessment = null,
        clinicianId = null;

  const ClinicianAssessmentScreen.production({
    super.key,
    required this.assessment,
    required this.clinicianId,
    this.displayId,
  }) : controller = null;

  @override
  State<ClinicianAssessmentScreen> createState() =>
      _ClinicianAssessmentScreenState();
}

class _ClinicianAssessmentScreenState
    extends State<ClinicianAssessmentScreen> {
  late final ClinicianAssessmentController _controller;
  CentralBackendClinicianAssessmentRepository? _ownedRepository;
  final _note = TextEditingController();

  @override
  void initState() {
    super.initState();
    if (widget.controller != null) {
      _controller = widget.controller!;
    } else {
      final repository = CentralBackendClinicianAssessmentRepository();
      _ownedRepository = repository;
      _controller = ClinicianAssessmentController(
        assessment: widget.assessment!,
        clinicianId: widget.clinicianId ?? '',
        repository: repository,
      );
    }
    _note.text = _controller.note;
  }

  @override
  void dispose() {
    _note.dispose();
    if (widget.controller == null) {
      _controller.dispose();
      _ownedRepository?.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, _) => Scaffold(
        appBar: AppBar(title: const Text('Clinician Assessment')),
        body: ListView(
          padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, Ds.s10),
          children: [
            if ((widget.displayId ?? '').isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(bottom: Ds.s3),
                child: Text(
                  widget.displayId!,
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 16,
                  ),
                ),
              ),
            _ModelAssessmentPanel(assessment: _controller.assessment),
            const SizedBox(height: Ds.s5),
            if (_controller.submitted)
              _ReceiptPanel(controller: _controller)
            else
              _AssessmentForm(controller: _controller, noteController: _note),
          ],
        ),
      ),
    );
  }
}

class _ModelAssessmentPanel extends StatelessWidget {
  final AssessmentSummary assessment;

  const _ModelAssessmentPanel({required this.assessment});

  @override
  Widget build(BuildContext context) {
    final current = assessment.currentAssessment;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SectionLabel('Current model assessment'),
        Panel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                '${_tierLabel(current.tier)} · ${current.score?.toStringAsFixed(2) ?? '—'}',
                style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: Ds.s2),
              Text(
                assessment.fusionResultId == null
                    ? 'Fusion result: Not reported'
                    : 'Fusion result #${assessment.fusionResultId}',
                style: const TextStyle(color: Ds.inkMuted),
              ),
              if ((assessment.modelVersion ?? '').isNotEmpty)
                Text(
                  'Model ${assessment.modelVersion}',
                  style: const TextStyle(color: Ds.inkMuted),
                ),
              if (assessment.computedAt != null)
                Text(
                  'Computed ${DateFormat.yMMMd().add_jm().format(assessment.computedAt!)}',
                  style: const TextStyle(color: Ds.inkMuted),
                ),
              const SizedBox(height: Ds.s3),
              const Text(
                'Model output is read-only here. Your clinician assessment is stored separately from the model and does not replace the fusion result.',
                style: TextStyle(fontSize: 12.5, height: 1.4, color: Ds.inkMuted),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _AssessmentForm extends StatelessWidget {
  final ClinicianAssessmentController controller;
  final TextEditingController noteController;

  const _AssessmentForm({
    required this.controller,
    required this.noteController,
  });

  @override
  Widget build(BuildContext context) {
    final noFusion = controller.assessment.fusionResultId == null;
    final noClinician = controller.clinicianId.trim().isEmpty;
    final controlsDisabled = controller.submitting || noFusion || noClinician;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SectionLabel('Your clinician assessment'),
        if (noFusion)
          const Padding(
            padding: EdgeInsets.only(bottom: Ds.s3),
            child: Panel(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Clinician assessment cannot be recorded',
                    style: TextStyle(fontWeight: FontWeight.w700),
                  ),
                  SizedBox(height: Ds.s1),
                  Text(
                    'No authoritative fusion-result identifier is available.',
                    style: TextStyle(color: Ds.inkMuted),
                  ),
                ],
              ),
            ),
          ),
        if (noClinician)
          const Padding(
            padding: EdgeInsets.only(bottom: Ds.s3),
            child: Panel(
              child: Text(
                'Clinician identity is unavailable. Sign in again before recording an auditable assessment.',
                style: TextStyle(color: Ds.inkMuted),
              ),
            ),
          ),
        Panel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Choose your judgement before reviewing any calibration result.',
                style: TextStyle(fontSize: 12.5, color: Ds.inkMuted),
              ),
              const SizedBox(height: Ds.s2),
              RadioGroup<String>(
                groupValue: controller.selectedTier,
                onChanged: (value) {
                  if (!controlsDisabled && value != null) {
                    controller.selectTier(value);
                  }
                },
                child: Column(
                  children: [
                    for (final tier in const ['Low', 'Medium', 'High'])
                      Material(
                        type: MaterialType.transparency,
                        child: RadioListTile<String>(
                          contentPadding: EdgeInsets.zero,
                          dense: true,
                          title: Text(tier),
                          value: tier,
                          enabled: !controlsDisabled,
                        ),
                      ),
                  ],
                ),
              ),
              const SizedBox(height: Ds.s2),
              TextField(
                controller: noteController,
                enabled: !controller.submitting,
                maxLength: 255,
                minLines: 2,
                maxLines: 5,
                onChanged: controller.setNote,
                decoration: const InputDecoration(
                  labelText: 'Observation (optional)',
                  hintText: 'Brief verdict note or clinical observation',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: Ds.s2),
              Text(
                noClinician
                    ? 'Clinician: Not available'
                    : 'Clinician: ${controller.clinicianId.trim()}',
                style: const TextStyle(fontSize: 12, color: Ds.inkMuted),
              ),
              if (controller.error != null) ...[
                const SizedBox(height: Ds.s2),
                Text(
                  controller.error!,
                  style: const TextStyle(color: Ds.red),
                ),
              ],
              const SizedBox(height: Ds.s3),
              SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: controller.canSubmit ? controller.submit : null,
                  child: Text(
                    controller.submitting
                        ? 'Saving…'
                        : 'Save clinician assessment',
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _ReceiptPanel extends StatelessWidget {
  final ClinicianAssessmentController controller;

  const _ReceiptPanel({required this.controller});

  @override
  Widget build(BuildContext context) {
    final receipt = controller.receipt!;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SectionLabel('Server confirmation'),
        Panel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Clinician assessment recorded',
                style: TextStyle(fontWeight: FontWeight.w700, fontSize: 17),
              ),
              const SizedBox(height: Ds.s3),
              Text('Your assessment: ${receipt.tierLabel}'),
              Text('Fusion result: #${receipt.fusionResultId}'),
              if ((receipt.author ?? '').isNotEmpty)
                Text('Clinician: ${receipt.author}'),
              if ((receipt.note ?? '').isNotEmpty) ...[
                const SizedBox(height: Ds.s2),
                Text(receipt.note!),
              ],
              if (receipt.verdictId != null)
                Text('Server verdict ID: #${receipt.verdictId}'),
              if (receipt.agreesWithModel != null)
                Text(
                  'Model comparison: ${receipt.agreesWithModel! ? 'Same tier' : 'Different tier'}',
                ),
              if (receipt.calibrationLabelsTotal != null)
                Text(
                  'Calibration labels recorded: ${receipt.calibrationLabelsTotal}',
                  style: const TextStyle(fontSize: 12, color: Ds.inkMuted),
                ),
              const SizedBox(height: Ds.s3),
              const Text(
                'This receipt confirms the verdict POST only. ClinAnx does not invent a server assessment history or submission timestamp.',
                style: TextStyle(fontSize: 12, height: 1.4, color: Ds.inkMuted),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

String _tierLabel(RiskTier tier) => switch (tier) {
      RiskTier.low => 'Low',
      RiskTier.medium => 'Medium',
      RiskTier.high => 'High',
      RiskTier.unknown => 'Unknown',
    };
