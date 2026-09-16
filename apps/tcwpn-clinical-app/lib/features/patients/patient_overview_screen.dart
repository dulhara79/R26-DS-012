import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../data/api/session.dart';
import '../../data/repositories/central_backend_repositories.dart';
import '../../domain/contracts/assessment_summary.dart';
import '../../domain/contracts/contract_enums.dart';
import '../../state/async_data_state.dart';
import '../../state/patient_overview_controller.dart';
import 'clinical_notes_screen.dart';
import 'data_quality_screen.dart';
import 'signals_contributions_screen.dart';
import 'timeline_screen.dart';

typedef OpenClinicalNotes = void Function(
  String subjectId,
  String localRecordId,
);

class PatientOverviewScreen extends StatefulWidget {
  final String? displayId;
  final String? localRecordId;
  final PatientOverviewController? controller;
  final String? subjectId;
  final ValueChanged<AssessmentSummary>? onOpenSignalsContributions;
  final ValueChanged<AssessmentSummary>? onOpenDataQuality;
  final ValueChanged<String>? onOpenTimeline;
  final OpenClinicalNotes? onOpenClinicalNotes;

  const PatientOverviewScreen({
    super.key,
    required this.controller,
    this.displayId,
    this.localRecordId,
    this.onOpenSignalsContributions,
    this.onOpenDataQuality,
    this.onOpenTimeline,
    this.onOpenClinicalNotes,
  }) : subjectId = null;

  const PatientOverviewScreen.production({
    super.key,
    required this.subjectId,
    this.displayId,
    this.localRecordId,
    this.onOpenSignalsContributions,
    this.onOpenDataQuality,
    this.onOpenTimeline,
    this.onOpenClinicalNotes,
  }) : controller = null;

  @override
  State<PatientOverviewScreen> createState() => _PatientOverviewScreenState();
}

class _PatientOverviewScreenState extends State<PatientOverviewScreen> {
  late final PatientOverviewController _controller;
  late final bool _ownsController;

  @override
  void initState() {
    super.initState();
    _ownsController = widget.controller == null;
    _controller = widget.controller ??
        PatientOverviewController(
          subjectId: widget.subjectId!,
          repository: CentralBackendAssessmentRepository(),
          authRepository: CentralBackendAuthRepository(),
        );
    if (_ownsController) {
      _controller.load();
    }
  }

  @override
  void dispose() {
    if (_ownsController) {
      _controller.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, _) {
        final state = _controller.state;
        return Scaffold(
          appBar: AppBar(
            title: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  widget.displayId ?? _controller.subjectId,
                  style: AppTheme.display(size: 16.5),
                ),
                Text(
                  _controller.subjectId,
                  style: AppTheme.data(size: 10.5, color: Ds.inkFaint),
                ),
              ],
            ),
          ),
          body: _bodyForState(state),
        );
      },
    );
  }

  Widget _bodyForState(AsyncDataState<AssessmentSummary> state) {
    if (state.status == AsyncDataStatus.loading) {
      return const Center(child: CircularProgressIndicator());
    }

    final assessment = state.data;
    if (assessment == null) {
      final title = switch (state.status) {
        AsyncDataStatus.sessionExpired => 'Session expired',
        AsyncDataStatus.forbidden => 'Patient access unavailable',
        AsyncDataStatus.conflict => 'Assessment changed on the server',
        AsyncDataStatus.offline => 'Assessment unavailable offline',
        _ => 'Assessment unavailable',
      };

      return EmptyState(
        icon: state.status == AsyncDataStatus.offline
            ? Icons.cloud_off_rounded
            : Icons.monitor_heart_outlined,
        title: title,
        body: state.message ??
            'No authoritative current assessment is available for this patient.',
        actionLabel: state.status == AsyncDataStatus.sessionExpired ? null : 'Retry',
        onAction: state.status == AsyncDataStatus.sessionExpired
            ? null
            : () => _controller.load(),
      );
    }

    return RefreshIndicator(
      onRefresh: () => _controller.load(showLoading: false),
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, Ds.s10),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (state.status == AsyncDataStatus.offline)
              Padding(
                padding: const EdgeInsets.only(bottom: Ds.s4),
                child: InlineNotice(
                  icon: Icons.cloud_off_rounded,
                  text: state.message ??
                      'Offline. Showing the last server-provided assessment.',
                ),
              ),
            _ForecastSection(forecast: assessment.forecast),
            const SizedBox(height: Ds.s6),
            _CurrentAssessmentSection(assessment: assessment),
            const SizedBox(height: Ds.s6),
            _SignalsSection(
              modalities: assessment.modalities,
              onOpenDetails: () => _openSignalsContributions(assessment),
            ),
            const SizedBox(height: Ds.s6),
            _DataQualitySummary(
              assessment: assessment,
              onOpenDetails: () => _openDataQuality(assessment),
            ),
            const SizedBox(height: Ds.s6),
            const SectionLabel('Temporal context & notes'),
            Panel(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Timeline is historical backend context. Clinical-note history shown in ClinAnx is device-local until a server read contract is verified.',
                    style: TextStyle(
                      fontSize: 12,
                      height: 1.4,
                      color: Ds.inkMuted,
                    ),
                  ),
                  const SizedBox(height: Ds.s2),
                  TextButton.icon(
                    onPressed: _openTimeline,
                    icon: const Icon(Icons.timeline_rounded, size: 16),
                    label: const Text('View timeline'),
                  ),
                  TextButton.icon(
                    onPressed: _openClinicalNotes,
                    icon: const Icon(Icons.note_alt_outlined, size: 16),
                    label: const Text('View clinical notes'),
                  ),
                ],
              ),
            ),
            const SizedBox(height: Ds.s6),
            const DecisionSupportNotice(),
          ],
        ),
      ),
    );
  }

  void _openSignalsContributions(AssessmentSummary assessment) {
    final callback = widget.onOpenSignalsContributions;
    if (callback != null) {
      callback(assessment);
      return;
    }

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => SignalsContributionsScreen(
          assessment: assessment,
          displayId: widget.displayId,
        ),
      ),
    );
  }

  void _openDataQuality(AssessmentSummary assessment) {
    final callback = widget.onOpenDataQuality;
    if (callback != null) {
      callback(assessment);
      return;
    }

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => DataQualityScreen(
          assessment: assessment,
          displayId: widget.displayId,
        ),
      ),
    );
  }

  void _openTimeline() {
    final callback = widget.onOpenTimeline;
    if (callback != null) {
      callback(_controller.subjectId);
      return;
    }

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => TimelineScreen.production(
          subjectId: _controller.subjectId,
          displayId: widget.displayId,
        ),
      ),
    );
  }

  void _openClinicalNotes() {
    final localRecordId =
        widget.localRecordId ?? 'canonical-local::${_controller.subjectId}';
    final callback = widget.onOpenClinicalNotes;
    if (callback != null) {
      callback(_controller.subjectId, localRecordId);
      return;
    }

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => ClinicalNotesScreen.production(
          subjectId: _controller.subjectId,
          localRecordId: localRecordId,
          clinicianId: Session.clinicianId ?? '',
          refreshCanonicalAssessment: () =>
              _controller.load(showLoading: false),
          displayId: widget.displayId,
        ),
      ),
    );
  }
}

class _ForecastSection extends StatelessWidget {
  final ForecastResult? forecast;

  const _ForecastSection({required this.forecast});

  @override
  Widget build(BuildContext context) {
    final value = forecast;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SectionLabel('Acute escalation forecast'),
        Panel(
          child: value == null
              ? const Text(
                  'Forecast unavailable',
                  style: TextStyle(color: Ds.inkMuted),
                )
              : Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      _forecastScopeLabel(value.scope),
                      style: const TextStyle(
                        fontWeight: FontWeight.w700,
                        color: Ds.ink,
                      ),
                    ),
                    const SizedBox(height: Ds.s2),
                    Text(
                      '${_tierLabel(value.tier)} · ${_scoreLabel(value.score)}',
                      style: AppTheme.display(size: 24),
                    ),
                    if (value.horizonMinutes != null) ...[
                      const SizedBox(height: Ds.s2),
                      Text(
                        '${value.horizonMinutes}-minute horizon',
                        style: const TextStyle(color: Ds.inkMuted),
                      ),
                    ],
                    if (value.escalationPredicted != null) ...[
                      const SizedBox(height: Ds.s2),
                      Text(
                        value.escalationPredicted!
                            ? 'Potential escalation predicted within the forecast horizon.'
                            : 'No escalation flag in the current forecast response.',
                        style: const TextStyle(
                          fontSize: 12.5,
                          height: 1.4,
                          color: Ds.inkMuted,
                        ),
                      ),
                    ],
                    if (value.generatedAt != null || value.validUntil != null) ...[
                      const SizedBox(height: Ds.s3),
                      Wrap(
                        spacing: Ds.s4,
                        runSpacing: Ds.s1,
                        children: [
                          if (value.generatedAt != null)
                            Text(
                              'Generated ${_timeLabel(value.generatedAt!)}',
                              style: const TextStyle(
                                fontSize: 11.5,
                                color: Ds.inkFaint,
                              ),
                            ),
                          if (value.validUntil != null)
                            Text(
                              'Valid until ${_timeLabel(value.validUntil!)}',
                              style: const TextStyle(
                                fontSize: 11.5,
                                color: Ds.inkFaint,
                              ),
                            ),
                        ],
                      ),
                    ],
                  ],
                ),
        ),
      ],
    );
  }
}

class _CurrentAssessmentSection extends StatelessWidget {
  final AssessmentSummary assessment;

  const _CurrentAssessmentSection({required this.assessment});

  @override
  Widget build(BuildContext context) {
    final current = assessment.currentAssessment;
    final unavailable = assessment.assessmentStatus == AssessmentStatus.unavailable ||
        current.score == null;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SectionLabel('Current multimodal assessment'),
        Panel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (unavailable)
                const Text(
                  'Assessment unavailable — insufficient current data.',
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    color: Ds.inkMuted,
                  ),
                )
              else
                Text(
                  '${_tierLabel(current.tier)} · ${_scoreLabel(current.score)}',
                  style: AppTheme.display(size: 24),
                ),
              const SizedBox(height: Ds.s2),
              Text(
                _assessmentStatusLabel(assessment.assessmentStatus),
                style: const TextStyle(
                  fontWeight: FontWeight.w600,
                  color: Ds.inkMuted,
                ),
              ),
              if (assessment.confidence != null ||
                  assessment.uncertainty != null) ...[
                const SizedBox(height: Ds.s3),
                Wrap(
                  spacing: Ds.s4,
                  runSpacing: Ds.s1,
                  children: [
                    if (assessment.confidence != null)
                      Text(
                        'Confidence ${assessment.confidence!.toStringAsFixed(2)}',
                        style: const TextStyle(color: Ds.inkMuted),
                      ),
                    if (assessment.uncertainty != null)
                      Text(
                        'Uncertainty ${assessment.uncertainty!.toStringAsFixed(2)}',
                        style: const TextStyle(color: Ds.inkMuted),
                      ),
                  ],
                ),
              ],
              const SizedBox(height: Ds.s3),
              Wrap(
                spacing: Ds.s4,
                runSpacing: Ds.s1,
                children: [
                  if (assessment.fusionResultId != null)
                    Text(
                      'Fusion result #${assessment.fusionResultId}',
                      style: const TextStyle(
                        fontSize: 11.5,
                        color: Ds.inkFaint,
                      ),
                    ),
                  if (assessment.modelVersion != null)
                    Text(
                      'Model ${assessment.modelVersion}',
                      style: const TextStyle(
                        fontSize: 11.5,
                        color: Ds.inkFaint,
                      ),
                    ),
                  if (assessment.computedAt != null)
                    Text(
                      'Last updated ${_timeLabel(assessment.computedAt!)}',
                      style: const TextStyle(
                        fontSize: 11.5,
                        color: Ds.inkFaint,
                      ),
                    ),
                ],
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _SignalsSection extends StatelessWidget {
  static const _orderedIds = [
    'c1_physiological',
    'c2_behavioral',
    'c3_clinical_nlp',
    'c4_demographic',
  ];

  final List<ModalityStatus> modalities;
  final VoidCallback onOpenDetails;

  const _SignalsSection({
    required this.modalities,
    required this.onOpenDetails,
  });

  @override
  Widget build(BuildContext context) {
    final byId = {for (final modality in modalities) modality.componentId: modality};

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SectionLabel('Signals'),
        for (final componentId in _orderedIds)
          Padding(
            padding: const EdgeInsets.only(bottom: Ds.s3),
            child: _SignalCard(
              componentId: componentId,
              modality: byId[componentId],
            ),
          ),
        Align(
          alignment: Alignment.centerLeft,
          child: TextButton.icon(
            onPressed: onOpenDetails,
            icon: const Icon(Icons.arrow_forward_rounded, size: 16),
            label: const Text('View signals & contributions'),
          ),
        ),
      ],
    );
  }
}

class _DataQualitySummary extends StatelessWidget {
  final AssessmentSummary assessment;
  final VoidCallback onOpenDetails;

  const _DataQualitySummary({
    required this.assessment,
    required this.onOpenDetails,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SectionLabel('Data quality'),
        Panel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Assessment data status: ${_assessmentStatusLabel(assessment.assessmentStatus)}',
                style: const TextStyle(
                  fontWeight: FontWeight.w700,
                  color: Ds.ink,
                ),
              ),
              const SizedBox(height: Ds.s1),
              const Text(
                'Server-reported assessment data status. Open the detailed view for per-modality availability and quality fields.',
                style: TextStyle(
                  fontSize: 12,
                  height: 1.4,
                  color: Ds.inkMuted,
                ),
              ),
              const SizedBox(height: Ds.s2),
              TextButton.icon(
                onPressed: onOpenDetails,
                icon: const Icon(Icons.arrow_forward_rounded, size: 16),
                label: const Text('View data quality'),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _SignalCard extends StatelessWidget {
  final String componentId;
  final ModalityStatus? modality;

  const _SignalCard({
    required this.componentId,
    required this.modality,
  });

  @override
  Widget build(BuildContext context) {
    final value = modality;
    return Panel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  _signalLabel(componentId),
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    color: Ds.ink,
                  ),
                ),
              ),
              Text(
                _signalValue(value),
                style: AppTheme.data(
                  size: 14,
                  weight: FontWeight.w600,
                  color: value?.score == null ? Ds.inkFaint : Ds.ink,
                ),
              ),
            ],
          ),
          const SizedBox(height: Ds.s2),
          Text(
            _signalStatus(componentId, value),
            style: TextStyle(
              fontSize: 12,
              color: value?.state == ModalityState.stale ? Ds.amber : Ds.inkMuted,
              fontWeight: value?.state == ModalityState.stale
                  ? FontWeight.w600
                  : FontWeight.w400,
            ),
          ),
          if (value?.capturedAt != null) ...[
            const SizedBox(height: Ds.s1),
            Text(
              'Captured ${_timeLabel(value!.capturedAt!)}',
              style: const TextStyle(fontSize: 11.5, color: Ds.inkFaint),
            ),
          ],
        ],
      ),
    );
  }
}

String _forecastScopeLabel(ForecastScope scope) => switch (scope) {
      ForecastScope.physiological => 'Physiological forecast',
      ForecastScope.multimodal => 'Multimodal forecast',
      ForecastScope.unknown => 'Forecast scope unavailable',
    };

String _tierLabel(RiskTier tier) => switch (tier) {
      RiskTier.low => 'Low',
      RiskTier.medium => 'Medium',
      RiskTier.high => 'High',
      RiskTier.unknown => 'Unknown',
    };

String _assessmentStatusLabel(AssessmentStatus status) => switch (status) {
      AssessmentStatus.complete => 'Complete assessment',
      AssessmentStatus.partial => 'Partial assessment',
      AssessmentStatus.unavailable => 'Assessment unavailable',
      AssessmentStatus.unknown => 'Assessment status unknown',
    };

String _signalLabel(String componentId) => switch (componentId) {
      'c1_physiological' => 'Physiological',
      'c2_behavioral' => 'Behavioural',
      'c3_clinical_nlp' => 'Clinical NLP / TC-WPN',
      'c4_demographic' => 'Contextual',
      _ => componentId,
    };

String _signalValue(ModalityStatus? modality) {
  final score = modality?.score;
  return score == null ? '—' : score.toStringAsFixed(2);
}

String _signalStatus(String componentId, ModalityStatus? modality) {
  if (modality == null) return 'Unavailable';
  if (componentId == 'c2_behavioral' && modality.isExperimentalExcluded) {
    return 'Experimental — not included in fusion';
  }
  return switch (modality.state) {
    ModalityState.ok => modality.includedInFusion == false
        ? 'Available — not included in fusion'
        : 'Available',
    ModalityState.stale => 'Stale',
    ModalityState.notValidated => 'Experimental — not included in fusion',
    ModalityState.unavailable => 'Unavailable',
    ModalityState.error => 'Service error',
    ModalityState.unknown => 'Status unknown',
  };
}

String _scoreLabel(double? score) => score == null ? '—' : score.toStringAsFixed(2);

String _timeLabel(DateTime time) => DateFormat('d MMM y, HH:mm').format(time.toLocal());
