// lib/features/tcwpn/note_analysis_screen.dart
//
// Component 4 entry point: submit a clinical note for TC-WPN analysis.
//
// Two things this screen refuses to hide:
//   1. How many labelled examples the prototype will actually be built from.
//      A K of 0 means the model is running on meta-trained prototypes with no
//      site adaptation, and the clinician should know that before they read a
//      score.
//   2. That the note is saved before the network call. Analysis failing must
//      never cost a clinician their typing.

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/config/env.dart';
import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../data/api/api_client.dart';
import '../../data/local/stores.dart';
import '../../domain/models.dart';
import '../../state/controllers.dart';
import 'tcwpn_result_screen.dart';

class NoteAnalysisScreen extends StatefulWidget {
  /// The note being edited. Null for a new note.
  ///
  /// When this is set the screen UPDATES that note. It does not create a second
  /// one — which is what the previous build did on every edit, leaving the
  /// original behind as an orphan the clinician could not reach.
  final ClinicalNote? existing;

  const NoteAnalysisScreen({super.key, this.existing});

  @override
  State<NoteAnalysisScreen> createState() => _NoteAnalysisScreenState();
}

class _NoteAnalysisScreenState extends State<NoteAnalysisScreen> {
  late final TextEditingController _note;
  late String _type;
  late final String _originalText;
  late final String _originalType;
  bool _busy = false;

  static const _types = [
    'Psychiatry note',
    'Discharge summary',
    'Nursing note',
    'Social work note',
    'Physician note',
  ];

  bool get _isEditing => widget.existing != null;

  bool get _dirty =>
      _note.text.trim() != _originalText.trim() || _type != _originalType;

  @override
  void initState() {
    super.initState();
    final e = widget.existing;
    _originalText = e?.text ?? '';
    _originalType = e?.noteType ?? 'Psychiatry note';
    _note = TextEditingController(text: _originalText);
    _type = _originalType;
  }

  @override
  void dispose() {
    _note.dispose();
    super.dispose();
  }

  int get _words => _note.text.trim().isEmpty
      ? 0
      : _note.text.trim().split(RegExp(r'\s+')).length;

  @override
  Widget build(BuildContext context) {
    final chart = context.watch<ChartController>();
    final k = chart.effectiveSupportCount;

    return WorkingOverlay(
      active: _busy,
      message: 'Running TC-WPN.\nA cold model can take up to a minute to wake.',
      child: Scaffold(
        appBar: AppBar(
          leading: IconButton(
            icon: const Icon(Icons.close_rounded),
            onPressed: () => Navigator.pop(context),
          ),
          title:
              Text(_isEditing ? 'Edit clinical note' : 'Analyse clinical note'),
        ),
        body: ListView(
          padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, Ds.s10),
          children: [
            _SupportReadiness(k: k),
            const SizedBox(height: Ds.s4),
            DropdownButtonFormField<String>(
              initialValue: _type,
              isExpanded: true,
              decoration: const InputDecoration(labelText: 'Note type'),
              items: _types
                  .map((t) => DropdownMenuItem(value: t, child: Text(t)))
                  .toList(),
              onChanged: (v) => setState(() => _type = v!),
            ),
            const SizedBox(height: Ds.s3),
            TextField(
              controller: _note,
              maxLines: 12,
              minLines: 8,
              textCapitalization: TextCapitalization.sentences,
              onChanged: (_) => setState(() {}),
              style: const TextStyle(fontSize: 14, height: 1.55),
              decoration: const InputDecoration(
                labelText: 'Clinical note',
                alignLabelWithHint: true,
                hintText:
                    'History of present illness, mental state examination, '
                    'assessment, and current medications.',
              ),
            ),
            const SizedBox(height: Ds.s2),
            Row(
              children: [
                Text('$_words words',
                    style: AppTheme.data(size: 11, color: Ds.inkFaint)),
                const Spacer(),
                if (_words > 0 && _words < 25)
                  const Text('Short notes weaken the prediction',
                      style: TextStyle(fontSize: 11, color: Ds.amber)),
              ],
            ),
            if (Env.demoData) ...[
              const SizedBox(height: Ds.s5),
              Text('EXAMPLE NOTES', style: AppTheme.eyebrow),
              const SizedBox(height: Ds.s2),
              Wrap(
                spacing: Ds.s2,
                runSpacing: Ds.s2,
                children: _examples.entries
                    .map((e) => OutlinedButton(
                          onPressed: () {
                            _note.text = e.value;
                            setState(() {});
                          },
                          style: OutlinedButton.styleFrom(
                            minimumSize: Size.zero,
                            padding: const EdgeInsets.symmetric(
                                horizontal: Ds.s4, vertical: Ds.s2),
                            textStyle: const TextStyle(fontSize: 12),
                          ),
                          child: Text(e.key),
                        ))
                    .toList(),
              ),
            ],
            const SizedBox(height: Ds.s5),
            const InlineNotice(
              icon: Icons.privacy_tip_outlined,
              tone: Ds.amber,
              text:
                  'Remove direct identifiers before submitting. Note text leaves '
                  'the device and is processed by the model service.',
            ),
            const SizedBox(height: Ds.s5),
            ElevatedButton.icon(
              onPressed: (_words == 0 || _busy) ? null : () => _analyse(chart),
              icon: const Icon(Icons.analytics_rounded, size: 19),
              label: Text(widget.existing?.hasBeenAnalysed == true
                  ? 'Save and re-analyse'
                  : 'Analyse note'),
            ),
            const SizedBox(height: Ds.s3),
            OutlinedButton.icon(
              onPressed: (_words == 0 || _busy) ? null : () => _save(chart),
              icon: const Icon(Icons.save_outlined, size: 19),
              label: Text(
                  _isEditing ? 'Save changes' : 'Save as draft, analyse later'),
            ),
            if (_isEditing) ...[
              const SizedBox(height: Ds.s3),
              TextButton.icon(
                onPressed: _busy ? null : () => _delete(chart),
                icon: const Icon(Icons.delete_outline_rounded, size: 19),
                style: TextButton.styleFrom(foregroundColor: Ds.red),
                label: const Text('Delete note'),
              ),
            ],
          ],
        ),
      ),
    );
  }

  /// Writes the editor's contents to storage and returns the stored note.
  /// Creates on first save, updates in place thereafter. No network call.
  Future<ClinicalNote?> _persist(ChartController chart) async {
    final text = _note.text.trim();
    final existing = widget.existing;
    if (existing == null) {
      final clinician = await SecureStore.clinicianId() ?? 'unknown';
      return chart.saveDraft(
        text: text,
        noteType: _type,
        clinicianId: clinician,
      );
    }
    if (!_dirty) return existing;
    return chart.updateNote(existing.id, text: text, noteType: _type);
  }

  Future<void> _save(ChartController chart) async {
    setState(() => _busy = true);
    final saved = await _persist(chart);
    if (!mounted) return;
    setState(() => _busy = false);
    if (saved == null) return;
    Navigator.pop(context);
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(_isEditing
          ? 'Changes saved.'
          : 'Saved as a draft. Analyse it from the note history.'),
    ));
  }

  Future<void> _analyse(ChartController chart) async {
    setState(() => _busy = true);

    // Save FIRST, always. Everything after this point can fail without costing
    // the clinician a word of what they typed.
    final saved = await _persist(chart);
    if (!mounted || saved == null) {
      if (mounted) setState(() => _busy = false);
      return;
    }

    try {
      final note = await chart.analyseStoredNote(saved.id);
      if (!mounted) return;
      setState(() => _busy = false);
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => ChangeNotifierProvider.value(
            value: chart,
            child: TcwpnResultScreen(note: note),
          ),
        ),
      );
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() => _busy = false);
      await _showAnalysisUnavailable(chart, saved, e);
    }
  }

  /// Separates the two facts a clinician needs to keep apart: the note is safe,
  /// and the analysis did not run. A snackbar could not carry both, and the old
  /// one led with the failure.
  Future<void> _showAnalysisUnavailable(
    ChartController chart,
    ClinicalNote saved,
    ApiException e,
  ) async {
    if (!mounted) return;
    final action = await showDialog<String>(
      context: context,
      barrierDismissible: false,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Analysis unavailable'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Your note has been saved. It is safe on this device and nothing '
              'you wrote has been lost.',
              style: TextStyle(fontSize: 13.5, height: 1.5),
            ),
            const SizedBox(height: Ds.s3),
            Text(
              e.message,
              style: const TextStyle(
                  fontSize: 12.5, color: Ds.inkMuted, height: 1.45),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext, 'close'),
            child: const Text('Close'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(dialogContext, 'edit'),
            child: const Text('Edit note'),
          ),
          if (e.isRetryable)
            FilledButton(
              onPressed: () => Navigator.pop(dialogContext, 'retry'),
              child: const Text('Try again'),
            ),
        ],
      ),
    );

    if (!mounted) return;
    switch (action) {
      case 'retry':
        await _analyse(chart);
      case 'close':
        Navigator.pop(context);
      default:
        break; // 'edit' — stay on the editor with the text still in it.
    }
  }

  Future<void> _delete(ChartController chart) async {
    final existing = widget.existing;
    if (existing == null) return;

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: Text(existing.hasBeenAnalysed
            ? 'Delete this note?'
            : 'Delete this draft?'),
        content: const Text(
          'This action cannot be undone.',
          style: TextStyle(fontSize: 13.5, height: 1.5),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext, false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(dialogContext, true),
            style: TextButton.styleFrom(foregroundColor: Ds.red),
            child: const Text('Delete'),
          ),
        ],
      ),
    );

    if (confirmed != true || !mounted) return;
    setState(() => _busy = true);
    final removed = await chart.deleteNote(existing.id);
    if (!mounted) return;
    setState(() => _busy = false);
    Navigator.pop(context);
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content:
          Text(removed ? 'Note deleted.' : 'That note was already removed.'),
    ));
  }

  static const _examples = {
    'GAD, active': 'Patient presents with persistent and excessive worry about work, health, '
        'and finances over the past eight months. Reports difficulty controlling '
        'the worry, present most days. Associated fatigue, impaired concentration, '
        'irritability, muscle tension, and disturbed sleep. GAD-7 14, PHQ-9 16. '
        'Currently on sertraline 100mg daily, referred for CBT. Impression: '
        'generalised anxiety disorder.',
    'Panic disorder':
        'Recurrent unexpected panic attacks over six months, characterised by '
            'palpitations, chest tightness, sweating, tremor, and intense fear lasting '
            'ten to twenty minutes. Persistent anticipatory worry with avoidance of '
            'public transport. Commenced escitalopram 10mg.',
    'Stable follow-up':
        'Stable on sertraline 100mg. Marked reduction in anxiety symptoms. GAD-7 '
            'improved from 16 to 8. Sleep and concentration improved. Continuing CBT. '
            'No adverse effects. Plan: continue current management, review in six weeks.',
  };
}

/// Shows exactly what the prototype will be built from before the clinician
/// commits to an analysis.
class _SupportReadiness extends StatelessWidget {
  final int k;
  const _SupportReadiness({required this.k});

  @override
  Widget build(BuildContext context) {
    // The proposal targets K = 10–20 for site adaptation.
    final (tone, headline, detail) = switch (k) {
      0 => (
          Ds.amber,
          'No labelled examples',
          'The model will use its meta-trained prototypes with no adaptation to '
              'this site. Add labelled notes to the support set for site-specific '
              'performance.'
        ),
      < 10 => (
          Ds.amber,
          '$k of 10 labelled examples',
          'Below the adaptation target. Predictions are usable but less reliable '
              'than at K = 10 or above.'
        ),
      _ => (
          Ds.green,
          '$k labelled examples',
          'Prototypes will be formed from these, weighted by recency and by the '
              'model\'s confidence in each one.'
        ),
    };

    return Container(
      padding: const EdgeInsets.all(Ds.s4),
      decoration: BoxDecoration(
        color: tone.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(Ds.rMd),
        border: Border.all(color: tone.withValues(alpha: 0.2)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            k >= 10
                ? Icons.check_circle_outline_rounded
                : Icons.info_outline_rounded,
            size: 17,
            color: tone,
          ),
          const SizedBox(width: Ds.s3),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(headline,
                    style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        color: tone)),
                const SizedBox(height: 2),
                Text(detail,
                    style: const TextStyle(
                        fontSize: 12, color: Ds.inkMuted, height: 1.45)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
