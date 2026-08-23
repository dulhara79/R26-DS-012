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
import '../../state/controllers.dart';
import 'tcwpn_result_screen.dart';

class NoteAnalysisScreen extends StatefulWidget {
  const NoteAnalysisScreen({super.key});
  @override
  State<NoteAnalysisScreen> createState() => _NoteAnalysisScreenState();
}

class _NoteAnalysisScreenState extends State<NoteAnalysisScreen> {
  final _note = TextEditingController();
  String _type = 'Psychiatry note';
  bool _busy = false;

  static const _types = [
    'Psychiatry note',
    'Discharge summary',
    'Nursing note',
    'Social work note',
    'Physician note',
  ];

  @override
  void dispose() {
    _note.dispose();
    super.dispose();
  }

  int get _words =>
      _note.text.trim().isEmpty ? 0 : _note.text.trim().split(RegExp(r'\s+')).length;

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
          title: const Text('Analyse clinical note'),
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
              onPressed: (_words == 0 || _busy) ? null : () => _run(chart),
              icon: const Icon(Icons.analytics_rounded, size: 19),
              label: const Text('Analyse note'),
            ),
            const SizedBox(height: Ds.s3),
            OutlinedButton.icon(
              onPressed:
                  (_words == 0 || _busy) ? null : () => _run(chart, draft: true),
              icon: const Icon(Icons.save_outlined, size: 19),
              label: const Text('Save as draft, analyse later'),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _run(ChartController chart, {bool draft = false}) async {
    setState(() => _busy = true);
    final clinician = await SecureStore.clinicianId() ?? 'unknown';
    try {
      final note = await chart.analyseNote(
        text: _note.text.trim(),
        noteType: _type,
        clinicianId: clinician,
        skipAnalysis: draft,
      );
      if (!mounted) return;
      if (draft) {
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text('Saved as a draft. Analyse it from the note history.'),
        ));
        return;
      }
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
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text(e.message),
        duration: const Duration(seconds: 8),
        action: SnackBarAction(
          label: 'Keep draft',
          textColor: Colors.white,
          onPressed: () => Navigator.pop(context),
        ),
      ));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  static const _examples = {
    'GAD, active':
        'Patient presents with persistent and excessive worry about work, health, '
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
            k >= 10 ? Icons.check_circle_outline_rounded : Icons.info_outline_rounded,
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
                        fontSize: 13, fontWeight: FontWeight.w600, color: tone)),
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
