// lib/features/tcwpn/support_set_screen.dart
//
// Few-shot adaptation is TC-WPN's whole proposition, so the screen that curates
// the support set is a first-class destination — not the orphaned tab it became
// in the previous build.
//
// Two scopes:
//   site    — examples that seed every patient at this hospital
//   patient — examples specific to one person, layered on top
//
// Both feed prototype formation. The K counter reflects the union, because that
// is what the model actually receives.

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import 'package:uuid/uuid.dart';

import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../data/local/stores.dart';
import '../../domain/models.dart';
import '../../state/controllers.dart';

enum SupportScope { site, patient }

class SupportSetScreen extends StatefulWidget {
  final SupportScope scope;
  const SupportSetScreen({super.key, required this.scope});

  @override
  State<SupportSetScreen> createState() => _SupportSetScreenState();
}

class _SupportSetScreenState extends State<SupportSetScreen> {
  final _text = TextEditingController();
  String _label = 'anxiety';
  DateTime _noteDate = DateTime.now();

  bool get _isSite => widget.scope == SupportScope.site;

  @override
  void dispose() {
    _text.dispose();
    super.dispose();
  }

  List<SupportNote> _notes(BuildContext context) => _isSite
      ? context.watch<RosterController>().siteSupport
      : context.watch<ChartController>().patientSupport;

  @override
  Widget build(BuildContext context) {
    final notes = _notes(context);
    final anxiety = notes.where((n) => n.isAnxiety).length;
    final control = notes.length - anxiety;

    return Scaffold(
      appBar: AppBar(
        title: Text(_isSite ? 'Site support set' : 'Patient support set'),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, Ds.s10),
        children: [
          Panel(
            background: Ds.brandSoft,
            borderColor: Ds.brandEdge,
            padding: const EdgeInsets.all(Ds.s5),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('HOW ADAPTATION WORKS', style: AppTheme.eyebrow),
                const SizedBox(height: Ds.s2),
                Text(
                  _isSite
                      ? 'These labelled notes seed every patient at this site. The '
                          'model builds one prototype per class from them and '
                          'classifies new notes by distance — no retraining, and no '
                          'gradient update.'
                      : 'These examples apply to this patient only and are layered '
                          'on top of the site set. Use them when a patient\'s '
                          'presentation differs from the site norm.',
                  style: const TextStyle(
                      fontSize: 12.5, color: Ds.inkMuted, height: 1.5),
                ),
                const SizedBox(height: Ds.s4),
                Row(
                  children: [
                    _counter('Anxiety', anxiety, Ds.red),
                    const SizedBox(width: Ds.s4),
                    _counter('Control', control, Ds.green),
                    const Spacer(),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Text('K', style: AppTheme.eyebrow),
                        Text('${notes.length}',
                            style: AppTheme.data(
                                size: 24, weight: FontWeight.w600, color: Ds.brand)),
                      ],
                    ),
                  ],
                ),
                if (notes.isNotEmpty && (anxiety == 0 || control == 0)) ...[
                  const SizedBox(height: Ds.s4),
                  const InlineNotice(
                    icon: Icons.balance_rounded,
                    tone: Ds.amber,
                    text:
                        'Only one class is represented. Prototypical networks need '
                        'both an anxiety and a control prototype — add examples of '
                        'the missing class.',
                  ),
                ],
              ],
            ),
          ),
          const SizedBox(height: Ds.s5),

          SectionLabel('Add a labelled note'),
          Panel(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(child: _labelPick('anxiety', 'Anxiety', Ds.red)),
                    const SizedBox(width: Ds.s3),
                    Expanded(child: _labelPick('control', 'Control', Ds.green)),
                  ],
                ),
                const SizedBox(height: Ds.s3),
                TextField(
                  controller: _text,
                  maxLines: 6,
                  minLines: 4,
                  onChanged: (_) => setState(() {}),
                  decoration: const InputDecoration(
                    labelText: 'Note text',
                    alignLabelWithHint: true,
                    hintText: 'Paste a de-identified clinical note.',
                  ),
                ),
                const SizedBox(height: Ds.s3),
                // The date matters: TC-WPN discounts older notes by
                // exp(-λ·Δt/365). Defaulting to "today" would silently
                // overweight a note written three years ago.
                InkWell(
                  onTap: _pickDate,
                  borderRadius: BorderRadius.circular(Ds.rMd),
                  child: Container(
                    padding: const EdgeInsets.all(Ds.s3),
                    decoration: BoxDecoration(
                      color: Ds.surfaceSunken,
                      borderRadius: BorderRadius.circular(Ds.rMd),
                    ),
                    child: Row(
                      children: [
                        const Icon(Icons.event_outlined,
                            size: 16, color: Ds.inkMuted),
                        const SizedBox(width: Ds.s3),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text('Date the note was written',
                                  style: TextStyle(
                                      fontSize: 12, color: Ds.inkMuted)),
                              Text(DateFormat('d MMMM y').format(_noteDate),
                                  style: AppTheme.data(
                                      size: 12.5, weight: FontWeight.w600)),
                            ],
                          ),
                        ),
                        const Icon(Icons.edit_calendar_outlined,
                            size: 16, color: Ds.inkFaint),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: Ds.s2),
                Text(
                  'Recency weighting discounts older notes, so an accurate date '
                  'changes how much this example counts.',
                  style: const TextStyle(fontSize: 11, color: Ds.inkFaint, height: 1.4),
                ),
                const SizedBox(height: Ds.s4),
                ElevatedButton.icon(
                  onPressed: _text.text.trim().isEmpty ? null : _add,
                  icon: const Icon(Icons.add_rounded, size: 19),
                  label: const Text('Add to support set'),
                ),
              ],
            ),
          ),
          const SizedBox(height: Ds.s5),

          SectionLabel('Current examples'),
          if (notes.isEmpty)
            const Panel(
              child: Text(
                'No labelled examples yet. Until at least one of each class is '
                'added, the model runs on its meta-trained prototypes.',
                style: TextStyle(fontSize: 12.5, color: Ds.inkMuted, height: 1.5),
              ),
            )
          else
            ...notes.map(_noteRow),
        ],
      ),
    );
  }

  Widget _counter(String label, int n, Color tone) => Row(
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(color: tone, shape: BoxShape.circle),
          ),
          const SizedBox(width: 6),
          Text('$label ',
              style: const TextStyle(fontSize: 12, color: Ds.inkMuted)),
          Text('$n', style: AppTheme.data(size: 13, weight: FontWeight.w600)),
        ],
      );

  Widget _labelPick(String value, String label, Color tone) {
    final sel = _label == value;
    return GestureDetector(
      onTap: () => setState(() => _label = value),
      child: AnimatedContainer(
        duration: Ds.fast,
        padding: const EdgeInsets.symmetric(vertical: Ds.s3),
        decoration: BoxDecoration(
          color: sel ? tone.withValues(alpha: 0.10) : Ds.surface,
          borderRadius: BorderRadius.circular(Ds.rMd),
          border: Border.all(
              color: sel ? tone : Ds.hairlineStrong, width: sel ? 1.5 : 1),
        ),
        alignment: Alignment.center,
        child: Text(label,
            style: TextStyle(
                fontSize: 13.5,
                fontWeight: FontWeight.w600,
                color: sel ? tone : Ds.inkMuted)),
      ),
    );
  }

  Widget _noteRow(SupportNote n) => Padding(
        padding: const EdgeInsets.only(bottom: Ds.s3),
        child: Panel(
          padding: const EdgeInsets.all(Ds.s4),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 7,
                height: 7,
                margin: const EdgeInsets.only(top: 5),
                decoration: BoxDecoration(
                  color: n.isAnxiety ? Ds.red : Ds.green,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: Ds.s3),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text(n.isAnxiety ? 'Anxiety' : 'Control',
                            style: TextStyle(
                                fontSize: 11,
                                fontWeight: FontWeight.w700,
                                color: n.isAnxiety ? Ds.red : Ds.green)),
                        const SizedBox(width: Ds.s2),
                        Text(DateFormat('MMM y').format(n.noteDate),
                            style: AppTheme.data(size: 10, color: Ds.inkFaint)),
                      ],
                    ),
                    const SizedBox(height: 3),
                    Text(
                      n.text,
                      maxLines: 3,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                          fontSize: 12.5, color: Ds.inkMuted, height: 1.45),
                    ),
                  ],
                ),
              ),
              IconButton(
                icon: const Icon(Icons.delete_outline_rounded,
                    size: 18, color: Ds.inkFaint),
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(),
                onPressed: () => _remove(n),
              ),
            ],
          ),
        ),
      );

  Future<void> _pickDate() async {
    final d = await showDatePicker(
      context: context,
      initialDate: _noteDate,
      firstDate: DateTime(2010),
      lastDate: DateTime.now(),
    );
    if (d != null) setState(() => _noteDate = d);
  }

  Future<void> _add() async {
    final clinician = await SecureStore.clinicianId() ?? 'unknown';
    if (!mounted) return;
    final note = SupportNote(
      id: const Uuid().v4(),
      text: _text.text.trim(),
      label: _label,
      noteDate: _noteDate,
      addedAt: DateTime.now(),
      patientMrn:
          _isSite ? null : context.read<ChartController>().mrn,
      addedByClinician: clinician,
    );

    if (_isSite) {
      await context.read<RosterController>().addSiteSupport(note);
    } else {
      await context.read<ChartController>().addSupport(note);
    }
    _text.clear();
    setState(() => _noteDate = DateTime.now());
  }

  Future<void> _remove(SupportNote n) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Remove this example?'),
        content: const Text(
          'Future predictions will no longer use it to form prototypes.',
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel')),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Remove', style: TextStyle(color: Ds.red)),
          ),
        ],
      ),
    );
    if (ok != true || !mounted) return;
    if (_isSite) {
      await context.read<RosterController>().removeSiteSupport(n.id);
    } else {
      await context.read<ChartController>().removeSupport(n.id);
    }
  }
}
