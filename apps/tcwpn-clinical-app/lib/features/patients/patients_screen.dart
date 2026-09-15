import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';

import '../../core/design/components.dart';
import '../../core/design/theme.dart';
import '../../core/design/tokens.dart';
import '../../domain/models.dart';
import '../../state/controllers.dart';
import '../chart/patient_chart_screen.dart';
import 'scan_patient_id_screen.dart';

class PatientsScreen extends StatefulWidget {
  const PatientsScreen({super.key});

  @override
  State<PatientsScreen> createState() => _PatientsScreenState();
}

class _PatientsScreenState extends State<PatientsScreen> {
  final _search = TextEditingController();
  String _query = '';
  AlertBand? _band;

  @override
  void dispose() {
    _search.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final roster = context.watch<RosterController>();
    final results = roster.search(_query, band: _band);

    return Scaffold(
      appBar: AppBar(title: const Text('Patients')),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => _addPatient(context),
        backgroundColor: Ds.brand,
        foregroundColor: Colors.white,
        icon: const Icon(Icons.person_add_alt_1_rounded, size: 20),
        label: const Text('Add patient'),
      ),
      body: Column(
        children: [
          Container(
            color: Ds.surface,
            padding: const EdgeInsets.fromLTRB(Ds.s4, 0, Ds.s4, Ds.s3),
            child: Column(
              children: [
                TextField(
                  controller: _search,
                  onChanged: (v) => setState(() => _query = v),
                  decoration: InputDecoration(
                    hintText: 'Search by name or MRN',
                    prefixIcon: const Icon(Icons.search_rounded, size: 19),
                    suffixIcon: _query.isEmpty
                        ? null
                        : IconButton(
                            icon: const Icon(Icons.close_rounded, size: 17),
                            onPressed: () {
                              _search.clear();
                              setState(() => _query = '');
                            },
                          ),
                  ),
                ),
                const SizedBox(height: Ds.s3),
                SizedBox(
                  height: 32,
                  child: ListView(
                    scrollDirection: Axis.horizontal,
                    children: [
                      _filter('All', _band == null, null,
                          () => setState(() => _band = null)),
                      for (final b in AlertBandX.scored.reversed)
                        _filter(
                          b.protocolName,
                          _band == b,
                          b.fg,
                          () => setState(() => _band = _band == b ? null : b),
                        ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const Divider(height: 1),
          Expanded(
            child: results.isEmpty
                ? EmptyState(
                    icon: Icons.person_search_rounded,
                    title: roster.patients.isEmpty
                        ? 'No patients yet'
                        : 'No matching patients',
                    body: roster.patients.isEmpty
                        ? 'Add a patient to open their chart and run the first clinical-note analysis.'
                        : 'Try a different name, MRN, or alert band.',
                    actionLabel: roster.patients.isEmpty ? 'Add patient' : null,
                    onAction: () => _addPatient(context),
                  )
                : ListView.separated(
                    padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s4, Ds.s4, 96),
                    itemCount: results.length,
                    separatorBuilder: (_, __) => const SizedBox(height: Ds.s3),
                    itemBuilder: (_, i) => PatientRow(
                      patient: results[i],
                      fusion: roster.fusionFor(results[i].mrn),
                      onTap: () => openChart(context, results[i]),
                    ),
                  ),
          ),
        ],
      ),
    );
  }

  Widget _filter(
    String label,
    bool selected,
    Color? tone,
    VoidCallback onTap,
  ) =>
      Padding(
        padding: const EdgeInsets.only(right: Ds.s2),
        child: GestureDetector(
          onTap: onTap,
          child: AnimatedContainer(
            duration: Ds.fast,
            padding: const EdgeInsets.symmetric(horizontal: Ds.s4, vertical: 7),
            decoration: BoxDecoration(
              color: selected ? (tone ?? Ds.brand) : Ds.surface,
              borderRadius: BorderRadius.circular(Ds.rPill),
              border: Border.all(
                color: selected ? Colors.transparent : Ds.hairlineStrong,
              ),
            ),
            child: Text(
              label,
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: selected ? Colors.white : Ds.inkMuted,
              ),
            ),
          ),
        ),
      );

  Future<void> _addPatient(BuildContext context) async {
    final roster = context.read<RosterController>();
    final p = await showModalBottomSheet<Patient>(
      context: context,
      isScrollControlled: true,
      builder: (_) => const AddPatientSheet(),
    );
    if (p == null || !context.mounted) return;
    final err = await roster.addPatient(p);
    if (err != null && context.mounted) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(err)));
    }
  }
}

void openChart(BuildContext context, Patient p) => Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => PatientChartScreen(patient: p)),
    );

class PatientRow extends StatelessWidget {
  final Patient patient;
  final FusionResult? fusion;
  final VoidCallback onTap;

  const PatientRow({
    super.key,
    required this.patient,
    required this.fusion,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final band = (fusion != null && fusion!.hasComposite) ? fusion!.band : null;
    return Panel(
      onTap: onTap,
      padding: const EdgeInsets.all(Ds.s4),
      borderColor: (band == AlertBand.red || band == AlertBand.darkRed)
          ? band!.fg.withValues(alpha: 0.35)
          : Ds.hairline,
      child: Column(
        children: [
          Row(
            children: [
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: band?.bg ?? Ds.brandSoft,
                  borderRadius: BorderRadius.circular(Ds.rMd),
                ),
                alignment: Alignment.center,
                child: Text(
                  patient.initials,
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: band?.fg ?? Ds.brand,
                  ),
                ),
              ),
              const SizedBox(width: Ds.s3),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      patient.name,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: 14.5,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 1),
                    Text(
                      '${patient.mrn} · ${patient.age}y · ${patient.ward}',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: AppTheme.data(size: 11, color: Ds.inkFaint),
                    ),
                  ],
                ),
              ),
              if (band != null) BandChip(band: band),
            ],
          ),
          if (fusion != null) ...[
            const SizedBox(height: Ds.s3),
            FusionBar(
              compact: true,
              composite: fusion!.compositeScore,
              band: fusion!.band,
              segments: [
                for (final c in fusion!.contributions)
                  FusionSegment(
                    key: c.key,
                    label: c.label,
                    contribution: c.contribution,
                    weight: c.weight,
                    score: c.score,
                    excluded: c.excluded,
                    unavailableReason: c.note,
                    color: FusionBar.palette[c.key] ?? Ds.brand,
                  ),
              ],
            ),
            const SizedBox(height: Ds.s2),
            Row(
              children: [
                Text(
                  fusion!.compositeLabel,
                  style: AppTheme.data(size: 11.5, weight: FontWeight.w600),
                ),
                const SizedBox(width: Ds.s2),
                Expanded(
                  child: Text(
                    '${fusion!.assessmentLabel} · ${fusion!.modalitiesUsed} of 3 modalities',
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(fontSize: 11, color: Ds.inkFaint),
                  ),
                ),
                if (fusion!.updatedAt != null)
                  Text(
                    DateFormat('d MMM, HH:mm').format(fusion!.updatedAt!),
                    style: const TextStyle(fontSize: 11, color: Ds.inkFaint),
                  ),
              ],
            ),
          ],
        ],
      ),
    );
  }
}

class AddPatientSheet extends StatefulWidget {
  const AddPatientSheet({super.key});

  @override
  State<AddPatientSheet> createState() => _AddPatientSheetState();
}

class _AddPatientSheetState extends State<AddPatientSheet> {
  final _form = GlobalKey<FormState>();
  final _mrn = TextEditingController();
  final _name = TextEditingController();
  final _age = TextEditingController(text: '24');

  static const genders = ['Female', 'Male', 'Non-binary', 'Prefer not to say'];
  static const wards = [
    'Psychiatry OPD',
    'Ward 04 (Female)',
    'Ward 05 (Male)',
    'Emergency',
    'Community clinic',
  ];

  String _gender = 'Prefer not to say';
  String _ward = 'Psychiatry OPD';
  String _marital = 'Never';
  int _education = 3;
  double _income = 2.5;

  @override
  void dispose() {
    _mrn.dispose();
    _name.dispose();
    _age.dispose();
    super.dispose();
  }

  Future<void> _scanParticipantId() async {
    final participantId = await Navigator.of(context).push<String>(
      MaterialPageRoute(builder: (_) => const ScanPatientIdScreen()),
    );
    if (participantId == null || !mounted) return;
    _mrn.text = participantId;
  }

  @override
  Widget build(BuildContext context) => Padding(
        padding: EdgeInsets.only(
          left: Ds.s5,
          right: Ds.s5,
          top: Ds.s5,
          bottom: MediaQuery.of(context).viewInsets.bottom + Ds.s6,
        ),
        child: Form(
          key: _form,
          child: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Center(
                  child: Container(
                    width: 36,
                    height: 4,
                    decoration: BoxDecoration(
                      color: Ds.hairlineStrong,
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                ),
                const SizedBox(height: Ds.s5),
                Text('Add patient', style: AppTheme.display(size: 20)),
                const SizedBox(height: Ds.s1),
                const Text(
                  "Scan the QR shown in the patient's Aura app. This copies their Participant ID so all four components use the same record.",
                  style: TextStyle(
                    fontSize: 12.5,
                    color: Ds.inkMuted,
                    height: 1.45,
                  ),
                ),
                const SizedBox(height: Ds.s4),
                OutlinedButton.icon(
                  onPressed: _scanParticipantId,
                  icon: const Icon(Icons.qr_code_scanner_rounded),
                  label: const Text('Scan Aura QR'),
                ),
                const SizedBox(height: Ds.s3),
                TextFormField(
                  controller: _mrn,
                  textCapitalization: TextCapitalization.characters,
                  autocorrect: false,
                  decoration: const InputDecoration(
                    labelText: 'Aura Participant ID',
                    hintText: 'P_7F3A9C2E4B10D6C1',
                  ),
                  validator: (v) {
                    final id = (v ?? '').trim().toUpperCase();
                    if (id.isEmpty) return 'Participant ID is required';
                    if (!RegExp(r'^P_[A-F0-9]{16}$').hasMatch(id)) {
                      return 'Scan or enter a valid Aura Participant ID';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: Ds.s3),
                TextFormField(
                  controller: _name,
                  textCapitalization: TextCapitalization.words,
                  decoration: const InputDecoration(labelText: 'Full name'),
                  validator: (v) =>
                      (v ?? '').trim().isEmpty ? 'Name is required' : null,
                ),
                const SizedBox(height: Ds.s3),
                Row(
                  children: [
                    SizedBox(
                      width: 96,
                      child: TextFormField(
                        controller: _age,
                        keyboardType: TextInputType.number,
                        decoration: const InputDecoration(labelText: 'Age'),
                        validator: (v) {
                          final n = int.tryParse(v ?? '');
                          if (n == null) return 'Number';
                          if (n < 12 || n > 120) return 'Range';
                          return null;
                        },
                      ),
                    ),
                    const SizedBox(width: Ds.s3),
                    Expanded(
                      child: DropdownButtonFormField<String>(
                        initialValue: _gender,
                        isExpanded: true,
                        decoration: const InputDecoration(labelText: 'Gender'),
                        items: genders
                            .map((g) => DropdownMenuItem(value: g, child: Text(g)))
                            .toList(),
                        onChanged: (v) => setState(() => _gender = v!),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: Ds.s3),
                DropdownButtonFormField<String>(
                  initialValue: _ward,
                  isExpanded: true,
                  decoration: const InputDecoration(labelText: 'Ward'),
                  items: wards
                      .map((w) => DropdownMenuItem(value: w, child: Text(w)))
                      .toList(),
                  onChanged: (v) => setState(() => _ward = v!),
                ),
                const SizedBox(height: Ds.s5),
                Text('INTERVENTION MODEL INPUTS', style: AppTheme.eyebrow),
                const SizedBox(height: Ds.s1),
                const Text(
                  "Used only by the intervention engine's feature vector.",
                  style: TextStyle(fontSize: 11.5, color: Ds.inkFaint),
                ),
                const SizedBox(height: Ds.s3),
                DropdownButtonFormField<String>(
                  initialValue: _marital,
                  isExpanded: true,
                  decoration: const InputDecoration(labelText: 'Marital status'),
                  items: const ['Never', 'Married', 'Separated']
                      .map((m) => DropdownMenuItem(value: m, child: Text(m)))
                      .toList(),
                  onChanged: (v) => setState(() => _marital = v!),
                ),
                const SizedBox(height: Ds.s3),
                _slider(
                  'Education level',
                  _education.toDouble(),
                  1,
                  5,
                  4,
                  '$_education',
                  (v) => setState(() => _education = v.round()),
                ),
                _slider(
                  'Income (PIR)',
                  _income,
                  0,
                  5,
                  20,
                  _income.toStringAsFixed(1),
                  (v) => setState(() => _income = v),
                ),
                const SizedBox(height: Ds.s5),
                ElevatedButton(
                  onPressed: () {
                    if (!_form.currentState!.validate()) return;
                    Navigator.pop(
                      context,
                      Patient(
                        mrn: _mrn.text.trim().toUpperCase(),
                        name: _name.text.trim(),
                        age: int.parse(_age.text.trim()),
                        gender: _gender,
                        ward: _ward,
                        referredOn: DateTime.now(),
                        maritalStatus: _marital,
                        educationLevel: _education,
                        incomePir: _income,
                      ),
                    );
                  },
                  child: const Text('Add patient'),
                ),
              ],
            ),
          ),
        ),
      );

  Widget _slider(
    String label,
    double value,
    double min,
    double max,
    int divisions,
    String readout,
    ValueChanged<double> onChanged,
  ) =>
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                label,
                style: const TextStyle(
                  fontSize: 12.5,
                  fontWeight: FontWeight.w600,
                  color: Ds.inkMuted,
                ),
              ),
              const Spacer(),
              Text(readout, style: AppTheme.data(size: 12)),
            ],
          ),
          Slider(
            value: value,
            min: min,
            max: max,
            divisions: divisions,
            activeColor: Ds.brand,
            inactiveColor: Ds.surfaceSunken,
            onChanged: onChanged,
          ),
        ],
      );
}
