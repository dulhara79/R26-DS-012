// lib/features/shell.dart
//
// ONE shell. ONE bottom navigation bar. The two research components are never
// destinations — they are sections inside a patient's chart, so a clinician
// experiences one continuous application rather than two apps sharing an icon.
//
//   Caseload  — what needs attention today, across all four modalities
//   Patients  — the roster
//   Alerts    — the escalation queue with acknowledgement
//   Settings  — model configuration, session, governance

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';

import '../core/design/components.dart';
import '../core/design/theme.dart';
import '../core/design/tokens.dart';
import '../domain/models.dart';
import '../state/controllers.dart';
import 'alerts/alerts_screen.dart';
import 'chart/patient_chart_screen.dart';
import 'patients/scan_patient_id_screen.dart';
import 'settings/settings_screen.dart';
import 'tcwpn/support_set_screen.dart';

class AppShell extends StatefulWidget {
  const AppShell({super.key});
  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _tab = 0;

  @override
  Widget build(BuildContext context) {
    final roster = context.watch<RosterController>();

    return Scaffold(
      body: IndexedStack(
        index: _tab,
        children: const [
          _CaseloadTab(),
          _PatientsTab(),
          AlertsScreen(),
          SettingsScreen(),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: (i) => setState(() => _tab = i),
        destinations: [
          const NavigationDestination(
            icon: Icon(Icons.monitor_heart_outlined),
            selectedIcon: Icon(Icons.monitor_heart_rounded),
            label: 'Caseload',
          ),
          const NavigationDestination(
            icon: Icon(Icons.folder_shared_outlined),
            selectedIcon: Icon(Icons.folder_shared_rounded),
            label: 'Patients',
          ),
          NavigationDestination(
            icon: Badge(
              isLabelVisible: roster.unacknowledgedCount > 0,
              label: Text('${roster.unacknowledgedCount}'),
              backgroundColor: Ds.red,
              child: const Icon(Icons.notifications_none_rounded),
            ),
            selectedIcon: const Icon(Icons.notifications_rounded),
            label: 'Alerts',
          ),
          const NavigationDestination(
            icon: Icon(Icons.tune_outlined),
            selectedIcon: Icon(Icons.tune_rounded),
            label: 'Settings',
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Caseload — the opening thesis of the app
// ─────────────────────────────────────────────────────────────────────────────

class _CaseloadTab extends StatelessWidget {
  const _CaseloadTab();

  @override
  Widget build(BuildContext context) {
    final roster = context.watch<RosterController>();
    final review = roster.needingReview;
    final bandCounts = <AlertBand, int>{};
    for (final p in roster.patients) {
      final f = roster.fusionFor(p.mrn);
      // A GREY result is an assessment the gate BLOCKED. It has no composite,
      // so it belongs with the unscored patients rather than in a severity bar.
      if (f == null || !f.hasComposite) continue;
      bandCounts[f.band] = (bandCounts[f.band] ?? 0) + 1;
    }
    final unscored = roster.patients.length -
        bandCounts.values.fold<int>(0, (a, b) => a + b);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Caseload'),
        actions: [
          IconButton(
            tooltip: 'Support set',
            icon: const Icon(Icons.dataset_outlined),
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(
                  builder: (_) =>
                      const SupportSetScreen(scope: SupportScope.site)),
            ),
          ),
        ],
      ),
      body: roster.loading
          ? const Center(child: CircularProgressIndicator(color: Ds.brand))
          : ListView(
              padding: const EdgeInsets.fromLTRB(Ds.s4, Ds.s3, Ds.s4, Ds.s10),
              children: [
                _bandStrip(bandCounts, unscored, roster.patients.length),
                const SizedBox(height: Ds.s6),
                SectionLabel('Needs review'),
                if (review.isEmpty)
                  Panel(
                    child: Row(
                      children: [
                        const Icon(Icons.check_circle_outline_rounded,
                            size: 18, color: Ds.green),
                        const SizedBox(width: Ds.s3),
                        Expanded(
                          child: Text(
                            roster.patients.isEmpty
                                ? 'No patients on this device yet.'
                                : 'No patient is currently in the RED or DARK RED band.',
                            style: const TextStyle(
                                fontSize: 13, color: Ds.inkMuted),
                          ),
                        ),
                      ],
                    ),
                  )
                else
                  ...review.map((p) => Padding(
                        padding: const EdgeInsets.only(bottom: Ds.s3),
                        child: PatientRow(
                          patient: p,
                          fusion: roster.fusionFor(p.mrn),
                          onTap: () => openChart(context, p),
                        ),
                      )),
                const SizedBox(height: Ds.s6),
                SectionLabel('Framework'),
                const _FrameworkPanel(),
                const SizedBox(height: Ds.s4),
                const DecisionSupportNotice(),
              ],
            ),
    );
  }

  /// Distribution across the four alert bands. The proportions are the story,
  /// so it's drawn as one segmented rule rather than four separate counters.
  Widget _bandStrip(Map<AlertBand, int> counts, int unscored, int total) {
    if (total == 0) {
      return const Panel(
        child: Text(
          'Add a patient to begin. Each patient chart brings together clinical '
          'notes, passive sensing, and intervention history under one composite '
          'risk score.',
          style: TextStyle(fontSize: 13, color: Ds.inkMuted, height: 1.5),
        ),
      );
    }
    return Panel(
      padding: const EdgeInsets.all(Ds.s5),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('ACTIVE PATIENTS', style: AppTheme.eyebrow),
                  const SizedBox(height: 2),
                  Text('$total',
                      style: AppTheme.data(size: 34, weight: FontWeight.w600)),
                ],
              ),
              const Spacer(),
              if (unscored > 0)
                Text('$unscored not yet scored',
                    style: const TextStyle(fontSize: 11.5, color: Ds.inkFaint)),
            ],
          ),
          const SizedBox(height: Ds.s4),
          SizedBox(
            height: 10,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(Ds.rSm),
              child: Row(
                children: [
                  for (final b in AlertBandX.scored)
                    if ((counts[b] ?? 0) > 0)
                      Expanded(
                        flex: counts[b]!,
                        child: Container(color: b.fg),
                      ),
                  if (unscored > 0)
                    Expanded(
                        flex: unscored,
                        child: Container(color: Ds.surfaceSunken)),
                ],
              ),
            ),
          ),
          const SizedBox(height: Ds.s4),
          Wrap(
            spacing: Ds.s5,
            runSpacing: Ds.s3,
            children: [
              for (final b in AlertBandX.scored)
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                        width: 8,
                        height: 8,
                        decoration:
                            BoxDecoration(color: b.fg, shape: BoxShape.circle)),
                    const SizedBox(width: 6),
                    Text('${b.protocolName} ',
                        style: const TextStyle(
                            fontSize: 11.5, color: Ds.inkMuted)),
                    Text('${counts[b] ?? 0}',
                        style:
                            AppTheme.data(size: 12, weight: FontWeight.w600)),
                  ],
                ),
            ],
          ),
        ],
      ),
    );
  }
}

class _FrameworkPanel extends StatelessWidget {
  const _FrameworkPanel();

  // Wire keys are the backend's; labels carry the paper's component numbering.
  // NO WEIGHTS. The weights are derived server-side from each component's
  // validation AUROC above chance and are renormalised per assessment over the
  // modalities that actually reported, so no fixed number printed here would be
  // true of any particular patient. The real weights are shown in the fusion
  // breakdown, where they arrive from the server alongside the score they
  // multiplied.
  static const _rows = [
    (
      'c1_physiological',
      'Physiological · Component 1',
      'Wearable ECG, HRV, respiration, temperature'
    ),
    (
      'c2_behavioral',
      'Behavioural · Component 2',
      'Passive smartphone sensing. Recorded, and excluded from the composite '
          'by pre-registered rule.'
    ),
    (
      'c3_clinical_nlp',
      'Clinical notes · Component 4',
      'TC-WPN few-shot analysis of written notes'
    ),
    (
      'c4_demographic',
      'Demographic prior · Component 4 contextual',
      'DCAR intake demographics and GAD-7'
    ),
  ];

  @override
  Widget build(BuildContext context) => Panel(
        padding: const EdgeInsets.all(Ds.s5),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Composite risk is computed by the Central Backend, which weights '
              'each modality by its validation performance above chance, decays '
              'it by age, and renormalises across the modalities that actually '
              'reported. At least two usable modalities are required, one of '
              'them time-varying, or no composite is produced.',
              style: TextStyle(fontSize: 12.5, color: Ds.inkMuted, height: 1.5),
            ),
            const SizedBox(height: Ds.s4),
            for (final r in _rows)
              Padding(
                padding: const EdgeInsets.only(bottom: Ds.s3),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      width: 3,
                      height: 30,
                      decoration: BoxDecoration(
                        color: FusionBar.palette[r.$1],
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                    const SizedBox(width: Ds.s3),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(r.$2,
                              style: const TextStyle(
                                  fontSize: 13, fontWeight: FontWeight.w600)),
                          Text(r.$3,
                              style: const TextStyle(
                                  fontSize: 11.5,
                                  color: Ds.inkFaint,
                                  height: 1.3)),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      );
}

// ─────────────────────────────────────────────────────────────────────────────
// Patients
// ─────────────────────────────────────────────────────────────────────────────

class _PatientsTab extends StatefulWidget {
  const _PatientsTab();
  @override
  State<_PatientsTab> createState() => _PatientsTabState();
}

class _PatientsTabState extends State<_PatientsTab> {
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
                            () =>
                                setState(() => _band = _band == b ? null : b)),
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
                        ? 'Add a patient to open their chart and run the first '
                            'clinical-note analysis.'
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
          String label, bool selected, Color? tone, VoidCallback onTap) =>
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
                  color: selected ? Colors.transparent : Ds.hairlineStrong),
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

// ─────────────────────────────────────────────────────────────────────────────

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
    // Only a scored assessment gets a band chip. A GREY row is "no assessment",
    // and dressing it as a band would put it on the severity scale.
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
                      color: band?.fg ?? Ds.brand),
                ),
              ),
              const SizedBox(width: Ds.s3),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(patient.name,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                            fontSize: 14.5, fontWeight: FontWeight.w600)),
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
                // compositeLabel, not toStringAsFixed: a blocked assessment
                // shows an em dash here, never 0.000.
                Text(fusion!.compositeLabel,
                    style: AppTheme.data(size: 11.5, weight: FontWeight.w600)),
                const SizedBox(width: Ds.s2),
                Text(
                    '${fusion!.modalitiesUsed} of ${Modality.all.length} modalities',
                    style: const TextStyle(fontSize: 11, color: Ds.inkFaint)),
                const Spacer(),
                if (fusion!.updatedAt != null)
                  Text(DateFormat('d MMM, HH:mm').format(fusion!.updatedAt!),
                      style: const TextStyle(fontSize: 11, color: Ds.inkFaint)),
              ],
            ),
          ],
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────

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

  // One gender vocabulary for the whole app. The previous build had three
  // different lists across three screens, and "Other" was silently recoded.
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
                  'Scan the QR shown in the patient\'s Aura app. This copies '
                  'their Participant ID so all four components use the same '
                  'record.',
                  style: TextStyle(
                      fontSize: 12.5, color: Ds.inkMuted, height: 1.45),
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
                            .map((g) =>
                                DropdownMenuItem(value: g, child: Text(g)))
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
                  'Used only by the intervention engine\'s feature vector.',
                  style: TextStyle(fontSize: 11.5, color: Ds.inkFaint),
                ),
                const SizedBox(height: Ds.s3),
                DropdownButtonFormField<String>(
                  initialValue: _marital,
                  isExpanded: true,
                  decoration:
                      const InputDecoration(labelText: 'Marital status'),
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
                    (v) => setState(() => _education = v.round())),
                _slider(
                    'Income (PIR)',
                    _income,
                    0,
                    5,
                    20,
                    _income.toStringAsFixed(1),
                    (v) => setState(() => _income = v)),
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

  Widget _slider(String label, double v, double min, double max, int div,
          String readout, ValueChanged<double> onCh) =>
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(label,
                  style: const TextStyle(
                      fontSize: 12.5,
                      fontWeight: FontWeight.w600,
                      color: Ds.inkMuted)),
              const Spacer(),
              Text(readout, style: AppTheme.data(size: 12)),
            ],
          ),
          Slider(
            value: v,
            min: min,
            max: max,
            divisions: div,
            activeColor: Ds.brand,
            inactiveColor: Ds.surfaceSunken,
            onChanged: onCh,
          ),
        ],
      );
}
