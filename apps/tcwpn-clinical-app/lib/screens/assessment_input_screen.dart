import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/models.dart';
import '../theme/app_theme.dart';
import '../services/patient_provider.dart';
import '../services/api_service.dart';
import 'assessment_result_screen.dart';

class AssessmentInputScreen extends StatefulWidget {
  final Patient patient;
  const AssessmentInputScreen({super.key, required this.patient});

  @override
  State<AssessmentInputScreen> createState() => _AssessmentInputScreenState();
}

class _AssessmentInputScreenState extends State<AssessmentInputScreen> {
  final _noteCtrl = TextEditingController();
  String _noteType = 'Psychiatry note';
  bool _submitting = false;

  static const _exampleNotes = {
    'GAD — active': 'Patient presents with persistent and excessive worry about work, health, and finances for the past 8 months. Reports difficulty controlling the worry, which is present most days. Associated symptoms include fatigue, difficulty concentrating, irritability, muscle tension, and disturbed sleep. PHQ-9 score 16. GAD-7 score 14. Currently prescribed sertraline 100mg daily and referred for CBT. Diagnosis: Generalized anxiety disorder F41.1.',
    'Panic disorder': '28-year-old patient presenting with recurrent unexpected panic attacks over 6 months. Episodes characterised by palpitations, chest tightness, sweating, trembling, and intense fear of dying lasting 10–20 minutes. Persistent worry about further attacks causing avoidance of public transport. Diagnosis: Panic disorder F41.0. Started on escitalopram 10mg.',
    'Follow-up stable': 'Patient stable on sertraline 100mg. Reports significant reduction in anxiety symptoms. GAD-7 score improved from 16 to 8. Sleep and concentration improved. Continuing CBT sessions. No side effects reported. Plan: continue current management, review in 6 weeks.',
  };

  final List<String> _noteTypes = [
    'Psychiatry note',
    'Discharge summary',
    'Nursing note',
    'Social work note',
    'Physician note',
  ];

  Future<void> _submit({bool skipAnalysis = false}) async {
    final text = _noteCtrl.text.trim();
    if (text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter a clinical note')),
      );
      return;
    }

    setState(() => _submitting = true);

    try {
      final result = await context.read<PatientProvider>().saveAssessment(
        patientId: widget.patient.id,
        noteText:  text,
        noteType:  _noteType,
        skipAnalysis: skipAnalysis,
      );

      if (!mounted) return;

      if (result != null) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (_) => AssessmentResultScreen(
              patient: widget.patient,
              noteText: text,
              noteType: _noteType,
              result: result,
            ),
          ),
        );
      } else {
        // Offline save success
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Assessment saved locally (Draft)')),
        );
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _submitting = false);
      
      String errorMsg = e.toString();
      if (e is ApiException) {
        errorMsg = e.userMessage;
      }

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Analysis failed: $errorMsg'),
          backgroundColor: AppColors.riskHigh,
          duration: const Duration(seconds: 10),
          action: SnackBarAction(
            label: 'Save Offline',
            textColor: Colors.white,
            onPressed: () => _submit(skipAnalysis: true),
          ),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final isOffline = context.watch<PatientProvider>().isOffline;

    return Scaffold(
      backgroundColor: AppColors.surfaceSecond,
      appBar: AppBar(
        title: const Text('New assessment'),
        leading: IconButton(
          icon: const Icon(Icons.close_rounded),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: Column(
        children: [
          // Offline Banner
          if (isOffline)
            Container(
              color: AppColors.riskHigh,
              width: double.infinity,
              padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 16),
              child: const Row(
                children: [
                  Icon(Icons.wifi_off_rounded, color: Colors.white, size: 14),
                  SizedBox(width: 8),
                  Text(
                    'Offline Mode: AI analysis unavailable. Saving as local draft.',
                    style: TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.bold),
                  ),
                ],
              ),
            ),

          // Patient header strip
          Container(
            color: AppColors.primarySurface,
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
            child: Row(
              children: [
                Container(
                  width: 36, height: 36,
                  decoration: BoxDecoration(
                    color: AppColors.primary,
                    shape: BoxShape.circle,
                  ),
                  alignment: Alignment.center,
                  child: Text(
                    widget.patient.initials,
                    style: const TextStyle(
                      color: Colors.white, fontSize: 14,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(widget.patient.name,
                        style: Theme.of(context).textTheme.titleMedium
                            ?.copyWith(color: AppColors.primary)),
                    Text(
                      '${widget.patient.id} · Visit ${widget.patient.totalVisits + 1}',
                      style: const TextStyle(
                          fontSize: 12, color: AppColors.textSecondary),
                    ),
                  ],
                ),
              ],
            ),
          ),

          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Note type selector
                  DropdownButtonFormField<String>(
                    value: _noteType,
                    decoration: const InputDecoration(
                      labelText: 'Note type',
                      prefixIcon: Icon(Icons.description_outlined, size: 20),
                    ),
                    items: _noteTypes.map((t) => DropdownMenuItem(
                      value: t, child: Text(t),
                    )).toList(),
                    onChanged: (v) => setState(() => _noteType = v!),
                  ),
                  const SizedBox(height: 14),

                  // Note text area
                  TextField(
                    controller: _noteCtrl,
                    maxLines: 12,
                    onChanged: (_) => setState(() {}),
                    decoration: const InputDecoration(
                      labelText: 'Clinical note',
                      alignLabelWithHint: true,
                      hintText:
                          'Paste or type the clinical note here...\n\n'
                          'Include relevant sections: history of present illness, '
                          'mental status examination, assessment, medications.',
                    ),
                  ),
                  const SizedBox(height: 14),

                  // Word count
                  Align(
                    alignment: Alignment.centerRight,
                    child: Text(
                      '${_noteCtrl.text.split(RegExp(r'\s+')).where((w) => w.isNotEmpty).length} words',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ),
                  const SizedBox(height: 6),

                  // Example buttons
                  Text('Load example note',
                      style: Theme.of(context).textTheme.titleSmall),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8, runSpacing: 8,
                    children: _exampleNotes.entries.map((e) =>
                      OutlinedButton(
                        onPressed: () {
                          _noteCtrl.text = e.value;
                          setState(() {});
                        },
                        style: OutlinedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 12, vertical: 8),
                          textStyle: const TextStyle(fontSize: 12),
                        ),
                        child: Text(e.key),
                      ),
                    ).toList(),
                  ),
                  const SizedBox(height: 20),

                  // Disclaimer
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: AppColors.warning.withOpacity(0.05),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(
                          color: AppColors.warning.withOpacity(0.3)),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(Icons.warning_amber_rounded,
                            size: 16, color: AppColors.warning),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            isOffline 
                              ? 'Offline mode: You can save this note locally. It will not be analyzed by the AI until a connection is restored.'
                              : 'This analysis is for clinical decision support only. Remove all patient identifiers before submitting.',
                            style: TextStyle(
                              fontSize: 12, color: AppColors.warning, height: 1.4,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 24),

                  // Submit
                  ElevatedButton.icon(
                    onPressed: _submitting ? null : () => _submit(skipAnalysis: isOffline),
                    icon: _submitting
                        ? const SizedBox(
                            width: 18, height: 18,
                            child: CircularProgressIndicator(
                              strokeWidth: 2, color: Colors.white,
                            ),
                          )
                        : Icon(isOffline ? Icons.save_rounded : Icons.analytics_rounded, size: 20),
                    label: Text(_submitting 
                      ? (isOffline ? 'Saving...' : 'Analysing...') 
                      : (isOffline ? 'Save Offline Draft' : 'Analyse note')),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      backgroundColor: isOffline ? AppColors.info : AppColors.primary,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
