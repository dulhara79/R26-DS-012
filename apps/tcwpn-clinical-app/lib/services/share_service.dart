import 'package:share_plus/share_plus.dart';
import '../models/models.dart';

class ShareService {
  static Future<void> shareAssessment(Patient patient, Assessment assessment) async {
    final StringBuffer buffer = StringBuffer();
    
    buffer.writeln('--- CLINICAL ASSESSMENT REPORT ---');
    buffer.writeln('Patient: ${patient.name} (${patient.id})');
    buffer.writeln('Age/Gender: ${patient.age} / ${patient.gender}');
    buffer.writeln('Ward: ${patient.ward}');
    buffer.writeln('Date: ${_formatDate(assessment.timestamp)}');
    buffer.writeln('Clinician ID: ${assessment.clinicianId}');
    buffer.writeln('');
    buffer.writeln('NOTE TYPE: ${assessment.noteType}');
    buffer.writeln('--- CLINICAL NOTE ---');
    buffer.writeln(assessment.noteText);
    buffer.writeln('');
    
    if (assessment.result != null) {
      final r = assessment.result!;
      buffer.writeln('--- AI ANALYSIS RESULTS ---');
      buffer.writeln('Prediction: ${r.prediction}');
      buffer.writeln('Risk Level: ${r.riskLevel.label}');
      buffer.writeln('Confidence: ${(r.confidence * 100).toStringAsFixed(1)}%');
      buffer.writeln('Key Indicators: ${r.keyPhrases.join(", ")}');
      buffer.writeln('Temporal Context: ${r.temporalContext}');
    }
    
    buffer.writeln('');
    buffer.writeln('Generated via TC-WPN Clinical Dashboard');

    await Share.share(
      buffer.toString(),
      subject: 'Clinical Assessment: ${patient.name}',
    );
  }

  static String _formatDate(DateTime dt) {
    return '${dt.day}/${dt.month}/${dt.year} ${dt.hour}:${dt.minute.toString().padLeft(2, '0')}';
  }
}
