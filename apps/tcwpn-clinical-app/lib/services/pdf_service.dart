import 'dart:typed_data';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';
import '../models/models.dart';

class PdfService {
  static Future<void> generateAndSavePdf(Patient patient, Assessment assessment) async {
    final pdf = pw.Document();

    final font = await PdfGoogleFonts.robotoRegular();
    final boldFont = await PdfGoogleFonts.robotoBold();

    pdf.addPage(
      pw.MultiPage(
        pageFormat: PdfPageFormat.a4,
        margin: const pw.EdgeInsets.all(32),
        build: (pw.Context context) {
          return [
            // Header
            pw.Row(
              mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
              children: [
                pw.Column(
                  crossAxisAlignment: pw.CrossAxisAlignment.start,
                  children: [
                    pw.Text('CLINICAL ASSESSMENT REPORT', 
                      style: pw.TextStyle(font: boldFont, fontSize: 18, color: PdfColors.blue900)),
                    pw.Text('TC-WPN Clinical Anxiety Detection System', 
                      style: pw.TextStyle(font: font, fontSize: 10, color: PdfColors.grey700)),
                  ],
                ),
                pw.Column(
                  crossAxisAlignment: pw.CrossAxisAlignment.end,
                  children: [
                    pw.Text('Report ID: ${assessment.id}', style: pw.TextStyle(font: font, fontSize: 8)),
                    pw.Text('Date: ${_formatDate(assessment.timestamp)}', style: pw.TextStyle(font: font, fontSize: 8)),
                  ],
                ),
              ],
            ),
            pw.SizedBox(height: 20),
            pw.Divider(thickness: 1, color: PdfColors.grey300),
            pw.SizedBox(height: 20),

            // Patient Information Section
            pw.Text('PATIENT INFORMATION', style: pw.TextStyle(font: boldFont, fontSize: 12)),
            pw.SizedBox(height: 10),
            pw.Container(
              padding: const pw.EdgeInsets.all(10),
              decoration: pw.BoxDecoration(
                border: pw.Border.all(color: PdfColors.grey300),
                borderRadius: const pw.BorderRadius.all(pw.Radius.circular(4)),
              ),
              child: pw.Column(
                children: [
                  _buildPdfRow('Name', patient.name, 'Patient ID', patient.id, font, boldFont),
                  pw.SizedBox(height: 5),
                  _buildPdfRow('Age', '${patient.age}', 'Gender', patient.gender, font, boldFont),
                  pw.SizedBox(height: 5),
                  _buildPdfRow('Ward/Dept', patient.ward, 'Clinician', assessment.clinicianId, font, boldFont),
                ],
              ),
            ),
            pw.SizedBox(height: 24),

            // Clinical Note Section
            pw.Text('CLINICAL NOTE (${assessment.noteType})', style: pw.TextStyle(font: boldFont, fontSize: 12)),
            pw.SizedBox(height: 10),
            pw.Container(
              width: double.infinity,
              padding: const pw.EdgeInsets.all(12),
              decoration: pw.BoxDecoration(
                color: PdfColors.grey100,
                borderRadius: const pw.BorderRadius.all(pw.Radius.circular(4)),
              ),
              child: pw.Text(
                assessment.noteText,
                style: pw.TextStyle(font: font, fontSize: 11, lineSpacing: 1.5),
              ),
            ),
            pw.SizedBox(height: 24),

            // AI Analysis Section
            if (assessment.result != null) ...[
              pw.Text('AI ANALYSIS & PREDICTION', style: pw.TextStyle(font: boldFont, fontSize: 12)),
              pw.SizedBox(height: 10),
              pw.Container(
                padding: const pw.EdgeInsets.all(12),
                decoration: pw.BoxDecoration(
                  border: pw.Border.all(color: _getRiskColor(assessment.result!.riskLevel)),
                  borderRadius: const pw.BorderRadius.all(pw.Radius.circular(4)),
                ),
                child: pw.Column(
                  crossAxisAlignment: pw.CrossAxisAlignment.start,
                  children: [
                    pw.Row(
                      mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
                      children: [
                        pw.Text('Prediction: ${assessment.result!.prediction}', 
                          style: pw.TextStyle(font: boldFont, fontSize: 14)),
                        pw.Container(
                          padding: const pw.EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                          decoration: pw.BoxDecoration(
                            color: _getRiskColor(assessment.result!.riskLevel),
                            borderRadius: const pw.BorderRadius.all(pw.Radius.circular(4)),
                          ),
                          child: pw.Text(
                            assessment.result!.riskLevel.label.toUpperCase(),
                            style: pw.TextStyle(font: boldFont, color: PdfColors.white, fontSize: 10),
                          ),
                        ),
                      ],
                    ),
                    pw.SizedBox(height: 10),
                    _buildPdfRow('Confidence Score', '${(assessment.result!.confidence * 100).toStringAsFixed(1)}%', 
                        'Risk Score', assessment.result!.riskScore.toStringAsFixed(4), font, boldFont),
                    pw.SizedBox(height: 10),
                    pw.Text('Key Indicators:', style: pw.TextStyle(font: boldFont, fontSize: 10)),
                    pw.Bullet(text: assessment.result!.keyPhrases.join(', '), 
                      style: pw.TextStyle(font: font, fontSize: 10)),
                    pw.SizedBox(height: 5),
                    pw.Text('Temporal Context: ${assessment.result!.temporalContext}', 
                      style: pw.TextStyle(font: font, fontSize: 10, color: PdfColors.grey700)),
                  ],
                ),
              ),
            ],

            pw.Spacer(),

            // Footer
            pw.Divider(thickness: 0.5, color: PdfColors.grey300),
            pw.Row(
              mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
              children: [
                pw.Text('Confidential Medical Record', style: pw.TextStyle(font: font, fontSize: 8, color: PdfColors.grey600)),
                pw.Text('Generated by TC-WPN System', style: pw.TextStyle(font: font, fontSize: 8, color: PdfColors.grey600)),
                pw.Text('Signature: ____________________', style: pw.TextStyle(font: font, fontSize: 8)),
              ],
            ),
          ];
        },
      ),
    );

    // This will open the print/save dialog which allows "Save as PDF" (download to mobile)
    await Printing.layoutPdf(
      onLayout: (PdfPageFormat format) async => pdf.save(),
      name: 'Assessment_${patient.id}_${DateTime.now().millisecondsSinceEpoch}.pdf',
    );
  }

  static pw.Widget _buildPdfRow(String label1, String value1, String label2, String value2, pw.Font font, pw.Font boldFont) {
    return pw.Row(
      children: [
        pw.Expanded(
          child: pw.RichText(
            text: pw.TextSpan(
              children: [
                pw.TextSpan(text: '$label1: ', style: pw.TextStyle(font: boldFont, fontSize: 10)),
                pw.TextSpan(text: value1, style: pw.TextStyle(font: font, fontSize: 10)),
              ],
            ),
          ),
        ),
        pw.Expanded(
          child: pw.RichText(
            text: pw.TextSpan(
              children: [
                pw.TextSpan(text: '$label2: ', style: pw.TextStyle(font: boldFont, fontSize: 10)),
                pw.TextSpan(text: value2, style: pw.TextStyle(font: font, fontSize: 10)),
              ],
            ),
          ),
        ),
      ],
    );
  }

  static PdfColor _getRiskColor(RiskLevel risk) {
    switch (risk) {
      case RiskLevel.low:      return PdfColors.green;
      case RiskLevel.moderate: return PdfColors.orange;
      case RiskLevel.high:     return PdfColors.red;
      case RiskLevel.veryHigh: return PdfColors.red900;
    }
  }

  static String _formatDate(DateTime dt) {
    return '${dt.day}/${dt.month}/${dt.year} ${dt.hour}:${dt.minute.toString().padLeft(2, '0')}';
  }
}
