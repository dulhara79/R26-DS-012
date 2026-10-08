import 'dart:typed_data';

import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;

import 'c2_clinician_summary.dart';

/// Builds a one-page A4 PDF of the participant's clinician summary.
///
/// Uses the PDF standard Helvetica font so it works offline and never fetches
/// fonts. That font only covers Latin-1, so text is passed through [pdfSafe].
Future<Uint8List> buildClinicianSummaryPdf(
  ClinicianSummary summary, {
  required Set<SummarySectionId> include,
}) async {
  const purple = PdfColor.fromInt(0xFF5E60CE);
  const tint = PdfColor.fromInt(0xFFF0ECFF);
  const muted = PdfColor.fromInt(0xFF5A607F);
  const border = PdfColor.fromInt(0xFFE8E5F4);

  final doc = pw.Document(
    title: 'Aura summary for my clinician',
    author: 'Aura research app',
  );

  pw.Widget section(SummarySection s) => pw.Container(
    margin: const pw.EdgeInsets.only(bottom: 12),
    decoration: pw.BoxDecoration(
      border: pw.Border.all(color: border),
      borderRadius: pw.BorderRadius.circular(6),
    ),
    child: pw.Column(
      crossAxisAlignment: pw.CrossAxisAlignment.stretch,
      children: [
        pw.Container(
          padding: const pw.EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: const pw.BoxDecoration(
            color: tint,
            borderRadius: pw.BorderRadius.only(
              topLeft: pw.Radius.circular(6),
              topRight: pw.Radius.circular(6),
            ),
          ),
          child: pw.Row(
            children: [
              pw.Text(
                pdfSafe(s.title),
                style: pw.TextStyle(
                  fontWeight: pw.FontWeight.bold,
                  fontSize: 11.5,
                  color: purple,
                ),
              ),
              pw.SizedBox(width: 8),
              pw.Expanded(
                child: pw.Text(
                  pdfSafe(s.description),
                  style: const pw.TextStyle(fontSize: 8.5, color: muted),
                ),
              ),
            ],
          ),
        ),
        pw.Padding(
          padding: const pw.EdgeInsets.fromLTRB(10, 6, 10, 8),
          child: pw.Column(
            crossAxisAlignment: pw.CrossAxisAlignment.stretch,
            children: [
              if (s.isEmpty)
                pw.Text(
                  pdfSafe(s.emptyText),
                  style: const pw.TextStyle(fontSize: 9.5, color: muted),
                )
              else
                for (final row in s.rows)
                  pw.Padding(
                    padding: const pw.EdgeInsets.symmetric(vertical: 2),
                    child: pw.Row(
                      crossAxisAlignment: pw.CrossAxisAlignment.start,
                      children: [
                        pw.Expanded(
                          flex: 5,
                          child: pw.Text(
                            pdfSafe(row.label),
                            style: const pw.TextStyle(fontSize: 9.5),
                          ),
                        ),
                        pw.Expanded(
                          flex: 6,
                          child: pw.Text(
                            pdfSafe(row.value),
                            style: pw.TextStyle(
                              fontSize: 9.5,
                              fontWeight: pw.FontWeight.bold,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
              pw.SizedBox(height: 5),
              pw.Text(
                pdfSafe('How to read this: ${s.note}'),
                style: pw.TextStyle(
                  fontSize: 8,
                  color: muted,
                  fontStyle: pw.FontStyle.italic,
                ),
              ),
            ],
          ),
        ),
      ],
    ),
  );

  final included = summary.sections
      .where((s) => include.contains(s.id))
      .toList();

  doc.addPage(
    pw.MultiPage(
      pageFormat: PdfPageFormat.a4,
      margin: const pw.EdgeInsets.fromLTRB(36, 36, 36, 30),
      footer: (context) => pw.Row(
        children: [
          pw.Expanded(
            child: pw.Text(
              pdfSafe(ClinicianSummary.privacyNote),
              style: const pw.TextStyle(fontSize: 7, color: muted),
            ),
          ),
          pw.Text(
            'Page ${context.pageNumber} of ${context.pagesCount}',
            style: const pw.TextStyle(fontSize: 7, color: muted),
          ),
        ],
      ),
      build: (context) => [
        pw.Row(
          crossAxisAlignment: pw.CrossAxisAlignment.end,
          children: [
            pw.Expanded(
              child: pw.Column(
                crossAxisAlignment: pw.CrossAxisAlignment.start,
                children: [
                  pw.Text(
                    'Aura',
                    style: pw.TextStyle(
                      fontSize: 20,
                      fontWeight: pw.FontWeight.bold,
                      color: purple,
                    ),
                  ),
                  pw.Text(
                    'Summary for my clinician',
                    style: const pw.TextStyle(fontSize: 13),
                  ),
                ],
              ),
            ),
            pw.Column(
              crossAxisAlignment: pw.CrossAxisAlignment.end,
              children: [
                pw.Text(
                  pdfSafe('Participant ID: ${summary.participantId}'),
                  style: pw.TextStyle(
                    fontSize: 9,
                    fontWeight: pw.FontWeight.bold,
                  ),
                ),
                pw.Text(
                  pdfSafe('Period: ${summary.period.label}'),
                  style: const pw.TextStyle(fontSize: 9, color: muted),
                ),
                pw.Text(
                  pdfSafe(
                    'Prepared: ${ClinicianSummary.formatDate(summary.generatedAt)}',
                  ),
                  style: const pw.TextStyle(fontSize: 9, color: muted),
                ),
              ],
            ),
          ],
        ),
        pw.SizedBox(height: 8),
        pw.Divider(color: purple, thickness: 1.2),
        pw.SizedBox(height: 4),
        pw.Text(
          pdfSafe(ClinicianSummary.disclaimer),
          style: const pw.TextStyle(fontSize: 8.5, color: muted),
        ),
        pw.SizedBox(height: 12),
        ...included.map(section),
      ],
    ),
  );

  return doc.save();
}

/// Replaces characters outside the PDF standard fonts' Latin-1 range.
String pdfSafe(String text) {
  const replacements = {
    '—': '-', // em dash
    '–': '-', // en dash
    '’': "'",
    '‘': "'",
    '“': '"',
    '”': '"',
    '…': '...',
    '≥': '>=',
    '≤': '<=',
    '→': '->',
    'σ': 'SD',
  };
  final buf = StringBuffer();
  for (final rune in text.runes) {
    final char = String.fromCharCode(rune);
    final replacement = replacements[char];
    if (replacement != null) {
      buf.write(replacement);
    } else if (rune <= 0xFF) {
      buf.write(char);
    } else {
      buf.write('?');
    }
  }
  return buf.toString();
}
