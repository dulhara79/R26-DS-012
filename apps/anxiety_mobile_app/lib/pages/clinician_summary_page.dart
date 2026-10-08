import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:printing/printing.dart';

import '../services/background_service_helper.dart';
import '../services/c2_clinician_pdf.dart';
import '../services/c2_clinician_summary.dart';
import '../services/clinician_longitudinal_context_service.dart';
import '../theme/c2_palette.dart';

typedef _C = C2Palette;

/// "Prepare for my appointment": the participant previews a descriptive
/// summary, chooses the period and which sections to include, then shares it
/// as a PDF or copies it as text. Nothing is sent automatically.
class ClinicianSummaryPage extends StatefulWidget {
  final String? userId;

  const ClinicianSummaryPage({super.key, this.userId});

  @override
  State<ClinicianSummaryPage> createState() => _ClinicianSummaryPageState();
}

class _ClinicianSummaryPageState extends State<ClinicianSummaryPage> {
  bool _loading = true;
  bool _sharing = false;
  String _participantId = '';
  Map<String, dynamic> _context = const {};
  SummaryPeriod _period = SummaryPeriod.month;
  final Set<SummarySectionId> _included = SummarySectionId.values.toSet();

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final id = widget.userId ?? await BackgroundServiceHelper.getCachedId();
    Map<String, dynamic> context = const {};
    try {
      context = await ClinicianLongitudinalContextService.buildAndCache(id);
    } catch (e) {
      debugPrint('Clinician summary load error: $e');
    }
    if (!mounted) return;
    setState(() {
      _participantId = id;
      _context = context;
      _loading = false;
    });
  }

  ClinicianSummary get _summary => ClinicianSummary.fromContext(
    _context,
    participantId: _participantId,
    period: _period,
  );

  String get _fileName {
    final d = DateTime.now();
    final date =
        '${d.year}-${d.month.toString().padLeft(2, '0')}-'
        '${d.day.toString().padLeft(2, '0')}';
    return 'aura_summary_${_participantId}_$date.pdf';
  }

  Future<void> _sharePdf() async {
    setState(() => _sharing = true);
    try {
      final bytes = await buildClinicianSummaryPdf(
        _summary,
        include: _included,
      );
      await Printing.sharePdf(bytes: bytes, filename: _fileName);
    } catch (e) {
      debugPrint('Clinician PDF error: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Could not create the PDF.')),
        );
      }
    } finally {
      if (mounted) setState(() => _sharing = false);
    }
  }

  Future<void> _copyText() async {
    await Clipboard.setData(
      ClipboardData(text: _summary.toPlainText(include: _included)),
    );
    if (!mounted) return;
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('Summary copied as text.')));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _C.scaffold,
      appBar: AppBar(
        backgroundColor: _C.scaffold,
        elevation: 0,
        title: Text(
          'Prepare for my appointment',
          style: GoogleFonts.poppins(
            fontSize: 17,
            fontWeight: FontWeight.w600,
            color: _C.textPrimary,
          ),
        ),
      ),
      body: _loading
          ? Center(child: CircularProgressIndicator(color: _C.primary))
          : _body(),
      bottomNavigationBar: _loading ? null : _actions(),
    );
  }

  Widget _body() {
    final summary = _summary;
    return ListView(
      padding: const EdgeInsets.fromLTRB(18, 4, 18, 24),
      children: [
        Text(
          'A short summary you can give your clinician. Check it, turn off '
          'anything you don’t want to share, then share it as a PDF.',
          style: GoogleFonts.poppins(
            fontSize: 12.5,
            height: 1.5,
            color: _C.textSecondary,
          ),
        ),
        const SizedBox(height: 14),
        SegmentedButton<SummaryPeriod>(
          segments: [
            for (final p in SummaryPeriod.values)
              ButtonSegment(value: p, label: Text(p.label)),
          ],
          selected: {_period},
          showSelectedIcon: false,
          onSelectionChanged: (s) => setState(() => _period = s.first),
          style: SegmentedButton.styleFrom(
            selectedBackgroundColor: _C.p100,
            selectedForegroundColor: _C.primary,
            foregroundColor: _C.textSecondary,
            textStyle: GoogleFonts.poppins(
              fontSize: 12.5,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
        const SizedBox(height: 14),
        _headerCard(summary),
        const SizedBox(height: 12),
        for (final section in summary.sections) ...[
          _sectionCard(section),
          const SizedBox(height: 12),
        ],
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(Icons.lock_outline_rounded, size: 15, color: _C.textMuted),
            const SizedBox(width: 6),
            Expanded(
              child: Text(
                '${ClinicianSummary.privacyNote} Aura never sends this '
                'automatically.',
                style: GoogleFonts.poppins(
                  fontSize: 11,
                  height: 1.45,
                  color: _C.textMuted,
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _headerCard(ClinicianSummary summary) => Container(
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: _C.p100,
      borderRadius: BorderRadius.circular(16),
      border: Border.all(color: _C.p200),
    ),
    child: Row(
      children: [
        Icon(Icons.description_outlined, color: _C.primary),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Participant ID: ${summary.participantId}',
                style: GoogleFonts.poppins(
                  fontSize: 12.5,
                  fontWeight: FontWeight.w600,
                  color: _C.textPrimary,
                ),
              ),
              Text(
                '${summary.period.label} · prepared '
                '${ClinicianSummary.formatDate(summary.generatedAt)}',
                style: GoogleFonts.poppins(
                  fontSize: 11.5,
                  color: _C.textSecondary,
                ),
              ),
            ],
          ),
        ),
      ],
    ),
  );

  Widget _sectionCard(SummarySection section) {
    final included = _included.contains(section.id);
    return AnimatedOpacity(
      duration: const Duration(milliseconds: 180),
      opacity: included ? 1 : 0.5,
      child: Container(
        padding: const EdgeInsets.fromLTRB(16, 10, 10, 14),
        decoration: BoxDecoration(
          color: _C.cardBase,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: _C.border),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        section.title,
                        style: GoogleFonts.poppins(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                          color: _C.textPrimary,
                        ),
                      ),
                      Text(
                        section.description,
                        style: GoogleFonts.poppins(
                          fontSize: 11,
                          color: _C.textMuted,
                        ),
                      ),
                    ],
                  ),
                ),
                Semantics(
                  label: 'Include ${section.title}',
                  child: Switch(
                    value: included,
                    activeTrackColor: _C.primary,
                    onChanged: (v) => setState(() {
                      if (v) {
                        _included.add(section.id);
                      } else {
                        _included.remove(section.id);
                      }
                    }),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            if (section.isEmpty)
              Text(
                section.emptyText,
                style: GoogleFonts.poppins(
                  fontSize: 12,
                  color: _C.textSecondary,
                ),
              )
            else
              for (final row in section.rows)
                Padding(
                  padding: const EdgeInsets.only(right: 6, bottom: 6),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Expanded(
                        flex: 5,
                        child: Text(
                          row.label,
                          style: GoogleFonts.poppins(
                            fontSize: 12,
                            color: _C.textSecondary,
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        flex: 5,
                        child: Text(
                          row.value,
                          textAlign: TextAlign.right,
                          style: GoogleFonts.poppins(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: _C.textPrimary,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
            const SizedBox(height: 4),
            Padding(
              padding: const EdgeInsets.only(right: 6),
              child: Text(
                section.note,
                style: GoogleFonts.poppins(
                  fontSize: 10.5,
                  height: 1.4,
                  fontStyle: FontStyle.italic,
                  color: _C.textMuted,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _actions() {
    final nothingSelected = _included.isEmpty;
    return SafeArea(
      child: Container(
        padding: const EdgeInsets.fromLTRB(18, 10, 18, 12),
        decoration: BoxDecoration(
          color: _C.cardBase,
          border: Border(top: BorderSide(color: _C.border)),
        ),
        child: Row(
          children: [
            Expanded(
              child: OutlinedButton.icon(
                onPressed: nothingSelected ? null : _copyText,
                icon: const Icon(Icons.copy_rounded, size: 17),
                label: const Text('Copy as text'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: _C.primary,
                  side: BorderSide(color: _C.p200),
                  padding: const EdgeInsets.symmetric(vertical: 13),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: FilledButton.icon(
                onPressed: nothingSelected || _sharing ? null : _sharePdf,
                icon: _sharing
                    ? const SizedBox(
                        width: 15,
                        height: 15,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.picture_as_pdf_outlined, size: 18),
                label: const Text('Share PDF'),
                style: FilledButton.styleFrom(
                  backgroundColor: _C.primary,
                  foregroundColor: _C.cardBase,
                  padding: const EdgeInsets.symmetric(vertical: 13),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
