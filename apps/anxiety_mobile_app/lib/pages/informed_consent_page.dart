import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../theme/app_theme.dart';
import '../services/background/service_config.dart';
import '../services/background_service_helper.dart';
import 'login_page.dart';

class InformedConsentPage extends StatefulWidget {
  const InformedConsentPage({super.key});

  @override
  State<InformedConsentPage> createState() => _InformedConsentPageState();
}

class _InformedConsentPageState extends State<InformedConsentPage> {
  bool _hasScrolledToBottom = false;
  final ScrollController _scrollController = ScrollController();

  // ── Active consent checkboxes (PDPA requirement) ──
  bool _understandPurpose = false;
  bool _understandDataTypes = false;
  bool _understandStorage = false;
  bool _understandRights = false;
  bool _understandVoluntary = false;
  bool _consentToParticipate = false;

  bool get _allChecked =>
      _understandPurpose &&
      _understandDataTypes &&
      _understandStorage &&
      _understandRights &&
      _understandVoluntary &&
      _consentToParticipate;

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(() {
      if (_scrollController.position.pixels >=
          _scrollController.position.maxScrollExtent - 50) {
        if (!_hasScrolledToBottom) {
          setState(() => _hasScrolledToBottom = true);
        }
      }
    });
  }

  Future<void> _acceptConsent() async {
    final prefs = await SharedPreferences.getInstance();
    final timestamp = DateTime.now().toIso8601String();

    await prefs.setBool('consent_accepted', true);
    await prefs.setString('consent_timestamp', timestamp);
    await prefs.setString('consent_version', ServiceConfig.consentVersion);

    // Store individual checkbox states for audit trail
    final consentRecord = {
      'consent_version': ServiceConfig.consentVersion,
      'consent_date': ServiceConfig.consentDate,
      'timestamp': timestamp,
      'understand_purpose': _understandPurpose,
      'understand_data_types': _understandDataTypes,
      'understand_storage': _understandStorage,
      'understand_rights': _understandRights,
      'understand_voluntary': _understandVoluntary,
      'consent_to_participate': _consentToParticipate,
      'study_title': ServiceConfig.studyTitle,
      'erc_number': ServiceConfig.ercApprovalNumber,
    };
    await prefs.setString('consent_record', jsonEncode(consentRecord));

    // Send consent record to Google Sheet for audit trail
    final userId = prefs.getString('user_id') ?? 'pre_registration';
    await BackgroundServiceHelper.sendToSheet(
      userId,
      'Consent_Record',
      jsonEncode(consentRecord),
    );

    if (mounted) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const LoginPage()),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [AppTheme.kBgTop, AppTheme.kBgBottom],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              _buildHeader(),
              Expanded(
                child: Container(
                  margin: const EdgeInsets.symmetric(horizontal: 16),
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.95),
                    borderRadius: BorderRadius.circular(24),
                    border: Border.all(color: Colors.white, width: 2),
                  ),
                  child: Scrollbar(
                    controller: _scrollController,
                    thumbVisibility: true,
                    child: SingleChildScrollView(
                      controller: _scrollController,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          _buildStudyHeader(),
                          _divider(),
                          _buildSection1Purpose(),
                          _divider(),
                          _buildSection2DataTypes(),
                          _divider(),
                          _buildSection3Storage(),
                          _divider(),
                          _buildSection4Rights(),
                          _divider(),
                          _buildSection5RisksBenefits(),
                          _divider(),
                          _buildSection6Voluntary(),
                          _divider(),
                          _buildSection7Contact(),
                          _divider(),
                          _buildConsentCheckboxes(),
                          const SizedBox(height: 30),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
              _buildFooter(),
            ],
          ),
        ),
      ),
    );
  }

  // ── Header ──
  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.all(20.0),
      child: Row(
        children: [
          const Icon(Icons.gavel_rounded, color: AppTheme.kPrimaryDeep, size: 28),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              "Informed Consent",
              style: GoogleFonts.poppins(
                fontSize: 18,
                fontWeight: FontWeight.w600,
                color: AppTheme.kTextDark,
              ),
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: AppTheme.kPrimaryDeep.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              'v${ServiceConfig.consentVersion}',
              style: const TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.bold,
                color: AppTheme.kPrimaryDeep,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ── Study Header ──
  Widget _buildStudyHeader() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _sectionTitle("Participant Information Sheet & Consent Form"),
        const SizedBox(height: 12),
        _infoRow("Study:", ServiceConfig.studyTitle),
        _infoRow("Principal Investigator:", ServiceConfig.piName),
        _infoRow("Institution:", ServiceConfig.piAffiliation),
        _infoRow("Ethics Approval:", ServiceConfig.ercApprovalNumber),
        _infoRow("Consent Version:", '${ServiceConfig.consentVersion} (${ServiceConfig.consentDate})'),
        const SizedBox(height: 12),
        _paragraph(
          "You are invited to participate in a research study. Before you decide, "
          "it is important for you to understand why the research is being done and "
          "what it will involve. Please read the following information carefully. "
          "You may ask the research team any questions if there is anything that is "
          "not clear or if you would like more information.",
        ),
      ],
    );
  }

  // ── Section 1: Purpose ──
  Widget _buildSection1Purpose() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _subTitle("1. Purpose of the Study"),
        _paragraph(
          "This study investigates digital biomarkers of anxiety by analysing "
          "passive smartphone sensor data alongside self-reported clinical "
          "assessments (GAD-7 and PSS-10). The goal is to identify behavioural "
          "patterns associated with anxiety, which could inform future digital "
          "mental health interventions.",
        ),
        _paragraph(
          "This research is conducted in accordance with the Sri Lanka Personal "
          "Data Protection Act (PDPA), No. 9 of 2022, and has received ethical "
          "approval from the ${ServiceConfig.ercName}.",
        ),
      ],
    );
  }

  // ── Section 2: Data Types ──
  Widget _buildSection2DataTypes() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _subTitle("2. What Data Do We Collect?"),
        _paragraph("This app collects the following data for research purposes:"),
        _dataItem("Clinical Surveys", "GAD-7 anxiety and PSS-10 stress assessments, daily mood ratings (EMA)."),
        _dataItem("Screen Activity", "Screen on/off times and unlock events (no screen content is captured)."),
        _dataItem("Touch Pressure", "Pressure applied when interacting with the app's orb interface."),
        _dataItem("Motion Data", "Significant movement events from the accelerometer (threshold-based, not continuous)."),
        _dataItem("Communication Patterns", "Count of incoming/outgoing/missed calls and SMS messages. No names, numbers, or message content are collected."),
        _dataItem("App Usage", "Time spent in app categories (e.g., 'Social Media', 'Education'). Individual app names are replaced with categories for privacy."),
        _dataItem("Location", "Periodic GPS coordinates, fuzzed to ±1 km precision to protect your exact location."),
        _dataItem("Battery Level", "Device battery percentage and charging state."),
        _dataItem("Demographics", "Age, gender, marital status, employment, education, living situation, sleep quality, and anxiety diagnosis history — collected once during profile setup."),
        const SizedBox(height: 8),
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Colors.blue.shade50,
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: Colors.blue.shade200),
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(Icons.info_outline, size: 18, color: Colors.blue.shade700),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  "We do NOT collect: your name, phone number, email address, "
                  "message content, call recordings, photos, browsing history, "
                  "or any passwords.",
                  style: TextStyle(fontSize: 13, color: Colors.blue.shade900, height: 1.4),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  // ── Section 3: Storage ──
  Widget _buildSection3Storage() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _subTitle("3. Data Storage & Security"),
        _paragraph(
          "Your data is transmitted securely via HTTPS encryption and stored in "
          "Google Cloud infrastructure (Google Sheets via Google Apps Script). "
          "Data is associated only with your Participant ID — not your real name.",
        ),
        _bulletItem("Data is encrypted in transit (HTTPS/TLS)."),
        _bulletItem("Access is restricted to authorised researchers only."),
        _bulletItem("GPS coordinates are fuzzed to ±1 km before storage."),
        _bulletItem("App names are replaced with categories (e.g., 'Social Media')."),
        _bulletItem("No SMS/call content or contact names are stored."),
        const SizedBox(height: 12),
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Colors.orange.shade50,
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: Colors.orange.shade200),
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(Icons.language, size: 18, color: Colors.orange.shade800),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  "Cross-Border Transfer: By consenting, you acknowledge that "
                  "your pseudonymised data will be stored on Google Cloud servers "
                  "which may be located outside Sri Lanka, as permitted under the "
                  "PDPA (Section 25) for scientific research with appropriate safeguards.",
                  style: TextStyle(fontSize: 13, color: Colors.orange.shade900, height: 1.4),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        _paragraph(
          "Data Retention: Your data will be retained for "
          "${ServiceConfig.dataRetentionPeriod}, after which it will be permanently deleted.",
        ),
      ],
    );
  }

  // ── Section 4: Rights ──
  Widget _buildSection4Rights() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _subTitle("4. Your Rights Under the PDPA"),
        _paragraph(
          "Under the Sri Lanka Personal Data Protection Act, you have the following rights:",
        ),
        _rightItem("Right to Access", "Request a copy of all data collected about you."),
        _rightItem("Right to Rectification", "Request correction of inaccurate data."),
        _rightItem("Right to Erasure", "Request permanent deletion of your data."),
        _rightItem("Right to Withdraw", "Withdraw your consent and stop participation at any time, without penalty."),
        _rightItem("Right to Object", "Object to specific types of data processing."),
        _rightItem("Right to Complain", "Lodge a complaint with the Data Protection Authority of Sri Lanka."),
        const SizedBox(height: 8),
        _paragraph(
          "To exercise any of these rights, use the 'Data Rights' section in "
          "the app's Profile page, or contact the research team at "
          "${ServiceConfig.researchTeamEmail}.",
        ),
      ],
    );
  }

  // ── Section 5: Risks & Benefits ──
  Widget _buildSection5RisksBenefits() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _subTitle("5. Risks & Benefits"),
        _paragraph("Risks:"),
        _bulletItem("Minimal risk. The app collects passive sensor data and self-reported surveys."),
        _bulletItem("There is a small risk of data breach, mitigated by pseudonymisation and access controls."),
        _bulletItem("Battery usage may increase slightly due to background data collection."),
        const SizedBox(height: 8),
        _paragraph("Benefits:"),
        _bulletItem("You will gain awareness of your own anxiety patterns through regular self-assessments."),
        _bulletItem("Your participation contributes to research that may improve digital mental health tools."),
        _bulletItem("There is no direct financial compensation for participation."),
      ],
    );
  }

  // ── Section 6: Voluntary ──
  Widget _buildSection6Voluntary() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _subTitle("6. Voluntary Participation"),
        _paragraph(
          "Participation is completely voluntary. You may withdraw from the study "
          "at any time without giving a reason and without any negative consequences. "
          "If you withdraw, you may request deletion of all previously collected data.",
        ),
        _paragraph(
          "If you are a patient at the National Hospital of Sri Lanka, your decision "
          "to participate or not will have NO effect on your medical care or treatment.",
        ),
      ],
    );
  }

  // ── Section 7: Contact ──
  Widget _buildSection7Contact() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _subTitle("7. Contact Information"),
        _contactCard("Principal Investigator", ServiceConfig.piName, ServiceConfig.piEmail),
        const SizedBox(height: 8),
        _contactCard("Research Supervisor", ServiceConfig.supervisorName, ServiceConfig.supervisorEmail),
        const SizedBox(height: 8),
        _contactCard("Ethics Review Committee", ServiceConfig.ercName, ServiceConfig.ercSecretaryEmail),
        const SizedBox(height: 12),
        _paragraph(
          "If you have any concerns about how your data is being processed, you may "
          "also contact the Data Protection Authority of Sri Lanka.",
        ),
      ],
    );
  }

  // ── Consent Checkboxes ──
  Widget _buildConsentCheckboxes() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _subTitle("Consent Declaration"),
        _paragraph("Please confirm each of the following by ticking the checkboxes:"),
        _consentCheck(
          "I have read and understood the purpose and procedures of this study.",
          _understandPurpose,
          (v) => setState(() => _understandPurpose = v ?? false),
        ),
        _consentCheck(
          "I understand what data will be collected and how it will be used for research.",
          _understandDataTypes,
          (v) => setState(() => _understandDataTypes = v ?? false),
        ),
        _consentCheck(
          "I understand my data will be stored securely and may be transferred to servers outside Sri Lanka.",
          _understandStorage,
          (v) => setState(() => _understandStorage = v ?? false),
        ),
        _consentCheck(
          "I understand my rights under the PDPA, including the right to withdraw and request data deletion.",
          _understandRights,
          (v) => setState(() => _understandRights = v ?? false),
        ),
        _consentCheck(
          "I understand that participation is voluntary and I can withdraw at any time without penalty.",
          _understandVoluntary,
          (v) => setState(() => _understandVoluntary = v ?? false),
        ),
        const SizedBox(height: 12),
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Colors.green.shade50,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: Colors.green.shade300, width: 2),
          ),
          child: _consentCheck(
            "I voluntarily consent to participate in this study.",
            _consentToParticipate,
            (v) => setState(() => _consentToParticipate = v ?? false),
            isBold: true,
          ),
        ),
      ],
    );
  }

  // ── Footer ──
  Widget _buildFooter() {
    return Padding(
      padding: const EdgeInsets.all(20.0),
      child: Column(
        children: [
          if (!_hasScrolledToBottom)
            Text(
              "Please scroll to the bottom to read all sections",
              style: TextStyle(color: Colors.grey.shade600, fontSize: 12),
            ),
          if (_hasScrolledToBottom && !_allChecked)
            Text(
              "Please tick all checkboxes to proceed",
              style: TextStyle(color: Colors.orange.shade700, fontSize: 12),
            ),
          const SizedBox(height: 10),
          SizedBox(
            width: double.infinity,
            height: 56,
            child: ElevatedButton(
              onPressed: (_hasScrolledToBottom && _allChecked) ? _acceptConsent : null,
              style: ElevatedButton.styleFrom(
                backgroundColor: AppTheme.kPrimaryDeep,
                disabledBackgroundColor: Colors.grey.shade300,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
              ),
              child: Text(
                "I Consent & Continue",
                style: GoogleFonts.poppins(
                  fontWeight: FontWeight.w600,
                  fontSize: 16,
                  color: Colors.white,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ── Reusable Widgets ──

  Widget _sectionTitle(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Text(
        text,
        style: GoogleFonts.poppins(
          fontSize: 17,
          fontWeight: FontWeight.bold,
          color: AppTheme.kPrimaryDeep,
        ),
      ),
    );
  }

  Widget _subTitle(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8, top: 8),
      child: Text(
        text,
        style: const TextStyle(
          fontSize: 15,
          fontWeight: FontWeight.bold,
          color: Colors.black87,
        ),
      ),
    );
  }

  Widget _paragraph(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Text(
        text,
        style: TextStyle(fontSize: 13.5, height: 1.5, color: Colors.grey.shade800),
      ),
    );
  }

  Widget _bulletItem(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 5, left: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.only(top: 6),
            child: Icon(Icons.circle, size: 5, color: AppTheme.kAccentBlue),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(text, style: TextStyle(fontSize: 13, color: Colors.grey.shade800, height: 1.4)),
          ),
        ],
      ),
    );
  }

  Widget _dataItem(String label, String description) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8, left: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.only(top: 2),
            child: Icon(Icons.sensors, size: 16, color: AppTheme.kPrimaryDeep),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: RichText(
              text: TextSpan(
                style: TextStyle(fontSize: 13, color: Colors.grey.shade800, height: 1.4),
                children: [
                  TextSpan(text: '$label: ', style: const TextStyle(fontWeight: FontWeight.w600)),
                  TextSpan(text: description),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _rightItem(String right, String description) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6, left: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.only(top: 2),
            child: Icon(Icons.shield_outlined, size: 16, color: Colors.green),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: RichText(
              text: TextSpan(
                style: TextStyle(fontSize: 13, color: Colors.grey.shade800, height: 1.4),
                children: [
                  TextSpan(text: '$right — ', style: const TextStyle(fontWeight: FontWeight.w600)),
                  TextSpan(text: description),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _infoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 130,
            child: Text(label, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: Colors.black54)),
          ),
          Expanded(child: Text(value, style: const TextStyle(fontSize: 12, color: Colors.black87))),
        ],
      ),
    );
  }

  Widget _contactCard(String role, String name, String email) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.grey.shade50,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.grey.shade200),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(role, style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: Colors.black54)),
          const SizedBox(height: 4),
          Text(name, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
          Text(email, style: TextStyle(fontSize: 12, color: Colors.blue.shade700)),
        ],
      ),
    );
  }

  Widget _consentCheck(String text, bool value, ValueChanged<bool?> onChanged, {bool isBold = false}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 24,
            height: 24,
            child: Checkbox(
              value: value,
              onChanged: _hasScrolledToBottom ? onChanged : null,
              activeColor: AppTheme.kPrimaryDeep,
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: GestureDetector(
              onTap: _hasScrolledToBottom ? () => onChanged(!value) : null,
              child: Text(
                text,
                style: TextStyle(
                  fontSize: 13,
                  height: 1.4,
                  fontWeight: isBold ? FontWeight.w600 : FontWeight.normal,
                  color: Colors.grey.shade800,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _divider() {
    return Divider(height: 28, color: Colors.grey.shade200);
  }
}
