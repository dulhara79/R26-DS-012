import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:intl/intl.dart';

import '../theme/app_theme.dart';
import '../services/background/service_config.dart';
import 'login_page.dart';

/// InformedConsentPage
///
/// Used in TWO modes controlled by [readOnly]:
///
///   readOnly = false  (default)
///   ── First-run consent flow.
///      Participant must scroll to bottom AND confirm all declarations.
///      "I Consent & Continue" button unlocks only when both are done.
///      On confirmation, all state is persisted to SharedPreferences.
///
///   readOnly = true
///   ── Called from Profile / "Manage Data Rights" section.
///      All content is shown.  All checkboxes are ticked and LOCKED
///      (non-interactive) reflecting the consent already given.
///      The consent timestamp is shown at the bottom.
///      No "Consent" button is shown — a "Close" button replaces it.
///
/// The constructor parameter allows any page to open this in read-only mode:
///   Navigator.push(context, MaterialPageRoute(
///     builder: (_) => const InformedConsentPage(readOnly: true),
///   ));
class InformedConsentPage extends StatefulWidget {
  final bool readOnly;
  final Widget? nextPage;
  final Future<void> Function()? onAccepted;

  const InformedConsentPage({
    super.key,
    this.readOnly = false,
    this.nextPage,
    this.onAccepted,
  });

  @override
  State<InformedConsentPage> createState() => _InformedConsentPageState();
}

class _InformedConsentPageState extends State<InformedConsentPage> {
  final ScrollController _scrollController = ScrollController();

  bool _hasScrolledToBottom = false;
  String _consentTimestamp = '';

  // Eight declarations — loaded from prefs in readOnly mode.
  bool _cbAge = false;
  bool _cbPurpose = false;
  bool _cbData = false;
  bool _cbPhysio = false;
  bool _cbStorage = false;
  bool _cbRights = false;
  bool _cbVoluntary = false;
  bool _cbLiability = false;

  bool get _allChecked =>
      _cbAge &&
      _cbPurpose &&
      _cbData &&
      _cbPhysio &&
      _cbStorage &&
      _cbRights &&
      _cbVoluntary &&
      _cbLiability;

  bool get _canProceed => _hasScrolledToBottom && _allChecked;

  @override
  void initState() {
    super.initState();

    _scrollController.addListener(() {
      if (!_hasScrolledToBottom &&
          _scrollController.position.pixels >=
              _scrollController.position.maxScrollExtent - 60) {
        setState(() => _hasScrolledToBottom = true);
      }
    });

    if (widget.readOnly) {
      _loadPersistedConsent();
    }
  }

  /// In readOnly mode, populate all fields from the persisted consent record.
  Future<void> _loadPersistedConsent() async {
    final prefs = await SharedPreferences.getInstance();
    final String? ts = prefs.getString('consent_timestamp');
    if (!mounted) return;
    setState(() {
      _cbAge = prefs.getBool('consent_cb_age') ?? true;
      _cbPurpose = prefs.getBool('consent_cb_purpose') ?? true;
      _cbData = prefs.getBool('consent_cb_data') ?? true;
      _cbPhysio = prefs.getBool('consent_cb_physio') ?? true;
      _cbStorage = prefs.getBool('consent_cb_storage') ?? true;
      _cbRights = prefs.getBool('consent_cb_rights') ?? true;
      _cbVoluntary = prefs.getBool('consent_cb_voluntary') ?? true;
      _cbLiability = prefs.getBool('consent_cb_liability') ?? true;
      if (ts != null) {
        try {
          final dt = DateTime.parse(ts).toLocal();
          _consentTimestamp = DateFormat('dd MMM yyyy  HH:mm').format(dt);
        } catch (_) {
          _consentTimestamp = ts;
        }
      }
      // In read-only mode we always show the page as fully scrolled so the
      // timestamp footer is visible immediately.
      _hasScrolledToBottom = true;
    });
  }

  Future<void> _acceptConsent() async {
    final prefs = await SharedPreferences.getInstance();
    final String ts = DateTime.now().toIso8601String();
    await prefs.setBool('consent_accepted', true);
    await prefs.setString('consent_timestamp', ts);
    await prefs.setString('consent_version', ServiceConfig.consentVersion);
    await prefs.setBool('consent_cb_age', _cbAge);
    await prefs.setBool('consent_cb_purpose', _cbPurpose);
    await prefs.setBool('consent_cb_data', _cbData);
    await prefs.setBool('consent_cb_physio', _cbPhysio);
    await prefs.setBool('consent_cb_storage', _cbStorage);
    await prefs.setBool('consent_cb_rights', _cbRights);
    await prefs.setBool('consent_cb_voluntary', _cbVoluntary);
    await prefs.setBool('consent_cb_liability', _cbLiability);

    if (widget.onAccepted != null) {
      try {
        await widget.onAccepted!();
      } catch (error) {
        debugPrint('Could not resume services after consent: $error');
      }
    }

    if (mounted) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => widget.nextPage ?? const LoginPage()),
      );
    }
  }

  void _agreeToAllStatements() {
    if (widget.readOnly) return;
    setState(() {
      _cbAge = true;
      _cbPurpose = true;
      _cbData = true;
      _cbPhysio = true;
      _cbStorage = true;
      _cbRights = true;
      _cbVoluntary = true;
      _cbLiability = true;
    });
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  ColorScheme get _colors => Theme.of(context).colorScheme;

  // ─────────────────────────────────────────────────────────────────────────
  // BUILD
  // ─────────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: Theme.of(context).brightness == Brightness.dark
                ? const [Color(0xFF111218), Color(0xFF1A1B24)]
                : const [AppTheme.kBgTop, AppTheme.kBgBottom],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              _buildHeader(),
              Expanded(
                child: Container(
                  margin: const EdgeInsets.symmetric(horizontal: 14),
                  decoration: BoxDecoration(
                    color: _colors.surface,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: _colors.outlineVariant,
                      width: 1.5,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.06),
                        blurRadius: 18,
                        offset: const Offset(0, 6),
                      ),
                    ],
                  ),
                  child: Scrollbar(
                    controller: _scrollController,
                    thumbVisibility: true,
                    child: SingleChildScrollView(
                      controller: _scrollController,
                      padding: const EdgeInsets.fromLTRB(18, 22, 18, 8),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          _institutionBadge(),
                          const SizedBox(height: 18),

                          // readOnly banner
                          if (widget.readOnly) _readOnlyBanner(),

                          // ── Section 1 ─────────────────────────────────
                          _sectionTitle("1. About Aura and This Study"),
                          _paragraph(
                            "You are invited to consider joining research project R26-DS-012 at the "
                            "Sri Lanka Institute of Information Technology (SLIIT). Joining is voluntary. "
                            "Please read this information carefully before deciding.",
                          ),
                          _paragraph(
                            "The study examines whether wearable body signals, smartphone activity patterns, "
                            "and questionnaire answers can help estimate changes in anxiety "
                            "vulnerability among young adults aged 18 to 30. Aura may show short-term risk estimates "
                            "and reminders, but it is a research prototype, not a medical device or treatment service.",
                          ),
                          _divider(),

                          // ── Section 2 ─────────────────────────────────
                          _sectionTitle("2. Who Can Join and What You Do"),
                          _bulletItem(
                            "You must be 18 to 30 years old, able to give informed consent, and meet the recruitment criteria confirmed by the research team.",
                          ),
                          _bulletItem(
                            "You keep Aura installed on a compatible Android phone during your participation period. The final duration must match the ethics-approved research protocol.",
                          ),
                          _bulletItem(
                            "Three short mood check-ins each day, in the morning, afternoon, and evening. Each takes about 1 to 2 minutes.",
                          ),
                          _bulletItem(
                            "A 7-question anxiety check, called GAD-7, sent every week. It takes about 2 minutes.",
                          ),
                          _bulletItem(
                            "A 10-question stress check, called PSS-10, sent every week. It takes about 3 minutes.",
                          ),
                          _bulletItem(
                            "One set of questions about you, such as your age, education, and living situation, when you join.",
                          ),
                          _divider(),

                          // ── Section 3 ─────────────────────────────────
                          _sectionTitle("3. What Data Will Be Collected?"),
                          _paragraph(
                            "The table below lists every type of data collected. This includes "
                            "phone activity, such as movement and screen use, questionnaire answers, "
                            "and body readings from a wearable chest strap. Aura does not upload the "
                            "content of SMS messages or calls, contact names, phone numbers, your display "
                            "name, or your profile picture. App usage records do include individual app "
                            "package names and the foreground time recorded for each app.",
                          ),
                          _dataTable(),
                          _divider(),

                          // ── Section 4 ─────────────────────────────────
                          _sectionTitle(
                            "4. How Your Data Is Stored and Protected",
                          ),
                          _paragraph(
                            "Research data is sent using HTTPS and may be processed by the study's current "
                            "cloud services, including Google Sheets and Apps Script, Hugging Face Spaces, "
                            "and InfluxDB Cloud. These services may process or store data outside Sri Lanka.",
                          ),
                          _paragraph(
                            "Any processing outside Sri Lanka must follow the Sri Lanka Personal Data "
                            "Protection Act, No. 9 of 2022, as amended, including applicable requirements "
                            "for cross-border processing and the safeguards approved for the study.",
                          ),
                          _bulletItem(
                            "Aura does not upload your display name, phone number, or profile picture as research profile or sensor data. Contact details obtained separately for recruitment or a data-rights request must be kept separately with restricted access.",
                          ),
                          _bulletItem(
                            "All records are linked only to a randomly assigned Participant ID.",
                          ),
                          _bulletItem(
                            "Location records include precise GPS latitude and longitude, movement speed, and the phone's reported accuracy. These records must have restricted access because they can reveal sensitive movement patterns.",
                          ),
                          _bulletItem(
                            "App usage records include individual app package names and the foreground duration recorded for each app during the collection window.",
                          ),
                          _bulletItem(
                            "Access must be restricted to authorized researchers and service providers needed to run the study.",
                          ),
                          _bulletItem(
                            "Identifiable or pseudonymous data will be kept only for the retention period approved in the final research protocol, then deleted or irreversibly anonymized.",
                          ),
                          _divider(),

                          // ── Section 5 ─────────────────────────────────
                          _sectionTitle(
                            "5. Your Rights Over Your Data (PDPA No. 9 of 2022)",
                          ),
                          _paragraph(
                            "You have the following rights at any time:",
                          ),
                          _rightItem(
                            Icons.visibility_outlined,
                            "See Your Data",
                            "Ask for a full copy of all data collected about you.",
                          ),
                          _rightItem(
                            Icons.edit_outlined,
                            "Correct Your Data",
                            "Ask us to correct personal data that is wrong.",
                          ),
                          _rightItem(
                            Icons.delete_outline,
                            "Delete Your Data",
                            "Ask us to delete all of your data at any time without penalty.",
                          ),
                          _rightItem(
                            Icons.pause_circle_outline,
                            "Limit Data Use",
                            "Ask us to pause the use of your data while a complaint is being handled.",
                          ),
                          _rightItem(
                            Icons.exit_to_app_outlined,
                            "Leave the Study",
                            "Leave the study at any time without any negative consequences.",
                          ),
                          _paragraph(
                            "To exercise a right, use Your Data and Privacy inside Aura or email the research team at ${ServiceConfig.researchTeamEmail} with your Participant ID. Some requests may require identity verification.",
                          ),
                          _divider(),

                          // ── Section 6 ─────────────────────────────────
                          _sectionTitle("6. Possible Risks and Benefits"),
                          _subTitle("Possible risks"),
                          _bulletItem(
                            "Privacy or security incidents are possible whenever sensitive research data is collected or stored, although safeguards are used to reduce the risk.",
                          ),
                          _bulletItem(
                            "The app may use a little more battery, estimated at less than 5% per day.",
                          ),
                          _bulletItem(
                            "Some questions about mood or anxiety may feel upsetting. You do not have to answer any question that makes you uncomfortable.",
                          ),
                          _bulletItem(
                            "Sensor contact may feel uncomfortable, and movement, poor contact, illness, caffeine, or model error may cause false or missed anxiety estimates.",
                          ),
                          _subTitle("Possible benefits"),
                          _bulletItem(
                            "Your participation may help researchers understand how digital signals could support future anxiety monitoring.",
                          ),
                          _bulletItem(
                            "The findings may help create future anxiety checks that are carefully tested for health care use.",
                          ),
                          _bulletItem(
                            "You will not receive payment for joining.",
                          ),
                          _bulletItem(
                            "There is no guaranteed direct medical benefit to you.",
                          ),
                          _divider(),

                          // ── Section 7 ─────────────────────────────────
                          _sectionTitle(
                            "7. Joining Is Your Choice and You Can Leave",
                          ),
                          _paragraph(
                            "Joining is your choice. You may leave at any time without giving a reason and without "
                            "any negative consequences. To leave, uninstall the app and email the research team if "
                            "you also want your earlier data deleted.",
                          ),
                          _divider(),

                          // ── Section 8 ─────────────────────────────────
                          _sectionTitle("8. Research Status and Governance"),
                          _paragraph(
                            "Project ID: R26-DS-012. The study should follow the 2024 Declaration of Helsinki, "
                            "applicable research-ethics requirements, and the Sri Lanka PDPA No. 9 of 2022, as amended. "
                            "An ethics approval number must be confirmed and added before participant recruitment. "
                            "This screen does not itself prove ethics approval.",
                          ),
                          _divider(),

                          _sectionTitle("9. App Terms and Safety"),
                          _bulletItem(
                            "Aura is supplied for supervised research and testing. It is not an emergency, diagnosis, prevention, or treatment service.",
                          ),
                          _bulletItem(
                            "Do not delay professional or emergency help because of an Aura score, forecast, alert, or missing alert.",
                          ),
                          _bulletItem(
                            "Forecasts can change as new 60-second sensor windows arrive and may be unavailable when the phone, chest strap, network, or cloud service is offline.",
                          ),
                          _bulletItem(
                            "Use Aura lawfully, do not interfere with the service, and do not share another participant's identifier.",
                          ),
                          _bulletItem(
                            "The research team may update, suspend, or stop the prototype. Material changes to research data use require updated information and, where required, new consent.",
                          ),
                          _bulletItem(
                            "Nothing in these terms removes legal rights that cannot be excluded, including rights relating to privacy, data protection, or research-related harm.",
                          ),
                          _bulletItem(
                            "You may use Aura only for your own participation and as instructed by the research team. Do not copy, disrupt, probe, reverse engineer, or gain unauthorized access to the app or study systems except where applicable law expressly permits it.",
                          ),
                          _bulletItem(
                            "Aura software, content, and branding remain the property of their respective owners. You keep the rights and protections that apply to your personal data and research participation.",
                          ),
                          _bulletItem(
                            "The research team may restrict access where needed for safety, security, study completion, or misuse. This does not remove your right to withdraw or make a data-rights request.",
                          ),
                          _bulletItem(
                            "These terms are governed by applicable Sri Lankan law without limiting mandatory privacy, consumer, or research-participant protections.",
                          ),
                          _divider(),

                          // ── Section 10 — Declarations ─────────────────
                          _sectionTitle("10. Your Agreement"),
                          _paragraph(
                            widget.readOnly
                                ? "The following statements were confirmed when you agreed to join. "
                                      "The original record cannot be edited, but you can withdraw through Your Data and Privacy."
                                : "Please read each statement carefully and tick the box. "
                                      "Use Agree to all to select every statement at once, or select them individually.",
                          ),
                          const SizedBox(height: 6),

                          if (!widget.readOnly) _agreeToAllButton(),
                          if (!widget.readOnly) const SizedBox(height: 10),

                          _consentCheck(
                            value: _cbAge,
                            key: 'age',
                            label:
                                "I confirm that I am 18 to 30 years old and able to give informed consent to join this study.",
                          ),
                          _consentCheck(
                            value: _cbPurpose,
                            key: 'purpose',
                            label:
                                "I understand why this study is being done, what I need to do, and that I may stop participating at any time (Sections 1 & 2).",
                          ),
                          _consentCheck(
                            value: _cbData,
                            key: 'data',
                            label:
                                "I understand what data is collected from my phone, including precise GPS location, movement, individual app package names and usage duration, call and message totals, and my check-in answers, and I consent to this collection (Section 3).",
                          ),
                          _consentCheck(
                            value: _cbPhysio,
                            key: 'physio',
                            label:
                                "I consent to the collection of body readings from a wearable chest strap, including heart rate, breathing rate, body temperature, and movement, for anxiety monitoring and research (Section 3).",
                          ),
                          _consentCheck(
                            value: _cbStorage,
                            key: 'storage',
                            label:
                                "I understand that pseudonymous research data may be processed by Google, Hugging Face, and InfluxDB services outside Sri Lanka, subject to applicable safeguards (Section 4).",
                          ),
                          _consentCheck(
                            value: _cbRights,
                            key: 'rights',
                            label:
                                "I know my rights under PDPA No. 9 of 2022, including the right to see, correct, delete, limit, or withdraw my data at any time (Section 5).",
                          ),
                          _consentCheck(
                            value: _cbVoluntary,
                            key: 'voluntary',
                            label:
                                "I understand that joining is my choice and I may leave at any time without penalty (Section 7).",
                          ),
                          _consentCheck(
                            value: _cbLiability,
                            key: 'liability',
                            label:
                                "I understand that Aura is a research prototype, not medical care, and that I must not rely on a score, alert, or forecast in an emergency (Section 9).",
                          ),

                          const SizedBox(height: 14),

                          // Consent timestamp (readOnly only).
                          if (widget.readOnly && _consentTimestamp.isNotEmpty)
                            _timestampBadge(),

                          const SizedBox(height: 12),
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

  // ─────────────────────────────────────────────────────────────────────────
  // HEADER
  // ─────────────────────────────────────────────────────────────────────────

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(18, 14, 18, 10),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: _colors.primaryContainer.withValues(alpha: 0.6),
              borderRadius: BorderRadius.circular(11),
            ),
            child: Icon(Icons.gavel_rounded, color: _colors.primary, size: 26),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              widget.readOnly
                  ? "View Consent, Privacy & Terms"
                  : "Consent, Privacy & Terms",
              style: GoogleFonts.poppins(
                fontSize: 19,
                fontWeight: FontWeight.w700,
                color: _colors.onSurface,
              ),
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: _colors.tertiaryContainer.withValues(alpha: 0.65),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                color: _colors.tertiary.withValues(alpha: 0.55),
              ),
            ),
            child: Text(
              "R26-DS-012",
              style: TextStyle(
                fontSize: 11,
                color: _colors.onTertiaryContainer,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // FOOTER
  // ─────────────────────────────────────────────────────────────────────────

  Widget _buildFooter() {
    if (widget.readOnly) {
      // Read-only: show a plain Close / Back button.
      return Container(
        padding: const EdgeInsets.fromLTRB(14, 8, 14, 14),
        child: SizedBox(
          width: double.infinity,
          height: 50,
          child: OutlinedButton.icon(
            onPressed: () => Navigator.pop(context),
            icon: const Icon(Icons.close_rounded),
            label: const Text("Close"),
            style: OutlinedButton.styleFrom(
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(14),
              ),
            ),
          ),
        ),
      );
    }

    // First-run consent button.
    return Container(
      padding: const EdgeInsets.fromLTRB(14, 10, 14, 14),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (!_hasScrolledToBottom)
            _scrollHint()
          else if (!_allChecked)
            _checkboxHint(),
          const SizedBox(height: 10),
          SizedBox(
            width: double.infinity,
            height: 52,
            child: ElevatedButton(
              onPressed: _canProceed ? _acceptConsent : null,
              style: ElevatedButton.styleFrom(
                backgroundColor: _colors.primary,
                foregroundColor: _colors.onPrimary,
                disabledBackgroundColor: _colors.surfaceContainerHighest,
                disabledForegroundColor: _colors.onSurfaceVariant,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14),
                ),
                elevation: _canProceed ? 4 : 0,
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    _canProceed ? Icons.verified_outlined : Icons.lock_outline,
                    color: _canProceed
                        ? _colors.onPrimary
                        : _colors.onSurfaceVariant,
                    size: 20,
                  ),
                  const SizedBox(width: 10),
                  Text(
                    "I Consent & Continue",
                    style: GoogleFonts.poppins(
                      fontWeight: FontWeight.w600,
                      fontSize: 15,
                      color: _canProceed
                          ? _colors.onPrimary
                          : _colors.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 5),
          Text(
            "Consent timestamp is recorded automatically upon confirmation.",
            style: TextStyle(fontSize: 10, color: _colors.onSurfaceVariant),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CHECKBOX — interactive (first run) or locked (read-only)
  // ─────────────────────────────────────────────────────────────────────────

  Widget _agreeToAllButton() {
    return SizedBox(
      width: double.infinity,
      child: OutlinedButton.icon(
        onPressed: _allChecked ? null : _agreeToAllStatements,
        icon: Icon(
          _allChecked ? Icons.check_circle_rounded : Icons.done_all_rounded,
        ),
        label: Text(
          _allChecked ? 'All statements selected' : 'Agree to all statements',
        ),
        style: OutlinedButton.styleFrom(
          foregroundColor: _colors.primary,
          side: BorderSide(
            color: _allChecked ? _colors.tertiary : _colors.primary,
          ),
          padding: const EdgeInsets.symmetric(vertical: 13),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
      ),
    );
  }

  Widget _consentCheck({
    required bool value,
    required String key,
    required String label,
  }) {
    final bool locked = widget.readOnly;

    // In first-run mode wire up the setter.
    void onChange(bool? v) {
      if (locked) return; // no-op when locked
      setState(() {
        switch (key) {
          case 'age':
            _cbAge = v ?? false;
            break;
          case 'purpose':
            _cbPurpose = v ?? false;
            break;
          case 'data':
            _cbData = v ?? false;
            break;
          case 'physio':
            _cbPhysio = v ?? false;
            break;
          case 'storage':
            _cbStorage = v ?? false;
            break;
          case 'rights':
            _cbRights = v ?? false;
            break;
          case 'voluntary':
            _cbVoluntary = v ?? false;
            break;
          case 'liability':
            _cbLiability = v ?? false;
            break;
        }
      });
    }

    return GestureDetector(
      onTap: locked ? null : () => onChange(!value),
      child: Container(
        margin: const EdgeInsets.only(bottom: 9),
        padding: const EdgeInsets.all(11),
        decoration: BoxDecoration(
          // Ticked + locked → green tint. Ticked + interactive → purple tint.
          color: value
              ? (locked
                    ? _colors.tertiaryContainer.withValues(alpha: 0.45)
                    : _colors.primaryContainer.withValues(alpha: 0.45))
              : _colors.surfaceContainerLow,
          borderRadius: BorderRadius.circular(11),
          border: Border.all(
            color: value
                ? (locked
                      ? _colors.tertiary
                      : _colors.primary.withValues(alpha: 0.7))
                : _colors.outlineVariant,
            width: value ? 1.5 : 1.0,
          ),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: 22,
              height: 22,
              child: Checkbox(
                value: value,
                onChanged: locked ? null : onChange,
                activeColor: locked ? _colors.tertiary : _colors.primary,
                // Keep the checkmark visible even when disabled.
                fillColor: locked && value
                    ? WidgetStateProperty.all(_colors.tertiary)
                    : null,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(4),
                ),
                materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                visualDensity: VisualDensity.compact,
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    child: Text(
                      label,
                      style: TextStyle(
                        fontSize: 12.5,
                        height: 1.45,
                        color: value
                            ? _colors.onSurface
                            : _colors.onSurfaceVariant,
                        fontWeight: value ? FontWeight.w500 : FontWeight.normal,
                      ),
                    ),
                  ),
                  if (locked && value) ...[
                    const SizedBox(width: 6),
                    Icon(Icons.lock_outline, size: 13, color: _colors.tertiary),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // READ-ONLY WIDGETS
  // ─────────────────────────────────────────────────────────────────────────

  Widget _readOnlyBanner() {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: _colors.secondaryContainer.withValues(alpha: 0.65),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _colors.secondary.withValues(alpha: 0.55)),
      ),
      child: Row(
        children: [
          Icon(Icons.info_outline_rounded, color: _colors.secondary, size: 18),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              "You are viewing the consent form you agreed to when joining the study. "
              "The original agreement is recorded, and you may withdraw through Your Data and Privacy.",
              style: TextStyle(
                fontSize: 12,
                color: _colors.onSecondaryContainer,
                height: 1.45,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _timestampBadge() {
    return Container(
      margin: const EdgeInsets.only(top: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: _colors.tertiaryContainer.withValues(alpha: 0.65),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _colors.tertiary.withValues(alpha: 0.55)),
      ),
      child: Row(
        children: [
          Icon(Icons.verified_outlined, color: _colors.tertiary, size: 18),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Agreement Confirmed",
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: _colors.onTertiaryContainer,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  "Date & time: $_consentTimestamp",
                  style: TextStyle(
                    fontSize: 11.5,
                    color: _colors.onTertiaryContainer,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // HINT BANNERS (first-run only)
  // ─────────────────────────────────────────────────────────────────────────

  Widget _scrollHint() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 12),
      decoration: BoxDecoration(
        color: _colors.tertiaryContainer.withValues(alpha: 0.65),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: _colors.tertiary.withValues(alpha: 0.55)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.touch_app_rounded, color: _colors.tertiary, size: 17),
          const SizedBox(width: 6),
          Text(
            "Scroll to the bottom to read the full consent form",
            style: TextStyle(
              fontSize: 11,
              color: _colors.onTertiaryContainer,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _checkboxHint() {
    final pending = [
      if (!_cbAge) "Confirm age 18 to 30",
      if (!_cbPurpose) "Why the study is being done and what you need to do",
      if (!_cbData) "Information the app collects",
      if (!_cbPhysio) "Body readings from the chest strap",
      if (!_cbStorage) "Data stored outside Sri Lanka",
      if (!_cbRights) "Your data rights",
      if (!_cbVoluntary) "Joining is your choice",
      if (!_cbLiability) "Research use only and responsibility",
    ];
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(11),
      decoration: BoxDecoration(
        color: _colors.tertiaryContainer.withValues(alpha: 0.65),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: _colors.tertiary.withValues(alpha: 0.55)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            "Please tick all boxes to continue:",
            style: TextStyle(
              fontSize: 11.5,
              fontWeight: FontWeight.w600,
              color: _colors.onTertiaryContainer,
            ),
          ),
          const SizedBox(height: 4),
          ...pending.map(
            (p) => Text(
              "• $p",
              style: TextStyle(
                fontSize: 11,
                color: _colors.onTertiaryContainer,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CONTENT WIDGETS
  // ─────────────────────────────────────────────────────────────────────────

  Widget _institutionBadge() {
    return Container(
      padding: const EdgeInsets.all(13),
      decoration: BoxDecoration(
        color: _colors.primaryContainer.withValues(alpha: 0.45),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _colors.primary.withValues(alpha: 0.35)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.school_outlined, color: _colors.primary, size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Sri Lanka Institute of Information Technology (SLIIT)",
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: _colors.onPrimaryContainer,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  "Faculty of Computing  |  Research Group R26-DS-012\n"
                  "Supervisor: ${ServiceConfig.supervisorName}\n"
                  "Research contact: ${ServiceConfig.researchTeamEmail}\n"
                  "Ethics approval details must be confirmed before participant recruitment.",
                  style: TextStyle(
                    fontSize: 11,
                    color: _colors.onSurfaceVariant,
                    height: 1.5,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _dataTable() {
    const rows = [
      [
        "General Location",
        "Precise GPS latitude and longitude, movement speed, and the phone's reported location accuracy.",
        "Every 15 min",
      ],
      [
        "Screen Use",
        "When the screen is turned on, turned off, or unlocked. Screen content is never saved.",
        "As it happens",
      ],
      [
        "Phone Movement",
        "The event time and movement magnitude when the phone crosses the high-movement threshold.",
        "As it happens",
      ],
      [
        "Call Totals",
        "Numbers of incoming, outgoing, missed, and rejected calls. No phone numbers or call content.",
        "Every 15 min",
      ],
      [
        "Message Totals",
        "Numbers of sent and received text messages. No message content.",
        "Every 15 min",
      ],
      [
        "App Use",
        "Individual app package names and the foreground duration recorded for each app.",
        "Every 15 min",
      ],
      [
        "Battery",
        "Battery percentage and whether the phone is charging.",
        "Every 15 min",
      ],
      [
        "Service Status",
        "Heartbeat, restart, battery-warning, sync-request, and error records used to check whether collection is working.",
        "When generated",
      ],
      [
        "Daily Check-ins",
        "Stress, anxiety, tiredness, social feelings, and current activity.",
        "3 times daily",
      ],
      [
        "Weekly Anxiety Check",
        "The 7 GAD-7 answers and total score.",
        "Weekly",
      ],
      [
        "Weekly Stress Check",
        "The 10 PSS-10 answers and total score.",
        "Weekly",
      ],
      [
        "About You",
        "Age, gender, education, work, whether a health professional has told you that you have anxiety, and sleep quality.",
        "When you join",
      ],
      [
        "Heart Rate",
        "Heart rate and heart-rate-variability summaries from the wearable chest strap.",
        "About every 60 sec",
      ],
      [
        "Breathing Rate",
        "Breathing-rate summaries from the chest strap.",
        "About every 60 sec",
      ],
      [
        "Body Temperature",
        "Skin-temperature summaries where the chest strap touches your body.",
        "About every 60 sec",
      ],
      [
        "Chest Strap Movement",
        "Movement summaries from the chest strap.",
        "About every 60 sec",
      ],
      [
        "Forecasts and Feedback",
        "Model risk values, forecast trajectories, alerts, your ratings, and follow-up answers.",
        "When generated",
      ],
      [
        "Local Personalization",
        "Your display name and optional profile picture stay on this phone and are not uploaded as research data.",
        "When changed",
      ],
    ];

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        border: Border.all(color: _colors.outlineVariant),
        borderRadius: BorderRadius.circular(11),
      ),
      child: Column(
        children: [
          Container(
            decoration: BoxDecoration(
              color: _colors.primaryContainer.withValues(alpha: 0.55),
              borderRadius: const BorderRadius.vertical(
                top: Radius.circular(10),
              ),
            ),
            child: _tRow(
              "Information",
              "What Is Stored",
              "How Often",
              header: true,
            ),
          ),
          ...rows.asMap().entries.map((e) {
            final last = e.key == rows.length - 1;
            return Container(
              decoration: BoxDecoration(
                color: e.key.isEven
                    ? Theme.of(context).colorScheme.surface
                    : Theme.of(context).colorScheme.surfaceContainerHighest,
                borderRadius: last
                    ? const BorderRadius.vertical(bottom: Radius.circular(10))
                    : null,
                border: Border(top: BorderSide(color: _colors.outlineVariant)),
              ),
              child: _tRow(e.value[0], e.value[1], e.value[2]),
            );
          }),
        ],
      ),
    );
  }

  Widget _tRow(String a, String b, String c, {bool header = false}) {
    final s = TextStyle(
      fontSize: 11,
      fontWeight: header ? FontWeight.w700 : FontWeight.normal,
      color: header ? _colors.onPrimaryContainer : _colors.onSurface,
      height: 1.4,
    );
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(flex: 3, child: Text(a, style: s)),
          const SizedBox(width: 6),
          Expanded(flex: 5, child: Text(b, style: s)),
          const SizedBox(width: 6),
          Expanded(flex: 2, child: Text(c, style: s)),
        ],
      ),
    );
  }

  Widget _rightItem(IconData icon, String title, String desc) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 9),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(5),
            decoration: BoxDecoration(
              color: _colors.primaryContainer.withValues(alpha: 0.55),
              borderRadius: BorderRadius.circular(7),
            ),
            child: Icon(icon, size: 15, color: _colors.primary),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 1),
                Text(
                  desc,
                  style: TextStyle(
                    fontSize: 12,
                    color: _colors.onSurfaceVariant,
                    height: 1.4,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ── TYPOGRAPHY ────────────────────────────────────────────────────────────

  Widget _sectionTitle(String t) => Padding(
    padding: const EdgeInsets.only(bottom: 9, top: 2),
    child: Text(
      t,
      style: GoogleFonts.poppins(
        fontSize: 14,
        fontWeight: FontWeight.w700,
        color: _colors.primary,
      ),
    ),
  );

  Widget _subTitle(String t) => Padding(
    padding: const EdgeInsets.only(bottom: 5, top: 6),
    child: Text(
      t,
      style: TextStyle(
        fontSize: 13,
        fontWeight: FontWeight.w700,
        color: _colors.onSurface,
      ),
    ),
  );

  Widget _paragraph(String t) => Padding(
    padding: const EdgeInsets.only(bottom: 9),
    child: Text(
      t,
      style: TextStyle(
        fontSize: 12.5,
        height: 1.55,
        color: _colors.onSurfaceVariant,
      ),
    ),
  );

  Widget _bulletItem(String t) => Padding(
    padding: const EdgeInsets.only(bottom: 5, left: 4),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(top: 5.5),
          child: Container(
            width: 5,
            height: 5,
            decoration: BoxDecoration(
              color: _colors.primary,
              shape: BoxShape.circle,
            ),
          ),
        ),
        const SizedBox(width: 9),
        Expanded(
          child: Text(
            t,
            style: TextStyle(
              fontSize: 12.5,
              height: 1.5,
              color: _colors.onSurfaceVariant,
            ),
          ),
        ),
      ],
    ),
  );

  Widget _divider() => Padding(
    padding: const EdgeInsets.symmetric(vertical: 14),
    child: Divider(height: 1, color: _colors.outlineVariant),
  );
}
