import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';
import '../services/background/service_config.dart';

class PrivacyPolicyPage extends StatelessWidget {
  const PrivacyPolicyPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.kBgTop,
      appBar: AppBar(
        title: const Text('Privacy Policy'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: Center(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: AppTheme.kPrimaryDeep.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                  'v${ServiceConfig.consentVersion}',
                  style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: AppTheme.kPrimaryDeep),
                ),
              ),
            ),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          _header(),
          const SizedBox(height: 20),
          _section('1. Data Controller', 
            '${ServiceConfig.dataController}\n'
            'Principal Investigator: ${ServiceConfig.piName}\n'
            'Email: ${ServiceConfig.piEmail}\n\n'
            'The Data Controller is responsible for ensuring that your personal '
            'data is processed in accordance with the Sri Lanka Personal Data '
            'Protection Act (PDPA), No. 9 of 2022.',
          ),
          _section('2. Lawful Basis for Processing',
            'We process your personal data under the following lawful bases as '
            'defined by the PDPA:\n\n'
            '• Explicit Consent: You have provided informed consent through the '
            'in-app consent process.\n'
            '• Scientific Research: Processing is necessary for scientific research '
            'purposes, carried out in the public interest with appropriate safeguards.\n\n'
            'Ethics Approval: ${ServiceConfig.ercApprovalNumber}\n'
            'Approved by: ${ServiceConfig.ercName}',
          ),
          _section('3. Personal Data We Collect',
            'We collect the following categories of data:\n\n'
            'Clinical Data:\n'
            '• GAD-7 anxiety assessment scores (weekly)\n'
            '• PSS-10 perceived stress scores (weekly)\n'
            '• EMA mood ratings (3x daily)\n'
            '• Touch pressure interactions\n\n'
            'Sensor Data:\n'
            '• Screen on/off/unlock events\n'
            '• Significant motion events (accelerometer)\n'
            '• Battery level and charging state\n\n'
            'Communication Patterns (counts only):\n'
            '• Number of incoming/outgoing/missed calls\n'
            '• Number of sent/received SMS messages\n'
            '(No names, numbers, or content are collected)\n\n'
            'App Usage:\n'
            '• Time spent in app categories (Social Media, Browser, etc.)\n'
            '(Individual app names are replaced with categories)\n\n'
            'Location:\n'
            '• Periodic GPS coordinates fuzzed to ±1 km precision\n'
            '(Exact location is never stored)\n\n'
            'Demographics (collected once):\n'
            '• Age, gender, marital status, employment, education\n'
            '• Living situation, anxiety diagnosis history, medication status\n'
            '• Self-reported sleep quality',
          ),
          _section('4. Data We Do NOT Collect',
            '• Your real name, phone number, or email address\n'
            '• SMS or call content\n'
            '• Contact lists or address books\n'
            '• Photos, videos, or camera data\n'
            '• Browsing history or search queries\n'
            '• Passwords or financial information\n'
            '• Social media account details',
          ),
          _section('5. How We Use Your Data',
            'Your data is used exclusively for:\n\n'
            '• Identifying digital behavioural patterns associated with anxiety\n'
            '• Developing predictive models for anxiety detection\n'
            '• Publishing anonymised, aggregated research findings\n'
            '• Improving digital mental health interventions\n\n'
            'We do NOT use your data for:\n'
            '• Marketing or advertising\n'
            '• Selling to third parties\n'
            '• Individual profiling or automated decision-making\n'
            '• Clinical diagnosis or treatment recommendations',
          ),
          _section('6. Data Storage & Security',
            'Storage: Data is stored in Google Cloud infrastructure (Google Sheets '
            'via Google Apps Script), encrypted in transit via HTTPS/TLS.\n\n'
            'Pseudonymisation: All data is linked only to your Participant ID. '
            'Your real identity is not stored on research servers.\n\n'
            'Access Control: Data access is restricted to authorised members of '
            'the research team (${ServiceConfig.piAffiliation}).\n\n'
            'Privacy Measures:\n'
            '• GPS coordinates fuzzed to ±1 km\n'
            '• App names replaced with categories\n'
            '• Communication data limited to counts\n'
            '• No message content or contact names stored',
          ),
          _section('7. Cross-Border Data Transfer',
            'Your pseudonymised data may be transferred to and stored on servers '
            'located outside Sri Lanka (Google Cloud infrastructure).\n\n'
            'This transfer is conducted with appropriate safeguards as required '
            'by Section 25 of the PDPA, including:\n\n'
            '• Data pseudonymisation before transfer\n'
            '• HTTPS encryption for all data transmissions\n'
            '• Access restricted to authorised researchers\n'
            '• Your explicit consent for cross-border transfer',
          ),
          _section('8. Data Retention',
            'Your data will be retained for ${ServiceConfig.dataRetentionPeriod}.\n\n'
            'After this period, all data associated with your Participant ID '
            'will be permanently deleted from all storage systems.\n\n'
            'Anonymised, aggregated data (which cannot be linked back to you) '
            'may be retained indefinitely for future research reference.',
          ),
          _section('9. Your Rights',
            'Under the PDPA, you have the following rights:\n\n'
            'Right to Access (Section 17): Request a copy of your data.\n\n'
            'Right to Rectification (Section 18): Request correction of '
            'inaccurate data.\n\n'
            'Right to Erasure (Section 19): Request deletion of your data.\n\n'
            'Right to Withdraw Consent (Section 5): Withdraw your consent '
            'at any time without giving reasons.\n\n'
            'Right to Object (Section 20): Object to specific data processing.\n\n'
            'Right to Complain: Lodge a complaint with the Data Protection '
            'Authority of Sri Lanka.\n\n'
            'Response Time: We will respond to your request within 21 business '
            'days as required by the PDPA.\n\n'
            'To exercise these rights, use the Data Rights section in the app '
            'or contact: ${ServiceConfig.researchTeamEmail}',
          ),
          _section('10. Data Sharing',
            'We may share your data only in the following circumstances:\n\n'
            '• Anonymised, aggregated findings in academic publications\n'
            '• With the Ethics Review Committee for audit purposes\n'
            '• If required by law or court order\n\n'
            'We will NEVER sell your data to third parties or share identifiable '
            'data outside the research team without your explicit consent.',
          ),
          _section('11. Changes to This Policy',
            'We may update this privacy policy to reflect changes in our '
            'practices or legal requirements. The version number and date '
            'at the top of this document indicate the latest revision.\n\n'
            'Material changes will be communicated through in-app notification.\n\n'
            'Current Version: ${ServiceConfig.consentVersion}\n'
            'Last Updated: ${ServiceConfig.consentDate}',
          ),
          _section('12. Contact Information',
            'Principal Investigator:\n'
            '${ServiceConfig.piName}\n'
            '${ServiceConfig.piEmail}\n\n'
            'Research Supervisor:\n'
            '${ServiceConfig.supervisorName}\n'
            '${ServiceConfig.supervisorEmail}\n\n'
            'Ethics Review Committee:\n'
            '${ServiceConfig.ercName}\n'
            '${ServiceConfig.ercSecretaryEmail}\n\n'
            'Data Controller:\n'
            '${ServiceConfig.dataController}\n'
            '${ServiceConfig.researchTeamEmail}',
          ),
          const SizedBox(height: 40),
        ],
      ),
    );
  }

  Widget _header() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.kPrimaryDeep.withValues(alpha: 0.06),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Privacy Policy',
            style: GoogleFonts.poppins(fontSize: 20, fontWeight: FontWeight.bold, color: AppTheme.kPrimaryDeep),
          ),
          const SizedBox(height: 4),
          Text(ServiceConfig.studyTitle, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          Text(
            'This privacy policy explains how we collect, use, store, and protect '
            'your personal data in compliance with the Sri Lanka Personal Data '
            'Protection Act (PDPA), No. 9 of 2022.',
            style: TextStyle(fontSize: 13, color: Colors.grey.shade700, height: 1.5),
          ),
        ],
      ),
    );
  }

  Widget _section(String title, String content) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.black87)),
          const SizedBox(height: 8),
          Text(content, style: TextStyle(fontSize: 13, height: 1.6, color: Colors.grey.shade800)),
        ],
      ),
    );
  }
}
