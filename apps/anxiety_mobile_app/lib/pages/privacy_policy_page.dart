import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../services/background/service_config.dart';

class PrivacyPolicyPage extends StatelessWidget {
  const PrivacyPolicyPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: AppBar(
        title: const Text('Privacy Policy'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: Center(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: Theme.of(
                    context,
                  ).colorScheme.primaryContainer.withValues(alpha: 0.6),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                  'v${ServiceConfig.consentVersion}',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    color: Theme.of(context).colorScheme.onPrimaryContainer,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          _header(context),
          const SizedBox(height: 20),
          _section(
            '1. Who Is Responsible for Your Data',
            '${ServiceConfig.dataController}\n'
                'Project: R26-DS-012\n'
                'Research supervisor: ${ServiceConfig.supervisorName}\n'
                'Email: ${ServiceConfig.supervisorEmail}\n\n'
                'The final data-controller and ethics-approval details must be confirmed in the approved participant information sheet before recruitment.',
          ),
          _section(
            '2. Purpose and Legal Basis',
            'Aura processes pseudonymous research data to study personalized anxiety vulnerability and short-term escalation forecasting. Health-related data and other personal data are processed on the basis described in the final approved research protocol and your documented consent where consent is relied on.\n\n'
                'Withdrawing consent does not make earlier lawful processing unlawful, but it stops future consent-based processing after the request is applied.\n\n'
                'Ethics approval recorded in this app: ${ServiceConfig.ercApprovalNumber}',
          ),
          _section(
            '3. Information We Collect',
            'Research identity and profile:\n'
                '• Random Participant ID\n'
                '• Age, gender, marital status, employment, financial status, education, living situation, anxiety diagnosis, medication status, and sleep quality\n'
                '• Your display name and optional profile picture stay locally on the phone and are not uploaded as research data\n\n'
                'Check-in answers:\n'
                '• GAD-7 weekly anxiety check answers and score\n'
                '• PSS-10 weekly stress check answers and score\n'
                '• Short mood check-ins three times a day\n\n'
                'Phone activity:\n'
                '• When the screen is turned on, turned off, or unlocked\n'
                '• High-movement event times and movement magnitude\n'
                '• Battery level and charging state\n'
                '• Precise GPS latitude and longitude, movement speed, and the phone\'s reported location accuracy about every 15 minutes\n'
                '• Individual app package names and the foreground duration recorded for each app about every 15 minutes\n\n'
                'Communication totals:\n'
                '• Number of incoming/outgoing/missed calls\n'
                '• Number of sent/received SMS messages\n'
                '• No names, numbers, message text, or call content\n\n'
                'Wearable and model data:\n'
                '• Heart-rate, heart-rate-variability, breathing, skin-temperature, and movement summaries\n'
                '• Calibration values, risk estimates, 10-minute forecast trajectories, alerts, ratings, and follow-up feedback\n\n'
                'Service and rights-request records:\n'
                '• Heartbeat, restart, battery-warning, sync, and error records used to check data collection\n'
                '• Requests to access, delete, or withdraw data and the Participant ID used for the request',
          ),
          _section(
            '4. Data We Do NOT Collect',
            '• Your display name or profile picture as uploaded research data\n'
                '• Your phone number or email address through normal sensing\n'
                '• SMS or call content\n'
                '• Contact lists or address books\n'
                '• Uploaded photos, videos, microphone, or camera data\n'
                '• Browsing history or search queries\n'
                '• Passwords or financial information\n'
                '• Social media account details',
          ),
          _section(
            '5. How We Use Your Data',
            'Your data is used to:\n\n'
                '• Study wearable, phone-use, questionnaire, and related clinical patterns that may be associated with anxiety vulnerability\n'
                '• Build and evaluate personalized models and short-term forecasts\n'
                '• Show research estimates, reminders, alerts, and check-ins inside Aura\n'
                '• Publish research findings that group many people together and do not identify you\n'
                '• Test and improve the research system\n\n'
                'We do NOT use your data for:\n'
                '• Marketing or advertising\n'
                '• Selling to third parties\n'
                '• Automated decisions that determine legal rights, education, employment, insurance, or access to care\n'
                '• Diagnosing or treating a health condition',
          ),
          _section(
            '6. Data Storage & Security',
            'Behavioral records, questionnaire answers, service records, data-rights requests, and chest-strap vital summaries are currently sent through Google Apps Script to Google Sheets. Physiological windows, calibration, forecasts, and feedback are also processed through Hugging Face Spaces and InfluxDB Cloud. Some pending records are temporarily held on the phone until they can be uploaded.\n\n'
                'Research traffic uses HTTPS. Access controls, pseudonymous Participant IDs, and communication counts are used to reduce privacy risk. Precise location and package-level app usage are sensitive and must be available only to authorized researchers and approved processors. No system can promise perfect security.\n\n'
                'Identity protection: All data is linked only to your Participant ID. '
                'Your local display name and profile picture are not attached to uploaded research records.\n\n'
                'Access must be limited to authorized researchers, ethics or institutional reviewers where required, and service providers needed to operate the study.',
          ),
          _section(
            '7. Data Stored Outside Sri Lanka',
            'Google, Hugging Face, and InfluxDB services may process or store pseudonymous research data outside Sri Lanka. Cross-border processing must follow the Sri Lanka PDPA No. 9 of 2022, as amended, and any applicable approval, contractual, security, and consent requirements.\n\n'
                'The final participant information sheet must identify the approved processors and safeguards before recruitment.',
          ),
          _section(
            '8. How Long We Keep Your Data',
            'Pseudonymous research data will be kept for ${ServiceConfig.dataRetentionPeriod}. That period must be replaced with a definite, approved duration before recruitment.\n\n'
                'When the approved period ends, data linked to your Participant ID will be deleted or irreversibly anonymized unless law or an ethics-approved protocol requires otherwise. Truly anonymous grouped findings may be retained because they can no longer be linked to you.',
          ),
          _section(
            '9. Your Rights',
            'Subject to applicable law and research requirements, you may ask to access, correct, erase, or restrict processing of your personal data; withdraw consent where processing relies on consent; leave the study without penalty; and complain or appeal to the relevant authority.\n\n'
                'Withdrawal does not affect processing that was lawful before withdrawal. A request may require verification of your Participant ID and will be handled within the period required by law.\n\n'
                'To use these rights, open Your Data and Privacy in the app '
                'or contact: ${ServiceConfig.researchTeamEmail}',
          ),
          _section(
            '10. Data Sharing',
            'Data may be disclosed only as needed for:\n\n'
                '• Google, Hugging Face, and InfluxDB services that process or store the study data\n'
                '• Authorized researchers and institutional or ethics oversight\n'
                '• Research reports containing grouped or anonymous findings\n'
                '• If required by law or court order\n\n'
                'The research team will not sell participant data or use it for advertising.',
          ),
          _section(
            '11. Changes to This Policy',
            'We may update this privacy policy to reflect changes in our '
                'practices or legal requirements. The version number and date '
                'at the top of this document indicate the latest revision.\n\n'
                'Material changes to the research purpose, data types, processors, retention, or risk will be communicated. New consent will be requested where required before the changed processing begins.\n\n'
                'Current Version: ${ServiceConfig.consentVersion}\n'
                'Last Updated: ${ServiceConfig.consentDate}',
          ),
          _section(
            '12. Contact Information',
            'Research supervisor:\n'
                '${ServiceConfig.piName}\n'
                '${ServiceConfig.piEmail}\n\n'
                'Research team:\n'
                '${ServiceConfig.researchTeamEmail}\n\n'
                'Ethics Review Committee:\n'
                '${ServiceConfig.ercName}\n'
                '${ServiceConfig.ercSecretaryEmail}\n\n'
                'Data Controller:\n'
                '${ServiceConfig.dataController}',
          ),
          const SizedBox(height: 40),
        ],
      ),
    );
  }

  Widget _header(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(
          context,
        ).colorScheme.primaryContainer.withValues(alpha: 0.45),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Privacy Policy',
            style: GoogleFonts.poppins(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: Theme.of(context).colorScheme.onPrimaryContainer,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            ServiceConfig.studyTitle,
            style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 8),
          Text(
            'This privacy policy explains how we collect, use, store, and protect '
            'your personal data in compliance with the Sri Lanka Personal Data '
            'Protection Act (PDPA), No. 9 of 2022, as amended. This version is for the research prototype and must be finalized against the approved protocol before participant recruitment.',
            style: TextStyle(
              fontSize: 13,
              color: Theme.of(context).colorScheme.onSurfaceVariant,
              height: 1.5,
            ),
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
          Text(
            title,
            style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          Text(content, style: const TextStyle(fontSize: 13, height: 1.6)),
        ],
      ),
    );
  }
}
