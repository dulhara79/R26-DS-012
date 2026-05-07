import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../theme/app_theme.dart';
import 'login_page.dart';

class InformedConsentPage extends StatefulWidget {
  const InformedConsentPage({super.key});

  @override
  State<InformedConsentPage> createState() => _InformedConsentPageState();
}

class _InformedConsentPageState extends State<InformedConsentPage> {
  bool _hasScrolledToBottom = false;
  final ScrollController _scrollController = ScrollController();

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
    await prefs.setBool('consent_accepted', true);
    await prefs.setString('consent_timestamp', DateTime.now().toIso8601String());

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
                  margin: const EdgeInsets.symmetric(horizontal: 20),
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.9),
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
                          _sectionTitle("Informed Consent for Research"),
                          _paragraph(
                            "You are invited to participate in a research study exploring digital biomarkers of anxiety. This study is conducted in accordance with the Sri Lanka Personal Data Protection Act (PDPA) No. 9 of 2022.",
                          ),
                          _divider(),
                          _subTitle("1. What data do we collect?"),
                          _bulletItem("Clinical Surveys: GAD-7 and Daily Mood ratings."),
                          _bulletItem("Sensor Data: Screen on/off times and significant motion."),
                          _bulletItem("Digital Phenotyping: Frequency and counts of SMS/Calls (no message content is read)."),
                          _bulletItem("Usage Stats: Time spent on different mobile applications."),
                          _bulletItem("Location: Periodic GPS tracking to understand mobility patterns."),
                          _divider(),
                          _subTitle("2. Where is my data stored?"),
                          _paragraph(
                            "Your data is transmitted securely via HTTPS and stored in Google Cloud servers. By clicking 'I Consent', you explicitly acknowledge that your data will be stored on overseas infrastructure, as permitted under the PDPA for scientific research.",
                          ),
                          _divider(),
                          _subTitle("3. Your Privacy Rights"),
                          _paragraph(
                            "All data is associated with a Participant ID. We do not store your real name or phone number on our research servers. You have the right to:",
                          ),
                          _bulletItem("Request a copy of your collected data."),
                          _bulletItem("Withdraw from the study at any time by emailing the research team."),
                          _bulletItem("Request complete deletion of your data via email."),
                          _divider(),
                          _subTitle("4. Voluntary Participation"),
                          _paragraph(
                            "Participation is 100% voluntary. You may stop using the app at any time without penalty.",
                          ),
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

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Row(
        children: [
          const Icon(Icons.gavel_rounded, color: AppTheme.kPrimaryDeep, size: 32),
          const SizedBox(width: 16),
          Expanded(
            child: Text(
              "Ethical Approval & Consent",
              style: GoogleFonts.poppins(
                fontSize: 20,
                fontWeight: FontWeight.w600,
                color: AppTheme.kTextDark,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFooter() {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        children: [
          if (!_hasScrolledToBottom)
            Text(
              "Please scroll to the bottom to accept",
              style: TextStyle(color: Colors.grey.shade600, fontSize: 12),
            ),
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            height: 56,
            child: ElevatedButton(
              onPressed: _hasScrolledToBottom ? _acceptConsent : null,
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

  Widget _sectionTitle(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Text(
        text,
        style: GoogleFonts.poppins(
          fontSize: 18,
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
      padding: const EdgeInsets.only(bottom: 12),
      child: Text(
        text,
        style: TextStyle(
          fontSize: 14,
          height: 1.5,
          color: Colors.grey.shade800,
        ),
      ),
    );
  }

  Widget _bulletItem(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6, left: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.only(top: 6),
            child: Icon(Icons.circle, size: 6, color: AppTheme.kAccentBlue),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              text,
              style: TextStyle(fontSize: 14, color: Colors.grey.shade800),
            ),
          ),
        ],
      ),
    );
  }

  Widget _divider() {
    return Divider(height: 32, color: Colors.grey.shade200);
  }
}
