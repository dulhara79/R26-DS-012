// lib/features/consent/agreement_text.dart
//
// ─────────────────────────────────────────────────────────────────────────────
// BEFORE DEPLOYMENT
//
// Every [BRACKETED] item must be completed, and the finished text must be
// reviewed and signed off by:
//   • the SLIIT Faculty of Computing legal/compliance contact,
//   • the NHSL Ethics Review Committee, and
//   • whoever is named as Data Protection Officer below.
//
// This is a competent working draft, not legal advice, and it has not been
// reviewed by a lawyer. Do not deploy it unreviewed to a hospital.
//
// If you change ANY character of the text below, bump `kAgreementVersion` in
// lib/domain/consent.dart. The stored SHA-256 anchor depends on it, and every
// user will be required to accept again.
// ─────────────────────────────────────────────────────────────────────────────

const String kTermsOfUse = '''
TERMS OF USE — ClinAnx

Version $kAgreementVersionInline · Effective $kEffectiveDateInline

These Terms form a binding agreement between you and the Providers. Read them
before you tap "I agree". If you do not accept them, do not use this software.

1. WHAT ClinAnx IS

1.1  ClinAnx is research software developed for study R26-DS-012, "A Multimodal
     Digital Biomarker Framework for Personalized Vulnerability Mapping and
     Acute Escalation Forecasting in Young Adults with Anxiety Disorders".

1.2  It is provided by the Sri Lanka Institute of Information Technology
     ("SLIIT") together with the National Hospital of Sri Lanka ("NHSL")
     (together, "the Providers", "we", "us").

1.3  ClinAnx is CLINICAL DECISION SUPPORT SOFTWARE FOR RESEARCH USE. It is not a
     registered medical device. It has not been approved, cleared or certified
     by the National Medicines Regulatory Authority of Sri Lanka or by any
     comparable authority in any other country.

1.4  ClinAnx does not diagnose, treat, cure or prevent any condition. Its outputs
     are statistical estimates produced by machine-learning models that are
     still under evaluation and are known to be imperfect.

2. WHO MAY USE IT

2.1  You may use ClinAnx only if all of the following are true:
     (a) you are a registered medical practitioner, or a clinician or research
         staff member expressly authorised in writing by the study's Principal
         Investigator;
     (b) you have been issued credentials for your own individual use;
     (c) you have completed any training the Providers require; and
     (d) you are acting within the scope of the NHSL Ethics Review Committee
         approval for study R26-DS-012 [ERC REFERENCE NUMBER].

2.2  Credentials are personal. You must not share them, and you are responsible
     for everything done using your credentials.

3. LICENCE

3.1  We grant you a personal, revocable, non-exclusive, non-transferable licence
     to use ClinAnx for the purposes of study R26-DS-012 only.

3.2  You must not: use ClinAnx for any clinical purpose outside the study;
     reverse engineer, decompile or extract the models; copy, publish or
     redistribute the software or any model output except as the study protocol
     permits; use it to make automated decisions about any person; or use it on
     a device that is rooted, jailbroken, or otherwise compromised.

4. YOUR OBLIGATIONS — DE-IDENTIFICATION

4.1  THIS CLAUSE IS THE MOST IMPORTANT OBLIGATION IN THESE TERMS.

4.2  Clinical note text that you submit for analysis LEAVES YOUR DEVICE AND
     LEAVES SRI LANKA. It is transmitted to model inference services hosted
     outside Sri Lanka. See clause 4 of the Privacy Notice.

4.3  Accordingly, before submitting any clinical note you must remove all direct
     identifiers, including but not limited to: patient names, NIC numbers,
     addresses, telephone numbers, e-mail addresses, hospital numbers other than
     the study MRN, dates of birth, and the names of relatives or employers.

4.4  You are solely responsible for de-identification. ClinAnx does not perform
     it, does not verify it, and cannot detect its absence. Submitting
     identifiable patient data through ClinAnx is a breach of these Terms, of the
     study protocol, and may be a breach of the Personal Data Protection Act,
     No. 9 of 2022.

4.5  You must also: keep your device locked with a passcode or biometric; report
     any lost or stolen device or suspected credential compromise to the
     Principal Investigator without delay; and not install ClinAnx on a shared or
     public device.

5. NO CLINICAL WARRANTY

5.1  Every clinical decision remains yours. You must exercise independent
     professional judgement and must not rely on any ClinAnx output as the sole
     or determining basis for any diagnosis, treatment, referral, discharge,
     admission, escalation or risk assessment.

5.2  We make no representation that any output is accurate, complete, current,
     clinically valid, or fit for any particular patient. Model performance
     figures reported in the study or displayed in the application describe
     performance on research datasets and DO NOT PREDICT PERFORMANCE ON YOUR
     PATIENTS.

5.3  A low risk score is not clearance. A high risk score is not a diagnosis.
     Absence of an alert is not evidence that a patient is safe.

6. SOFTWARE AND DEVICE RISK

6.1  ClinAnx is provided "AS IS" and "AS AVAILABLE". To the fullest extent
     permitted by law, we exclude all warranties, conditions and terms implied
     by statute, common law or otherwise, including any implied warranty of
     merchantability, satisfactory quality or fitness for a particular purpose.

6.2  We do not warrant that ClinAnx will be uninterrupted, timely, secure or
     error-free, that defects will be corrected, or that model services will be
     available at any given moment. Model services may be unreachable, slow, or
     may return incorrect results.

6.3  You accept that you install and run ClinAnx on your own device at your own
     risk. We are not responsible for: your device, its hardware, its operating
     system, or its configuration; loss of or damage to your device or to data
     on it; battery, storage, performance or network-data consumption; conflicts
     with other applications; or any consequence of you modifying your device or
     its operating system.

6.4  ClinAnx stores clinical records on your device. Those records are protected
     by the operating system's own security. If your device is lost, stolen,
     rooted, jailbroken, or shared, that protection may fail. Keeping the device
     secure is your responsibility under clause 4.5.

7. LIMITATION OF LIABILITY

7.1  Nothing in these Terms limits or excludes liability for death or personal
     injury caused by our negligence, for fraud or fraudulent misrepresentation,
     or for any other liability that cannot lawfully be limited or excluded
     under the laws of Sri Lanka. If any part of this clause is unenforceable,
     the remainder continues to apply.

7.2  Subject to clause 7.1, and to the fullest extent permitted by law, we are
     not liable for: any clinical decision, act or omission by you or by any
     other person; any indirect, incidental, special, consequential or punitive
     loss; loss of profit, revenue, goodwill, reputation, opportunity or
     anticipated saving; loss or corruption of data; or damage to, or the cost
     of repairing or replacing, any device.

7.3  Subject to clause 7.1, our total aggregate liability arising out of or in
     connection with these Terms and your use of ClinAnx, whether in contract,
     tort (including negligence), breach of statutory duty or otherwise, is
     limited to [SLR 25,000.00] or the amount you paid to use ClinAnx (which is
     nil), whichever is greater.

7.4  You acknowledge that ClinAnx is supplied free of charge for research and
     that this allocation of risk is reasonable in that context.

8. INDEMNITY

8.1  You will indemnify and hold us harmless against any claim, loss, liability
     or cost (including reasonable legal fees) arising from: your breach of
     clause 4 (de-identification); your use of ClinAnx outside the ethics
     approval or the study protocol; your sharing of credentials; or your breach
     of any applicable law.

9. SUSPENSION AND TERMINATION

9.1  We may suspend or withdraw your access at any time, with or without notice,
     including where the study ends, where ethics approval is varied or
     withdrawn, or where we reasonably suspect a breach of these Terms.

9.2  You may stop using ClinAnx at any time. Clause 10 of the Privacy Notice
     explains how to withdraw and what happens to data.

10. CHANGES TO THESE TERMS

10.1 These Terms are version-controlled. If we publish a new version, you will
     be required to read and accept it before you can continue to use ClinAnx.
     Your existing acceptance is not altered and is retained as a record.

11. GOVERNING LAW

11.1 These Terms are governed by the laws of Sri Lanka. The courts of Sri Lanka
     have exclusive jurisdiction.

11.2 If any provision is held invalid or unenforceable, it is severed and the
     remaining provisions continue in full force.

12. CONTACT

     Principal Investigator: [NAME], [E-MAIL], [TELEPHONE]
     SLIIT Faculty of Computing, New Kandy Road, Malabe, Sri Lanka
''';

const String kPrivacyNotice = '''
PRIVACY NOTICE — ClinAnx

Version $kAgreementVersionInline · Effective $kEffectiveDateInline

1. WHO CONTROLS YOUR DATA

1.1  SLIIT and NHSL are JOINT CONTROLLERS of personal data processed through
     ClinAnx, within the meaning of the Personal Data Protection Act, No. 9 of
     2022 ("PDPA").

1.2  Allocation of responsibility between them:
     (a) NHSL is responsible for the lawfulness of patient care records, for
         patient-facing consent, and for clinical governance.
     (b) SLIIT is responsible for the application, the models, the inference
         infrastructure, and the security of data in transit and at rest within
         the research systems.
     (c) SLIIT is the point of contact for exercising your rights. You may
         nevertheless exercise your rights against either controller.

1.3  Data Protection Officer: [NAME], [E-MAIL], [TELEPHONE].

2. WHAT WE PROCESS

2.1  About you, as a clinician user:
     your name and clinician identifier; your institutional e-mail; your
     authentication credentials; your acceptance of this agreement (version,
     timestamp, and a cryptographic hash of the text you accepted); your device
     platform and application version; the analyses you run and the clinical
     verdicts you record; and your acknowledgement of alerts.

2.2  About patients, entered by you:
     the study MRN; age, gender, ward and referral date; the demographic fields
     required by the intervention model; GAD-7 responses; and the text of
     clinical notes you submit for analysis.

2.3  Patient data is processed under the separate written informed consent
     obtained from each patient by NHSL under Ethics Review Committee approval
     [ERC REFERENCE NUMBER]. ClinAnx does not obtain patient consent. Do not
     enter data for any patient who has not consented on paper.

3. WHY WE PROCESS IT, AND ON WHAT BASIS

3.1  Purposes: to operate the application; to generate research risk estimates;
     to conduct and validate study R26-DS-012; to maintain an audit trail of
     model outputs and clinician review; and to meet our legal, ethical and
     regulatory obligations.

3.2  Lawful basis: your consent for your own account data; the consent obtained
     by NHSL for patient data; and our legitimate interest in maintaining
     security and an accurate research audit trail.

3.3  We do not use your data or patient data for advertising, marketing, or
     profiling unrelated to the study, and we do not sell it.

4. CROSS-BORDER TRANSFER — PLEASE READ

4.1  Clinical note text and derived model inputs are transmitted to machine-
     learning inference services hosted OUTSIDE SRI LANKA, currently on Hugging
     Face infrastructure located in [JURISDICTION].

4.2  This means your submissions are processed in a country whose data
     protection regime may differ from Sri Lanka's, and may be subject to lawful
     access requests by authorities in that country.

4.3  Safeguards in place: transmission over TLS; a written processing agreement
     with the hosting provider [CONFIRM EXECUTED]; a requirement that no direct
     identifiers are transmitted (clause 4 of the Terms of Use); and retention
     limits on the inference service [CONFIRM CONFIGURED].

4.4  THE EFFECTIVENESS OF THESE SAFEGUARDS DEPENDS ON YOU DE-IDENTIFYING NOTES
     BEFORE SUBMISSION. ClinAnx cannot do this for you and cannot detect a
     failure to do it.

5. WHERE DATA IS HELD

5.1  On your device: patient records, notes, support-set examples, cached risk
     scores and alerts, in application storage protected by the operating
     system. Your credentials are held in the device keychain (iOS) or encrypted
     shared preferences (Android), not in ordinary application storage.

5.2  Off device: only what is transmitted for inference under clause 4, plus
     research datasets held on SLIIT research infrastructure.

6. HOW LONG WE KEEP IT

6.1  On-device records: until you delete the patient, withdraw, or uninstall the
     application. Deleting a patient purges every record in every namespace for
     that MRN.

6.2  Research data: [RETENTION PERIOD, e.g. five years] from the end of the
     study, as required by the ethics approval, then destroyed or irreversibly
     anonymised.

6.3  Your acceptance record: retained for the same period as the research data,
     as evidence of the basis on which processing took place. This record is
     retained even after withdrawal.

7. SECURITY

7.1  Measures include: encrypted credential storage; per-patient data
     namespacing; transport encryption; authenticated model endpoints; and an
     audit trail of analyses and clinician verdicts.

7.2  No system is perfectly secure. We do not guarantee that a determined
     attacker, a compromised device, or a failure at a third-party provider will
     not result in unauthorised access.

8. WHO ELSE SEES IT

     The study team; the NHSL Ethics Review Committee and institutional auditors;
     the inference hosting provider (as processor, under clause 4); and any
     authority we are legally obliged to disclose to. Research outputs are
     published in aggregate only and will not identify any patient or clinician.

9. YOUR RIGHTS UNDER THE PDPA

     You have the right to: be informed; access your data; have inaccurate data
     corrected; have data erased in defined circumstances; object to processing;
     restrict processing; withdraw consent; and complain.

     To exercise a right, contact the Data Protection Officer at section 1.3. We
     will respond within [21] days.

10. WITHDRAWING

10.1 You may withdraw at any time, from Settings in the application or by
     contacting the Principal Investigator. Withdrawal is included because
     section 16 of the PDPA makes it a right that cannot be signed away.

10.2 What withdrawal does: it ends your access to ClinAnx, stops all further
     collection through your account, and — at your request — deletes the
     research data associated with you.

10.3 What withdrawal does not do: it does not make lawful processing that
     already occurred unlawful; it does not remove your data from analyses
     already completed or from results already published; and it does not delete
     your acceptance record, which we retain under clause 6.3.

10.4 Where erasure of research data would render study results invalid or
     unverifiable, we may retain that data in irreversibly anonymised form. This
     is permitted for scientific research purposes and we will tell you if we
     rely on it.

11. COMPLAINTS

     If you are unsatisfied with our response, you may complain to the Data
     Protection Authority of Sri Lanka.

12. CHANGES

     This Notice is version-controlled. A new version requires fresh acceptance
     before continued use. Prior acceptances are retained, not overwritten.
''';

// Interpolation shims so the version and date appear inside the text bodies and
// are therefore covered by the SHA-256 anchor.
const String kAgreementVersionInline = '1.0.0';
const String kEffectiveDateInline = '1 August 2026';
