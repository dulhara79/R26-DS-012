import 'package:flutter/material.dart';
import '../models/models.dart';
import '../services/api_service.dart';
import '../services/notification_service.dart';

class PatientProvider extends ChangeNotifier {
  List<Patient> _patients = [];
  List<SupportNote> _supportNotes = [];
  List<AppNotification> _notifications = [];
  bool _isLoading = false;
  String? _error;

  List<Patient> get patients     => List.unmodifiable(_patients);
  List<SupportNote> get supportNotes => List.unmodifiable(_supportNotes);
  List<AppNotification> get notifications => List.unmodifiable(_notifications);
  bool get isLoading             => _isLoading;
  String? get error              => _error;

  List<SupportNote> get anxietySupport =>
      _supportNotes.where((n) => n.label == 'anxiety').toList();
  List<SupportNote> get controlSupport =>
      _supportNotes.where((n) => n.label == 'control').toList();

  // ─── Initialise with mock patients for demo ──────────────────────────────
  PatientProvider() {
    _loadMockPatients();
    NotificationService().init();
  }

  void _loadMockPatients() {
    _patients = [
      Patient(
        id: 'P001',
        name: 'Amali Perera',
        age: 24,
        gender: 'Female',
        ward: 'Psychiatry OPD',
        referralDate: '2025-04-01',
        totalVisits: 3,
        latestRisk: RiskLevel.high,
        hasAlert: true,
        assessments: [
          Assessment(
            id: 'A001', patientId: 'P001',
            timestamp: DateTime.now().subtract(const Duration(days: 14)),
            noteText: 'Patient reports persistent worry...',
            noteType: 'Psychiatry note',
            clinicianId: 'DR001',
            result: const PredictionResult(
              prediction: 'ANXIETY', riskLevel: RiskLevel.moderate,
              riskScore: 0.71, confidence: 0.71,
              keyPhrases: ['persistent worry', 'difficulty sleeping'],
              supportK: 5, threshold: 0.4036, latencyMs: 42,
              temporalContext: 'Visit 1 of 3',
            ),
          ),
          Assessment(
            id: 'A002', patientId: 'P001',
            timestamp: DateTime.now().subtract(const Duration(days: 7)),
            noteText: 'Worsening symptoms noted...',
            noteType: 'Psychiatry note',
            clinicianId: 'DR001',
            result: const PredictionResult(
              prediction: 'ANXIETY', riskLevel: RiskLevel.high,
              riskScore: 0.84, confidence: 0.84,
              keyPhrases: ['worsening anxiety', 'GAD-7 score 16', 'panic attacks'],
              supportK: 5, threshold: 0.4036, latencyMs: 38,
              temporalContext: 'Visit 2 of 3',
            ),
          ),
        ],
      ),
      Patient(
        id: 'P002',
        name: 'Kasun Silva',
        age: 31,
        gender: 'Male',
        ward: 'Psychiatry OPD',
        referralDate: '2025-03-15',
        totalVisits: 5,
        latestRisk: RiskLevel.low,
        assessments: [
          Assessment(
            id: 'A003', patientId: 'P002',
            timestamp: DateTime.now().subtract(const Duration(days: 3)),
            noteText: 'Patient reports significant improvement...',
            noteType: 'Psychiatry note',
            clinicianId: 'DR001',
            result: const PredictionResult(
              prediction: 'NO ANXIETY', riskLevel: RiskLevel.low,
              riskScore: 0.18, confidence: 0.82,
              keyPhrases: ['improvement', 'medication compliance'],
              supportK: 5, threshold: 0.4036, latencyMs: 35,
              temporalContext: 'Visit 5 of 5 · most recent',
            ),
          ),
        ],
      ),
      Patient(
        id: 'P003',
        name: 'Nimasha Fernando',
        age: 22,
        gender: 'Female',
        ward: 'Psychiatry OPD',
        referralDate: '2025-04-20',
        totalVisits: 1,
        latestRisk: RiskLevel.veryHigh,
        hasAlert: true,
        assessments: [
          Assessment(
            id: 'A004', patientId: 'P003',
            timestamp: DateTime.now().subtract(const Duration(hours: 6)),
            noteText: 'Severe panic attacks, unable to attend university...',
            noteType: 'Psychiatry note',
            clinicianId: 'DR001',
            result: const PredictionResult(
              prediction: 'ANXIETY', riskLevel: RiskLevel.veryHigh,
              riskScore: 0.94, confidence: 0.94,
              keyPhrases: ['severe panic attacks', 'agoraphobia', 'GAD-7 score 19', 'unable to function'],
              supportK: 5, threshold: 0.4036, latencyMs: 41,
              temporalContext: 'Visit 1 of 1 · first assessment',
            ),
          ),
        ],
      ),
      Patient(
        id: 'P004',
        name: 'Tharindu Rajapakse',
        age: 27,
        gender: 'Male',
        ward: 'Psychiatry OPD',
        referralDate: '2025-02-10',
        totalVisits: 8,
        latestRisk: RiskLevel.moderate,
        assessments: [
          Assessment(
            id: 'A005', patientId: 'P004',
            timestamp: DateTime.now().subtract(const Duration(days: 10)),
            noteText: 'Stable on sertraline, some residual anxiety...',
            noteType: 'Discharge summary',
            clinicianId: 'DR001',
            result: const PredictionResult(
              prediction: 'ANXIETY', riskLevel: RiskLevel.moderate,
              riskScore: 0.62, confidence: 0.62,
              keyPhrases: ['residual anxiety', 'sertraline 100mg', 'GAD-7 score 8'],
              supportK: 5, threshold: 0.4036, latencyMs: 44,
              temporalContext: 'Visit 8 of 8 · most recent',
            ),
          ),
        ],
      ),
    ];
    notifyListeners();
  }

  // ─── Run new assessment ───────────────────────────────────────────────────
  Future<PredictionResult> runAssessment({
    required String patientId,
    required String noteText,
    required String noteType,
    String clinicianId = 'DR001',
  }) async {
    _isLoading = true;
    _error     = null;
    notifyListeners();

    try {
      final result = await ApiService.predict(
        noteText:       noteText,
        noteType:       noteType,
        anxietySupport: anxietySupport.map((n) => n.text).toList(),
        controlSupport: controlSupport.map((n) => n.text).toList(),
      );

      final assessment = Assessment(
        id:          'A${DateTime.now().millisecondsSinceEpoch}',
        patientId:   patientId,
        timestamp:   DateTime.now(),
        noteText:    noteText,
        noteType:    noteType,
        clinicianId: clinicianId,
        result:      result,
      );

      final idx = _patients.indexWhere((p) => p.id == patientId);
      if (idx >= 0) {
        final p = _patients[idx];
        final updated = Patient(
          id:            p.id,
          name:          p.name,
          age:           p.age,
          gender:        p.gender,
          ward:          p.ward,
          referralDate:  p.referralDate,
          assessments:   [...p.assessments, assessment],
          latestRisk:    result.riskLevel,
          totalVisits:   p.totalVisits + 1,
          hasAlert:      result.riskLevel == RiskLevel.high ||
                         result.riskLevel == RiskLevel.veryHigh,
        );
        _patients[idx] = updated;
      }

      // Trigger notification
      final notification = AppNotification(
        id: 'N${DateTime.now().millisecondsSinceEpoch}',
        title: result.riskLevel == RiskLevel.high || result.riskLevel == RiskLevel.veryHigh 
            ? '⚠️ High Risk Detected' 
            : 'New Assessment',
        body: 'Patient ${_patients[idx].name} has a ${result.riskLevel.label}.',
        timestamp: DateTime.now(),
        type: result.riskLevel == RiskLevel.high || result.riskLevel == RiskLevel.veryHigh 
            ? NotificationType.riskAlert 
            : NotificationType.info,
        riskLevel: result.riskLevel,
        patientId: patientId,
        patientName: _patients[idx].name,
      );
      _notifications.add(notification);
      
      NotificationService().showRiskNotification(
        patientName: _patients[idx].name,
        riskLevel: result.riskLevel,
      );

      _isLoading = false;
      notifyListeners();
      return result;
    } catch (e) {
      _isLoading = false;
      _error     = e.toString();
      notifyListeners();
      rethrow;
    }
  }

  // ─── Notification management ──────────────────────────────────────────────
  void markNotificationAsRead(String id) {
    final idx = _notifications.indexWhere((n) => n.id == id);
    if (idx >= 0) {
      _notifications[idx].isRead = true;
      notifyListeners();
    }
  }

  void clearNotifications() {
    _notifications.clear();
    notifyListeners();
  }

  // ─── Support set management ───────────────────────────────────────────────
  void addSupportNote(String text, String label) {
    _supportNotes.add(SupportNote(
      id:            'SN${DateTime.now().millisecondsSinceEpoch}',
      text:          text,
      label:         label,
      addedAt:       DateTime.now(),
      weight:        1.0,
      recencyWeight: 1.0,
    ));
    notifyListeners();
  }

  void removeSupportNote(String id) {
    _supportNotes.removeWhere((n) => n.id == id);
    notifyListeners();
  }

  void clearSupportNotes() {
    _supportNotes.clear();
    notifyListeners();
  }

  // ─── Helpers ─────────────────────────────────────────────────────────────
  List<Patient> get alertPatients =>
      _patients.where((p) => p.hasAlert).toList();

  List<Patient> searchPatients(String query) {
    if (query.isEmpty) return _patients;
    final q = query.toLowerCase();
    return _patients
        .where((p) =>
            p.name.toLowerCase().contains(q) ||
            p.id.toLowerCase().contains(q))
        .toList();
  }
}
