import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import '../models/models.dart';
import '../services/api_service.dart';
import '../services/notification_service.dart';

class PatientProvider extends ChangeNotifier {
  List<Patient> _patients = [];
  List<SupportNote> _supportNotes = [];
  List<AppNotification> _notifications = [];
  bool _isLoading = false;
  String? _error;
  bool _isOffline = false;

  List<Patient> get patients     => List.unmodifiable(_patients);
  List<SupportNote> get supportNotes => List.unmodifiable(_supportNotes);
  List<AppNotification> get notifications => List.unmodifiable(_notifications);
  bool get isLoading             => _isLoading;
  String? get error              => _error;
  bool get isOffline             => _isOffline;
  
  List<SupportNote> get anxietySupport =>
      _supportNotes.where((n) => n.label == 'anxiety').toList();
  List<SupportNote> get controlSupport =>
      _supportNotes.where((n) => n.label == 'control').toList();

  static const String _keyPatients = 'patients_v1';
  static const String _keySupportNotes = 'support_notes_v1';
  static const String _keyNotifications = 'notifications_v1';

  PatientProvider() {
    _init();
    _checkConnectivity();
    Connectivity().onConnectivityChanged.listen((results) {
      // Check if any of the results indicate a connection
      final hasConnection = results.any((result) => result != ConnectivityResult.none);
      _isOffline = !hasConnection;
      notifyListeners();
    });
  }

  Future<void> _checkConnectivity() async {
    final results = await Connectivity().checkConnectivity();
    final hasConnection = results.any((result) => result != ConnectivityResult.none);
    _isOffline = !hasConnection;
    notifyListeners();
  }

  Future<void> _init() async {
    _isLoading = true;
    notifyListeners();
    
    await _loadFromDisk();
    
    if (_patients.isEmpty) {
      _loadMockPatients();
    }
    
    await NotificationService().init();
    
    _isLoading = false;
    notifyListeners();
  }

  // ─── Persistence ──────────────────────────────────────────────────────────
  
  Future<void> _saveToDisk() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyPatients, jsonEncode(_patients.map((p) => p.toJson()).toList()));
    await prefs.setString(_keySupportNotes, jsonEncode(_supportNotes.map((n) => n.toJson()).toList()));
    await prefs.setString(_keyNotifications, jsonEncode(_notifications.map((n) => n.toJson()).toList()));
  }

  Future<void> _loadFromDisk() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      
      final pStr = prefs.getString(_keyPatients);
      if (pStr != null) {
        final List decoded = jsonDecode(pStr);
        _patients = decoded.map((p) => Patient.fromJson(p)).toList();
      }

      final snStr = prefs.getString(_keySupportNotes);
      if (snStr != null) {
        final List decoded = jsonDecode(snStr);
        _supportNotes = decoded.map((n) => SupportNote.fromJson(n)).toList();
      }

      final nStr = prefs.getString(_keyNotifications);
      if (nStr != null) {
        final List decoded = jsonDecode(nStr);
        _notifications = decoded.map((n) => AppNotification.fromJson(n)).toList();
      }
    } catch (e) {
      debugPrint('Error loading from disk: $e');
    }
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
        ],
      ),
    ];
    _saveToDisk();
  }

  // ─── Patient Management ───────────────────────────────────────────────────
  
  Future<void> addPatient({
    required String id,
    required String name,
    required int age,
    required String gender,
    required String ward,
  }) async {
    final patient = Patient(
      id: id,
      name: name,
      age: age,
      gender: gender,
      ward: ward,
      referralDate: DateTime.now().toIso8601String().split('T')[0],
      assessments: [],
      latestRisk: RiskLevel.low,
      totalVisits: 0,
    );
    
    _patients.add(patient);
    addNotification(
      title: 'New Patient Added',
      body: 'Patient $name ($id) has been registered successfully.',
      type: NotificationType.info,
      patientId: id,
      patientName: name,
    );
    await _saveToDisk();
    notifyListeners();
  }

  Future<void> updatePatient(Patient updatedPatient) async {
    final idx = _patients.indexWhere((p) => p.id == updatedPatient.id);
    if (idx >= 0) {
      _patients[idx] = updatedPatient;
      addNotification(
        title: 'Patient Updated',
        body: 'Details for ${updatedPatient.name} have been updated.',
        type: NotificationType.info,
        patientId: updatedPatient.id,
        patientName: updatedPatient.name,
      );
      await _saveToDisk();
      notifyListeners();
    }
  }

  Future<void> removePatient(String id) async {
    final idx = _patients.indexWhere((p) => p.id == id);
    if (idx >= 0) {
      final name = _patients[idx].name;
      _patients.removeAt(idx);
      addNotification(
        title: 'Patient Removed',
        body: 'Patient $name ($id) has been removed from the system.',
        type: NotificationType.system,
      );
      await _saveToDisk();
      notifyListeners();
    }
  }

  // ─── Assessment Management ────────────────────────────────────────────────
  
  Future<PredictionResult?> saveAssessment({
    required String patientId,
    required String noteText,
    required String noteType,
    bool skipAnalysis = false,
    String clinicianId = 'DR001',
  }) async {
    _isLoading = true;
    _error     = null;
    notifyListeners();

    PredictionResult? result;
    
    if (!skipAnalysis && !_isOffline) {
      try {
        result = await ApiService.predict(
          noteText:       noteText,
          noteType:       noteType,
          anxietySupport: _supportNotes.where((n) => n.label == 'anxiety').map((n) => n.text).toList(),
          controlSupport: _supportNotes.where((n) => n.label == 'control').map((n) => n.text).toList(),
        );
      } catch (e) {
        _error = e.toString();
        // If analysis fails but it's not a connectivity error, we might want to stop.
        // But if it IS connectivity, we can still save as draft.
        if (e is! ApiException || e.statusCode != 0) {
          _isLoading = false;
          notifyListeners();
          rethrow;
        }
      }
    }

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
        latestRisk:    result?.riskLevel ?? p.latestRisk,
        totalVisits:   p.totalVisits + 1,
        hasAlert:      result != null && (result.riskLevel == RiskLevel.high || result.riskLevel == RiskLevel.veryHigh),
      );
      _patients[idx] = updated;

      addNotification(
        title: result == null ? 'Assessment Saved (Offline)' : (result.riskLevel == RiskLevel.high || result.riskLevel == RiskLevel.veryHigh ? '⚠️ High Risk Detected' : 'New Assessment'),
        body: result == null ? 'Note for ${p.name} saved without analysis.' : 'Patient ${p.name} has a ${result.riskLevel.label}.',
        type: result != null && (result.riskLevel == RiskLevel.high || result.riskLevel == RiskLevel.veryHigh) ? NotificationType.riskAlert : NotificationType.info,
        riskLevel: result?.riskLevel,
        patientId: patientId,
        patientName: p.name,
      );
    }

    await _saveToDisk();
    _isLoading = false;
    notifyListeners();
    return result;
  }

  // ─── Notification management ──────────────────────────────────────────────
  void addNotification({
    required String title,
    required String body,
    NotificationType type = NotificationType.info,
    RiskLevel? riskLevel,
    String? patientId,
    String? patientName,
  }) {
    final notification = AppNotification(
      id: 'N${DateTime.now().millisecondsSinceEpoch}',
      title: title,
      body: body,
      timestamp: DateTime.now(),
      type: type,
      riskLevel: riskLevel,
      patientId: patientId,
      patientName: patientName,
    );
    _notifications.insert(0, notification);
    _saveToDisk();
    notifyListeners();
  }

  void markNotificationAsRead(String id) {
    final idx = _notifications.indexWhere((n) => n.id == id);
    if (idx >= 0) {
      _notifications[idx].isRead = true;
      _saveToDisk();
      notifyListeners();
    }
  }

  void clearNotifications() {
    _notifications.clear();
    _saveToDisk();
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
    _saveToDisk();
    notifyListeners();
  }

  void removeSupportNote(String id) {
    _supportNotes.removeWhere((n) => n.id == id);
    _saveToDisk();
    notifyListeners();
  }

  void clearSupportNotes() {
    _supportNotes.clear();
    _saveToDisk();
    notifyListeners();
  }

  // ─── Helpers ─────────────────────────────────────────────────────────────
  List<Patient> get alertPatients =>
      _patients.where((p) => p.hasAlert).toList();

  List<Patient> searchPatients(String query, {RiskLevel? riskFilter, String? wardFilter}) {
    List<Patient> results = _patients;
    if (query.isNotEmpty) {
      final q = query.toLowerCase();
      results = results.where((p) => p.name.toLowerCase().contains(q) || p.id.toLowerCase().contains(q)).toList();
    }
    if (riskFilter != null) results = results.where((p) => p.latestRisk == riskFilter).toList();
    if (wardFilter != null && wardFilter != 'All') results = results.where((p) => p.ward == wardFilter).toList();
    return results;
  }
}
