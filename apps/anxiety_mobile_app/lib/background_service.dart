// Backwards-compatible re-export for moved background service implementation.
// This ensures that existing imports in the project don't break, while the 
// actual logic is maintained in the modular lib/services/ directory.

export 'services/background/background_service.dart';
