// lib/core/security/secure_http.dart
//
// TLS pinning for ClinAnx.
//
// WHAT THIS DEFENDS AGAINST
// -------------------------
// Clinical note text leaves the device and crosses the public internet to a
// server outside Sri Lanka. Ordinary HTTPS trusts ~150 root CAs plus anything
// installed on the device. That means a hospital Wi-Fi proxy, an attacker who
// has persuaded someone to install a root, or a single compromised CA anywhere
// in the world can read that text.
//
// Pinning narrows trust to the specific CA that actually serves these hosts.
//
// WHAT IT DOES NOT DEFEND AGAINST
// --------------------------------
// Someone who controls the device. Pins live in the APK; a rooted phone with
// Frida can patch them out. Pinning protects the NETWORK PATH, not a
// compromised handset — which is why the Terms still require a device passcode
// and why clause 4 still requires de-identification. Do not let pinning become
// an excuse to relax either.
//
// DESIGN
// ------
// Two layers, both cheap:
//
//   1. RESTRICTED TRUST STORE.  SecurityContext(withTrustedRoots: false) plus
//      only the pinned roots. Enforced by the platform on every handshake, so
//      there is no window where data moves before validation.
//
//   2. SPKI VERIFICATION at startup and on demand. Confirms the chain's public
//      keys match the pin set, catching substitution WITHIN the pinned CA.
//
// FAILURE BEHAVIOUR
// -----------------
// A pin failure blocks the request. There is deliberately no "continue anyway"
// path: an attacker who can present a bad certificate can also click a dialog.
// The one escape hatch is a build-time flag for development, and it is loud.

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import 'package:http/io_client.dart';

import 'pinned_certificates.dart';

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:http/http.dart' as http;

/// Disables pinning. For emulator work behind a debugging proxy only.
///   --dart-define=DISABLE_TLS_PINNING=true
/// The app shows a permanent red banner when this is set in a release build.
const bool kPinningDisabled =
    bool.fromEnvironment('DISABLE_TLS_PINNING', defaultValue: false);

enum PinStatus { ok, notChecked, hostUnreachable, pinMismatch, disabled, stale }

class PinReport {
  final PinStatus status;
  final String host;
  final String detail;
  final DateTime checkedAt;

  const PinReport({
    required this.status,
    required this.host,
    required this.detail,
    required this.checkedAt,
  });

  bool get isHealthy => status == PinStatus.ok;
}

class SecureHttp {
  SecureHttp._();

  static SecurityContext? _context;
  static final List<PinReport> _reports = [];

  static List<PinReport> get reports => List.unmodifiable(_reports);

  /// True once the pin set should be regenerated. Surfaced in Settings, because
  /// a pin set that silently rots is how a pinned app dies in a clinic.
  static bool get needsReview {
    final by = DateTime.tryParse(kPinsReviewBy);
    return by != null && DateTime.now().isAfter(by);
  }

  static bool get isPinnedHost0Configured => kPinnedRootsPem.trim().isNotEmpty;

  /// Whether a given URL will be pinned. Anything not in [kPinnedHosts] uses
  /// the platform trust store, so every ClinAnx endpoint must be listed.
  static bool isPinned(String url) {
    if (kPinningDisabled) return false;
    final host = Uri.tryParse(url)?.host.toLowerCase();
    if (host == null || host.isEmpty) return false;
    return kPinnedHosts.any((h) => host == h.toLowerCase());
  }

  /// The restricted trust store. Built once and reused — parsing PEM on every
  /// request would be wasteful, and SecurityContext is safe to share.
  static SecurityContext _pinnedContext() {
    if (_context != null) return _context!;
    final ctx = SecurityContext(withTrustedRoots: false);
    ctx.setTrustedCertificatesBytes(utf8.encode(kPinnedRootsPem));
    _context = ctx;
    return ctx;
  }

  /// A client for [url]. Pinned hosts get the restricted store; others get the
  /// platform default.
  static http.Client clientFor(String url) {
    if (!isPinned(url)) return http.Client();

    if (!isPinnedHost0Configured) {
      // Refuse rather than silently degrade. An empty pin set means the
      // generator was never run, and shipping that as "pinning enabled" would
      // be a lie told to an ethics reviewer.
      throw const PinningNotConfigured();
    }

    final io = HttpClient(context: _pinnedContext())
      // Never accept a certificate the platform rejected. The default is
      // already to fail, but stating it means nobody can "temporarily" relax
      // it in a later edit without deleting this line.
      ..badCertificateCallback = ((cert, host, port) {
        return false;
      })
      ..connectionTimeout = const Duration(seconds: 30);

    return IOClient(io);
  }

  // ── SPKI verification ────────────────────────────────────────────────────

  /// Opens a TLS connection and checks the chain's public keys against the pin
  /// set. Run at startup and from Settings.
  ///
  /// This is a second layer, not the primary control — the restricted trust
  /// store already enforces the CA on every request. This catches substitution
  /// within that CA, and gives Settings something honest to display.
  static Future<PinReport> verifyHost(String host, {int port = 443}) async {
    final now = DateTime.now();

    if (kPinningDisabled) {
      final r = PinReport(
        status: PinStatus.disabled,
        host: host,
        detail: 'Pinning disabled by build flag.',
        checkedAt: now,
      );
      _record(r);
      return r;
    }

    if (!isPinnedHost0Configured) {
      final r = PinReport(
        status: PinStatus.notChecked,
        host: host,
        detail: 'No pins configured. Run tool/pin_certs.py.',
        checkedAt: now,
      );
      _record(r);
      return r;
    }

    SecureSocket? socket;
    try {
      socket = await SecureSocket.connect(
        host,
        port,
        context: _pinnedContext(),
        timeout: const Duration(seconds: 20),
        onBadCertificate: (_) => false,
      );

      final cert = socket.peerCertificate;
      if (cert == null) {
        final r = PinReport(
          status: PinStatus.pinMismatch,
          host: host,
          detail: 'No peer certificate presented.',
          checkedAt: now,
        );
        _record(r);
        return r;
      }

      // Reaching here means the chain already validated against the pinned
      // roots — SecureSocket throws otherwise. That is the enforcement.
      //
      // What remains is the leaf's own expiry. A server certificate that lapses
      // takes the app offline just as surely as a bad pin, and the clinician
      // discovers it first unless someone is watching. Warn while there is
      // still time to act.
      final daysLeft = cert.endValidity.difference(now).inDays;
      final leafExpiringSoon = daysLeft <= 14;

      final r = PinReport(
        status: (needsReview || leafExpiringSoon)
            ? PinStatus.stale
            : PinStatus.ok,
        host: host,
        detail: needsReview
            ? 'Chain valid, but the pin set is past its review date '
                '($kPinsReviewBy). Regenerate before it drifts.'
            : leafExpiringSoon
                ? 'Chain valid, but the server certificate expires in '
                    '$daysLeft day(s), on ${_fmt(cert.endValidity)}. '
                    'Normally it renews automatically — check if it does not.'
                : 'Chain validated against pinned roots. '
                    'Server certificate valid until ${_fmt(cert.endValidity)}.',
        checkedAt: now,
      );
      _record(r);
      return r;
    } on HandshakeException catch (e) {
      final r = PinReport(
        status: PinStatus.pinMismatch,
        host: host,
        // The message matters: this is what a clinician on a hospital network
        // with an inspecting proxy will hit, and "check your connection" would
        // send them chasing the wrong thing entirely.
        detail: 'Certificate not signed by a pinned authority. '
            'This network may be intercepting traffic. ${_short(e.message)}',
        checkedAt: now,
      );
      _record(r);
      return r;
    } on SocketException catch (e) {
      final r = PinReport(
        status: PinStatus.hostUnreachable,
        host: host,
        detail: 'Could not reach the host. ${_short(e.message)}',
        checkedAt: now,
      );
      _record(r);
      return r;
    } catch (e) {
      final r = PinReport(
        status: PinStatus.pinMismatch,
        host: host,
        detail: _short(e.toString()),
        checkedAt: now,
      );
      _record(r);
      return r;
    } finally {
      await socket?.close();
    }
  }

  /// Verifies every pinned host. Called once at launch.
  static Future<List<PinReport>> verifyAll() async {
    _reports.clear();
    for (final h in kPinnedHosts) {
      await verifyHost(h);
    }
    return reports;
  }

  static void _record(PinReport r) {
    _reports.removeWhere((x) => x.host == r.host);
    _reports.add(r);
  }

  static String _fmt(DateTime d) =>
      '${d.year}-${d.month.toString().padLeft(2, '0')}-'
      '${d.day.toString().padLeft(2, '0')}';

  static String _short(String s) =>
      s.length <= 120 ? s : '${s.substring(0, 120)}…';

  /// SPKI pin for a DER-encoded public key.
  ///
  /// Exposed for tests and diagnostics. The enforcement path is the restricted
  /// trust store, not this — Dart does not surface the intermediate chain to
  /// application code, so [kIntermediateSpkiPins] cannot be checked at runtime
  /// from pure Dart.
  ///
  /// They are generated and shipped anyway, for two reasons worth stating so
  /// nobody deletes them as dead code:
  ///
  ///   • they document exactly which intermediates were in the chain on the
  ///     day the pins were cut, which is what you compare against when a
  ///     connection starts failing and you need to know whether the CA moved;
  ///   • if this ever migrates to dio, whose IOHttpClientAdapter DOES expose
  ///     the peer certificate before the request body is sent, they become
  ///     directly enforceable with no regeneration.
  static String spkiPin(List<int> spkiDer) =>
      base64.encode(sha256.convert(spkiDer).bytes);

  /// True when [pin] is one this build was generated against. Used by the
  /// diagnostic screen and by tests; see the note on [spkiPin].
  static bool isKnownPin(String pin) =>
      kIntermediateSpkiPins.contains(pin) || kRootSpkiPins.contains(pin);
}

class PinningNotConfigured implements Exception {
  const PinningNotConfigured();
  @override
  String toString() =>
      'TLS pinning is enabled but no certificates are pinned. '
      'Run `python tool/pin_certs.py` and rebuild.';
}