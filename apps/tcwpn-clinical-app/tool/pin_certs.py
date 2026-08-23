#!/usr/bin/env python3
"""
pin_certs.py — generate ClinAnx certificate pins from the live hosts.

    pip install cryptography certifi
    python tool/pin_certs.py

WHY GENERATED, NOT HAND-WRITTEN
--------------------------------
Pin values must come from the real servers. One mistyped base64 character
produces an app that cannot reach the model service, failing in a way
indistinguishable from a network outage. This reads the actual chain over TLS.

WHAT IT PINS — AND WHY NOT THE LEAF
------------------------------------
HuggingFace leaf certificates rotate every 60-90 days. Pinning one means the app
stops connecting a few weeks after release, in the field, with no warning. For a
clinical tool that is worse than no pinning at all.

So this pins two layers up:

  ROOT CA       written as PEM into a restricted trust store. The app then
                trusts ONLY these roots — a hospital TLS-inspection proxy, an
                attacker root installed on the device, or a certificate from any
                other CA is refused before a byte of note text moves.

  INTERMEDIATE  SPKI SHA-256 pins. Catches substitution within the same CA.
                Several are emitted because CAs rotate between intermediates
                and issuing from any of them is normal.

HOW THE ROOT IS FOUND
---------------------
Servers do not send their root — it is expected to be in your trust store. Three
strategies, in order:

  1. ssl.SSLSocket.get_verified_chain()   Python 3.13+, includes the root
  2. get_unverified_chain() + certifi     match the top intermediate's issuer
                                          against the Mozilla CA bundle
  3. manual OpenSSL                       printed instructions if both fail
"""

import base64
import datetime
import hashlib
import socket
import ssl
import sys
from pathlib import Path

try:
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
except ImportError:
    sys.exit("pip install cryptography certifi")

try:
    import certifi
except ImportError:
    certifi = None

# Every host ClinAnx itself connects to.
#
# NOT huggingface.co or the CDN: the app never downloads the model checkpoint —
# the Space does that, server-side. Pinning a host the app never contacts adds
# rotation risk for no security gain.
#
# Add the fusion and C3 Spaces here the day they exist. A host absent from this
# list is reached with the platform trust store, and would be the one unpinned
# hole in an otherwise pinned app.
HOSTS = [
    "dulharakaushalya-tc-wpn-demo.hf.space",
]

REVIEW_AFTER_DAYS = 365
OUT = Path(__file__).resolve().parents[1] / "lib/core/security/pinned_certificates.dart"


# =============================================================================
# CHAIN RETRIEVAL
# =============================================================================
def _load_any(blob):
    """Parses a certificate given as PEM str/bytes or DER bytes."""
    if isinstance(blob, str):
        return x509.load_pem_x509_certificate(blob.encode())
    try:
        return x509.load_der_x509_certificate(blob)
    except Exception:
        return x509.load_pem_x509_certificate(blob)


def fetch_chain(host, port=443):
    """Returns (chain, source). Chain is leaf-first and may exclude the root."""
    ctx = ssl.create_default_context()
    with socket.create_connection((host, port), timeout=20) as sock:
        with ctx.wrap_socket(sock, server_hostname=host) as tls:
            # 1. Verified chain — includes the root. Python 3.13+.
            for name in ("get_verified_chain", "get_unverified_chain"):
                getter = getattr(tls, name, None)
                if getter is None:
                    obj = getattr(tls, "_sslobj", None)
                    getter = getattr(obj, name, None) if obj else None
                if getter is None:
                    continue
                try:
                    raw = getter()
                    certs = []
                    for c in raw:
                        # _ssl.Certificate.public_bytes() defaults to PEM.
                        certs.append(_load_any(c.public_bytes()))
                    if certs:
                        return certs, name
                except Exception:
                    continue

            # 2. Leaf only.
            der = tls.getpeercert(binary_form=True)
            return [x509.load_der_x509_certificate(der)], "leaf_only"


def find_root_in_bundle(issuer_name):
    """Looks up a root by subject DN in the certifi CA bundle.

    Servers do not transmit their root, so when the runtime cannot rebuild the
    verified chain we resolve it locally against the same Mozilla bundle every
    browser uses.
    """
    if certifi is None:
        return None
    data = Path(certifi.where()).read_bytes()
    blocks, cur, inside = [], [], False
    for line in data.splitlines(keepends=True):
        if b"BEGIN CERTIFICATE" in line:
            inside, cur = True, [line]
        elif b"END CERTIFICATE" in line and inside:
            cur.append(line)
            blocks.append(b"".join(cur))
            inside = False
        elif inside:
            cur.append(line)

    for b in blocks:
        try:
            c = x509.load_pem_x509_certificate(b)
        except Exception:
            continue
        if c.subject == issuer_name:
            return c
    return None


# =============================================================================
# HELPERS
# =============================================================================
def spki_pin(cert) -> str:
    """base64(sha256(SubjectPublicKeyInfo)).

    The public KEY is hashed, not the certificate — so a CA reissuing with the
    same key does not invalidate the pin.
    """
    spki = cert.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return base64.b64encode(hashlib.sha256(spki).digest()).decode()


def dn(cert) -> str:
    try:
        cn = cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)
        if cn:
            return cn[0].value
    except Exception:
        pass
    try:
        return cert.subject.rfc4514_string()
    except Exception:
        return "unknown"


def expiry(cert):
    for attr in ("not_valid_after_utc", "not_valid_after"):
        v = getattr(cert, attr, None)
        if v is not None:
            return v
    return None


def is_root(cert) -> bool:
    return cert.issuer == cert.subject


# =============================================================================
# MAIN
# =============================================================================
def main():
    roots, intermediates, seen = [], [], set()
    reached = 0

    for host in HOSTS:
        print(f"\n  {host}")
        try:
            chain, source = fetch_chain(host)
        except Exception as e:
            print(f"    unreachable: {e}")
            continue
        reached += 1
        print(f"    chain source: {source}  ({len(chain)} certificate(s))")

        # If the runtime gave us no root, resolve the top cert's issuer.
        if not any(is_root(c) for c in chain):
            top = chain[-1]
            found = find_root_in_bundle(top.issuer)
            if found is not None:
                chain = chain + [found]
                print(f"    root resolved from certifi bundle")
            else:
                print("    root NOT found — see the OpenSSL fallback below")

        for i, c in enumerate(chain):
            role = "leaf" if i == 0 else ("root" if is_root(c) else "intermediate")
            pin = spki_pin(c)
            exp = expiry(c)
            exp_s = exp.strftime("%Y-%m-%d") if exp else "?"
            print(f"    {role:<13} {dn(c)[:52]:<52} exp {exp_s}")
            if role != "leaf":
                print(f"                  pin {pin}")

            if role == "leaf":
                continue  # never pinned: rotates too often
            if pin in seen:
                continue
            seen.add(pin)

            entry = {
                "pin": pin,
                "name": dn(c),
                "pem": c.public_bytes(serialization.Encoding.PEM).decode(),
                "expires": exp_s,
            }
            (roots if role == "root" else intermediates).append(entry)

    if reached == 0:
        sys.exit("\n  No host was reachable. Check your network and try again.\n")

    if not roots:
        sys.exit(f"""
  No root certificate could be extracted.

  Neither this Python build nor the certifi bundle produced one. Get it with
  OpenSSL — on Windows, use Git Bash:

     openssl s_client -showcerts -servername {HOSTS[0]} \\
             -connect {HOSTS[0]}:443 < /dev/null

  The LAST -----BEGIN CERTIFICATE----- block in the output is the root.
  Paste it into kPinnedRootsPem in:
     lib/core/security/pinned_certificates.dart

  Or install certifi and re-run:   pip install certifi
""")

    review = (
        datetime.date.today() + datetime.timedelta(days=REVIEW_AFTER_DAYS)
    ).isoformat()
    today = datetime.date.today().isoformat()

    pem_blob = "\n".join(r["pem"].strip() for r in roots)

    def pin_list(items, indent="  "):
        if not items:
            return f"{indent}// none found\n"
        out = ""
        for i in items:
            out += f"{indent}// {i['name'][:66]}\n"
            out += f"{indent}// expires {i['expires']}\n"
            out += f"{indent}'{i['pin']}',\n"
        return out

    hosts_dart = "\n".join(f"  '{h}'," for h in HOSTS)
    src_comment = "\n".join(f"//   {h}" for h in HOSTS)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        f"""// GENERATED FILE — do not edit by hand.
//
//     python tool/pin_certs.py
//
// Generated {today} from:
{src_comment}
//
// Pins are on the ROOT and INTERMEDIATE certificates, never the leaf. Leaf
// certificates rotate every 60-90 days; pinning one would take the app offline
// in the field a few weeks after release.

/// Review the pin set by this date. Settings shows a warning once it passes —
/// a pin set that silently rots is how a pinned app dies in a clinic.
const String kPinsReviewBy = '{review}';

const String kPinsGeneratedOn = '{today}';

/// The ONLY certificate authorities ClinAnx trusts. Anything signed by another
/// CA — including a TLS-inspecting proxy or an attacker root installed on the
/// device — fails the handshake before any note text is transmitted.
const String kPinnedRootsPem = \'\'\'
{pem_blob}
\'\'\';

/// SPKI SHA-256 pins for intermediates. Several are listed because CAs rotate
/// between issuing intermediates as a matter of course. A chain matching ANY
/// pin is accepted.
const List<String> kIntermediateSpkiPins = <String>[
{pin_list(intermediates)}];

/// Root SPKI pins, for the startup verification pass.
const List<String> kRootSpkiPins = <String>[
{pin_list(roots)}];

/// Hosts pinning is enforced for. A host NOT in this list is reached with the
/// platform trust store — add the fusion and C3 Spaces the day they exist.
const List<String> kPinnedHosts = <String>[
{hosts_dart}
];
""",
        encoding="utf-8",
    )

    print(f"\n  Wrote {OUT}")
    print(f"  {len(roots)} root(s), {len(intermediates)} intermediate(s)")
    print(f"  Review by {review}")
    print("\n  Next: flutter run, then Settings → Connection security\n")


if __name__ == "__main__":
    main()
