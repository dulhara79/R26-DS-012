#!/usr/bin/env python3
"""
diagnose_c3.py — find out what is ACTUALLY wrong with the C3 login.

modality_clients._get_c3_token() catches every exception and returns "",
so the real failure never reaches a log. This script performs the identical
login call with the identical credentials and prints the raw truth.

Run from the central_backend folder:
    ./.venv/bin/python3 diagnose_c3.py
"""

import os, sys, json

try:
    import httpx
except ImportError:
    sys.exit("httpx not installed. Run: ./.venv/bin/python3 -m pip install httpx")

# ── load .env exactly the way the backend does ───────────────────────────────
env = {}
if os.path.exists(".env"):
    for line in open(".env"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
else:
    sys.exit("No .env in this folder. cd to central_backend first.")

url        = env.get("C3_URL", "").rstrip("/")
token      = env.get("C3_TOKEN", "")
clinician  = env.get("C3_CLINICIAN_ID", "")
password   = env.get("C3_PASSWORD", "")

print("=" * 70)
print("CONFIG AS THE BACKEND SEES IT")
print("=" * 70)
print(f"  C3_URL           = {url or '(empty)'}")
print(f"  C3_TOKEN         = {'(empty - will log in)' if not token else '(SET - login SKIPPED)'}")
print(f"  C3_CLINICIAN_ID  = {clinician or '(empty)'}")
print(f"  C3_PASSWORD      = {'*' * len(password)} ({len(password)} chars)")
print()

if token:
    print("!! C3_TOKEN is set, so _get_c3_token() returns it directly and never")
    print("   logs in. If that token is stale you get a permanent 401.")
    print("   Set C3_TOKEN= (empty) to force a fresh login.")
    print()

if not url:
    sys.exit("C3_URL is empty - nothing to test.")

# ── 1. is the Space awake? ───────────────────────────────────────────────────
print("=" * 70)
print("STEP 1 - IS THE SPACE REACHABLE?")
print("=" * 70)
try:
    r = httpx.get(url, timeout=60.0, follow_redirects=True)
    print(f"  GET {url}")
    print(f"  -> HTTP {r.status_code}")
    if r.status_code == 200:
        print("  Space is awake.")
    else:
        print(f"  Body: {r.text[:300]}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    print("  The Space may be asleep or the URL may be wrong.")
print()

# ── 2. what does the login endpoint actually say? ────────────────────────────
print("=" * 70)
print("STEP 2 - THE LOGIN CALL THE BACKEND SWALLOWS")
print("=" * 70)
if not clinician or not password:
    print("  C3_CLINICIAN_ID or C3_PASSWORD is empty.")
    print("  _get_c3_token() returns '' immediately -> guaranteed 401.")
else:
    payload = {"clinician_id": clinician, "password": password}
    print(f"  POST {url}/auth/login")
    print(f"  payload: {json.dumps({'clinician_id': clinician, 'password': '***'})}")
    try:
        r = httpx.post(f"{url}/auth/login", json=payload, timeout=60.0)
        print(f"  -> HTTP {r.status_code}")
        print(f"  -> body: {r.text[:500]}")
        print()
        if r.status_code == 200:
            body = r.json()
            if "access_token" in body:
                print("  *** LOGIN WORKS. Credentials are correct. ***")
                print(f"      token starts: {body['access_token'][:25]}...")
                print(f"      expires_in:   {body.get('expires_in', 'not sent')}")
                print()
                print("  If notes still fail, the problem is downstream of login")
                print("  (the /predict call), not the credentials.")
            else:
                print("  !! HTTP 200 but no 'access_token' field.")
                print("     modality_clients.py does body['access_token'] and will")
                print("     KeyError -> caught -> empty token -> 401.")
                print(f"     Keys present: {list(body.keys())}")
        elif r.status_code in (401, 403):
            print("  *** CREDENTIALS ARE REJECTED. ***")
            print("      The clinician_id / password pair is wrong, or DR001")
            print("      does not exist on this Space. Ask Dulhara for the")
            print("      correct pair - do not keep guessing.")
        elif r.status_code == 404:
            print("  *** /auth/login DOES NOT EXIST on this Space. ***")
            print("      The route may have been renamed. Check the Space's")
            print("      /docs page for the real auth path.")
        elif r.status_code == 422:
            print("  *** SCHEMA MISMATCH. ***")
            print("      The Space expects different field names than")
            print("      clinician_id/password. The body above says which.")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        print("  This is the exception _get_c3_token() silently discards.")
print()

# ── 3. what routes does the Space actually expose? ───────────────────────────
print("=" * 70)
print("STEP 3 - WHAT ROUTES DOES THE SPACE EXPOSE?")
print("=" * 70)
try:
    r = httpx.get(f"{url}/openapi.json", timeout=60.0)
    if r.status_code == 200:
        paths = r.json().get("paths", {})
        for p in sorted(paths):
            methods = ",".join(m.upper() for m in paths[p])
            print(f"  {methods:12s} {p}")
    else:
        print(f"  /openapi.json -> HTTP {r.status_code} (not a FastAPI app?)")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print()
print("=" * 70)
print("Send this whole output to whoever owns the C3 Space.")
print("=" * 70)
