"""
Space Probe — find out what a teammate's Hugging Face Space actually returns.

You do NOT need to know anything about their code. This pokes the Space a few
different ways and shows you exactly what comes back, so you can plug the right
URL and the right field name into your fusion service.

Run it like this (one component at a time):

    python probe_space.py https://their-username-their-space.hf.space

If their Space needs a password (token), add it:

    python probe_space.py https://... --token hf_theirtoken

Read the output from top to bottom. The last section tells you what to do next.
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

# The endpoint paths people most commonly use on a FastAPI/Gradio Space.
COMMON_PATHS = [
    "/predict", "/fusion_component", "/score", "/api/predict",
    "/run/predict", "/infer", "/", "/health", "/docs",
]

# The shapes people most commonly expect in the POST body.
COMMON_BODIES = [
    {"patient_id": "TEST-001", "mrn": "TEST-001"},
    {"gender": "female", "age": 21, "edu": "bachelor's degree",
     "smoke": "never smokes", "drink": "never drinks"},
    {"data": ["TEST-001"]},           # Gradio-style
    {"inputs": "TEST-001"},           # HF Inference-style
    {"text": "patient clinical note goes here"},
]


def hr(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def find_scores(obj, path="response"):
    """Walk the JSON and report anything that looks like a risk score."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if any(w in k.lower() for w in
                       ("score", "risk", "prob", "value", "conf", "tier", "level")):
                    hits.append((f"{path}.{k}", k, v))
            hits += find_scores(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += find_scores(v, f"{path}[{i}]")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url", help="e.g. https://user-space.hf.space")
    ap.add_argument("--token", default=None, help="bearer token if the Space is private")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    hr(f"PROBING  {base}")

    # 1. Is it awake at all?
    print("Step 1 — is the Space awake?")
    try:
        r = httpx.get(base, headers=headers, timeout=40)
        print(f"  reachable (HTTP {r.status_code}) — if this was slow, it was waking up")
    except Exception as exc:
        print(f"  COULD NOT REACH IT: {exc}")
        print("  Check the URL is exactly right and the Space isn't private without a token.")
        sys.exit(1)

    # 2. Does it have interactive docs? (FastAPI Spaces do — huge shortcut.)
    hr("Step 2 — does it have a /docs page?")
    try:
        r = httpx.get(f"{base}/docs", headers=headers, timeout=20)
        if r.status_code == 200 and "swagger" in r.text.lower():
            print(f"  YES — open this in your browser and you'll see every endpoint:")
            print(f"     {base}/docs")
            print("  This is the easiest path. Note the endpoint path and the fields it wants.")
        else:
            print("  No Swagger docs. That's fine — we'll poke it directly below.")
    except Exception:
        print("  No /docs. Fine, moving on.")

    # 3. Try common endpoint + body combinations.
    hr("Step 3 — trying common endpoints and request shapes")
    successes = []
    for path in COMMON_PATHS:
        url = f"{base}{path}"
        # GET first for health-ish paths
        if path in ("/health", "/", "/docs"):
            try:
                r = httpx.get(url, headers=headers, timeout=20)
                if r.status_code == 200:
                    body = r.json() if "json" in r.headers.get("content-type", "") else r.text[:200]
                    print(f"  GET  {path:20s} -> 200")
                    if isinstance(body, (dict, list)):
                        scores = find_scores(body)
                        if scores:
                            successes.append(("GET", path, None, body, scores))
            except Exception:
                pass
            continue

        # POST with each candidate body
        for bidx, bod in enumerate(COMMON_BODIES):
            try:
                r = httpx.post(url, json=bod, headers=headers, timeout=25)
                if r.status_code == 200:
                    try:
                        body = r.json()
                    except Exception:
                        continue
                    scores = find_scores(body)
                    tag = "  <-- has score-like fields" if scores else ""
                    print(f"  POST {path:20s} body#{bidx} -> 200{tag}")
                    if scores:
                        successes.append(("POST", path, bod, body, scores))
                        break  # got a working body for this path
            except Exception:
                pass

    # 4. Verdict
    hr("WHAT TO DO NEXT")
    if not successes:
        print("  Nothing returned a score automatically.")
        print("  -> Open  {}/docs  in a browser if it exists,".format(base))
        print("  -> or ask your teammate: 'what's the exact URL and JSON to get a risk score?'")
        return

    method, path, body, sample, scores = successes[0]
    print(f"  It works. Use this:\n")
    print(f"    URL     : {base}{path}")
    print(f"    Method  : {method}")
    if body is not None:
        print(f"    Send    : {json.dumps(body)}")
    print(f"\n  It returned:\n")
    print("    " + json.dumps(sample, indent=2)[:600].replace("\n", "\n    "))
    print(f"\n  The score is here:  {scores[0][0]}  (field name: '{scores[0][1]}', value {scores[0][2]})")
    print(f"\n  So in your .env file, set that component's URL to:")
    print(f"     {base}{path}")
    if scores[0][1] not in ("score", "risk_score", "value", "probability", "risk"):
        print(f"\n  NOTE: the field is called '{scores[0][1]}', which your clients.py doesn't")
        print(f"        recognise yet. Add '{scores[0][1]}' to the list in to_reading().")


if __name__ == "__main__":
    main()
