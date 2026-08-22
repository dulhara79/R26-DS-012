#!/usr/bin/env bash
#
# sync_to_space.sh — mirror deployment/huggingface/ into the HF Space.
#
# GitHub is the source of truth. The Space is a MIRROR of the deployment layer
# only (§14) — never a copy of the research repository.
#
# What crosses the boundary and what does not:
#
#   GOES TO THE SPACE          app.py, deployment_config.json, requirements.txt,
#                              README.md, tc_wpn/{__init__,model,preprocessing}.py
#   STAYS IN GITHUB ONLY       vendor.sh, scripts/, tests/, Dockerfile,
#                              DEPLOYMENT_PLAN.md
#   LIVES ONLY IN THE SPACE    accounts.py, mailer.py  (untouched by this script)
#   NEVER COMMITTED ANYWHERE   .env, create_account.py
#
# Usage:
#     bash deployment/huggingface/sync_to_space.sh ../tc-wpn-demo
#
set -euo pipefail

SPACE_DIR="${1:-}"
if [ -z "${SPACE_DIR}" ]; then
  echo "usage: bash sync_to_space.sh <path to cloned tc-wpn-demo Space>" >&2
  exit 1
fi

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SRC}/../.." && pwd)"
SPACE_DIR="$(cd "${SPACE_DIR}" && pwd)"

if [ ! -d "${SPACE_DIR}/.git" ]; then
  echo "ERROR: ${SPACE_DIR} is not a git clone of the Space." >&2
  exit 1
fi
if [ ! -f "${SPACE_DIR}/accounts.py" ]; then
  echo "ERROR: ${SPACE_DIR}/accounts.py not found — wrong directory?" >&2
  exit 1
fi

# Phase 3 must have run: the vendored model must be present and current.
if [ ! -f "${SRC}/tc_wpn/model.py" ]; then
  echo "ERROR: tc_wpn/model.py is missing. Run vendor.sh first (Phase 3)." >&2
  exit 1
fi
if ! diff -q "${REPO_ROOT}/src/tcwpn/model.py" "${SRC}/tc_wpn/model.py" >/dev/null; then
  echo "ERROR: the vendored tc_wpn/model.py differs from src/tcwpn/model.py." >&2
  echo "       Re-run vendor.sh so the deployed code matches the frozen commit." >&2
  exit 1
fi

# Phase 1/2 must have completed: no placeholders may reach the Space.
# Check the SAME fields app.py::_verify_config_complete checks — no more, no
# less. Fields that can only be known after the push (hf_space_commit) or that
# are release-time decisions (metrics_verified) warn instead of blocking.
python3 - "${SRC}/deployment_config.json" <<'PY' || exit 1
import json, sys
cfg = json.load(open(sys.argv[1]))
prov, op = cfg["provenance"], cfg["operating_point"]

def unfilled(v):
    return isinstance(v, str) and v.strip().startswith("<FILL")

blocking = {
    "provenance.tcwpn_git_commit":     prov.get("tcwpn_git_commit"),
    "provenance.checkpoint_filename":  prov.get("checkpoint_filename"),
    "provenance.checkpoint_sha256":    prov.get("checkpoint_sha256"),
    "model_version":                   cfg.get("model_version"),
    "operating_point.threshold":       op.get("threshold"),
}
advisory = {
    "provenance.hf_model_repo_revision": prov.get("hf_model_repo_revision"),
    "research_metrics.metrics_verified": cfg["research_metrics"].get("metrics_verified"),
}

bad = [k for k, v in blocking.items() if unfilled(v) or v in (None, "")]
if bad:
    print("ERROR: deployment_config.json is incomplete. app.py will not start.",
          file=sys.stderr)
    for k in bad:
        print(f"       - {k}", file=sys.stderr)
    print("       Run scripts/export_checkpoint.py (Phase 4) to fill these.",
          file=sys.stderr)
    sys.exit(1)

for k, v in advisory.items():
    if unfilled(v) or v in (None, "", False):
        print(f"WARNING: {k} not set. The Space will run, but /health will "
              f"withhold that part of the provenance chain.")
PY

echo "Removing the archived architecture from the Space..."
rm -rf "${SPACE_DIR}/tc_wpn/models"

echo "Copying the deployment layer..."
mkdir -p "${SPACE_DIR}/tc_wpn"
cp "${SRC}/app.py"                     "${SPACE_DIR}/app.py"
cp "${SRC}/deployment_config.json"     "${SPACE_DIR}/deployment_config.json"
cp "${SRC}/requirements.txt"           "${SPACE_DIR}/requirements.txt"
cp "${SRC}/README.md"                  "${SPACE_DIR}/README.md"
cp "${SRC}/tc_wpn/__init__.py"         "${SPACE_DIR}/tc_wpn/__init__.py"
cp "${SRC}/tc_wpn/model.py"            "${SPACE_DIR}/tc_wpn/model.py"
cp "${SRC}/tc_wpn/preprocessing.py"    "${SPACE_DIR}/tc_wpn/preprocessing.py"

# The Space repo is public. Keep the two local-only files ignored.
for entry in "create_account.py" ".env" "__pycache__/" "*.pyc"; do
  grep -qxF "${entry}" "${SPACE_DIR}/.gitignore" 2>/dev/null \
    || echo "${entry}" >> "${SPACE_DIR}/.gitignore"
done

echo
echo "Space tree now:"
(cd "${SPACE_DIR}" && git status --short)
echo
echo "Next:"
echo "  cd ${SPACE_DIR}"
echo "  git add -A && git commit -m 'deploy: TC-WPN clean serving layer' && git push"
echo
echo "Then confirm at https://huggingface.co/spaces/dulharakaushalya/tc-wpn-demo:"
echo "  - build log shows no CUDA torch download"
echo "  - GET /health returns status ok and empty startup_errors"
echo "  - GET /health provenance.tcwpn_git_commit matches the GitHub commit"
