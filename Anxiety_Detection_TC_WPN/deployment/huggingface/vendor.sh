#!/usr/bin/env bash
#
# vendor.sh — Phase 3. Copy the FINAL TC-WPN inference code out of the research
# tree and into the deployment layer.
#
# §14: the Space mirrors only the tested deployment layer, not the whole
# research repository. This script is the boundary.
#
# §3: source code + checkpoint + tokenizer + preprocessing are ONE versioned
# unit. This script stamps the commit it vendored from into
# deployment_config.json so the unit stays traceable.
#
# Run from the repository root:
#     bash deployment/huggingface/vendor.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${REPO_ROOT}/src/tcwpn"
DEST="${REPO_ROOT}/deployment/huggingface/tc_wpn"
CONFIG="${REPO_ROOT}/deployment/huggingface/deployment_config.json"

if [ ! -f "${SRC}/model.py" ]; then
  echo "ERROR: ${SRC}/model.py not found. Run from the tcwpn_test repo root." >&2
  exit 1
fi

if [ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]; then
  echo "ERROR: working tree is dirty. Phase 1 requires a frozen, committed" >&2
  echo "       source of truth — vendoring from uncommitted code makes the" >&2
  echo "       recorded commit hash a lie." >&2
  exit 1
fi

COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
SHORT="$(git -C "${REPO_ROOT}" rev-parse --short=7 HEAD)"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "${DEST}"

# --- model.py: the final validated implementation -----------------------------
# This is PrototypicalModel + build_model. It is self-contained (imports only
# torch and transformers), so it vendors as a single file with no rewriting.
cp "${SRC}/model.py" "${DEST}/model.py"

# --- preprocessing: chunk_tokenize must match the training corpus exactly -----
# preprocessing.py holds a verbatim copy of chunk_tokenize from
# scripts/tokenize_cohort.py. Verify it has not drifted.
if ! grep -q "return_overflowing_tokens=max_chunks > 1" "${REPO_ROOT}/scripts/tokenize_cohort.py"; then
  echo "ERROR: chunk_tokenize in scripts/tokenize_cohort.py has changed shape." >&2
  echo "       Re-copy it into deployment/huggingface/tc_wpn/preprocessing.py" >&2
  echo "       and bump preprocessing_version in deployment_config.json." >&2
  exit 1
fi

cat > "${DEST}/__init__.py" <<EOF
# Vendored from dulhara79/tcwpn_test @ ${COMMIT}
# Vendored at ${NOW} by deployment/huggingface/vendor.sh
# DO NOT EDIT model.py HERE. Edit src/tcwpn/model.py and re-run vendor.sh.
VENDORED_FROM_COMMIT = "${COMMIT}"
VENDORED_AT = "${NOW}"
EOF

# --- stamp provenance into deployment_config.json -----------------------------
python3 - "$CONFIG" "$COMMIT" "$SHORT" "$NOW" <<'PY'
import json, sys
path, commit, short, now = sys.argv[1:5]
with open(path) as fh:
    cfg = json.load(fh)
cfg["provenance"]["tcwpn_git_commit"] = commit
cfg["provenance"]["tcwpn_git_commit_verified_at"] = now
mv = cfg.get("model_version", "")
if not isinstance(mv, str) or mv.startswith("<FILL"):
    cfg["model_version"] = f"tcwpn-clean-benchmark-{short}"   # §15.2
with open(path, "w") as fh:
    json.dump(cfg, fh, indent=2)
    fh.write("\n")
print(f"stamped provenance.tcwpn_git_commit = {commit}")
print(f"model_version = {cfg['model_version']}")
PY

echo
echo "Vendored:"
echo "  ${SRC}/model.py -> ${DEST}/model.py"
echo
echo "STILL TO DO BEFORE DEPLOYMENT (Phase 2 / Phase 4):"
echo "  - deployment_config.json provenance.checkpoint_filename"
echo "  - deployment_config.json provenance.checkpoint_sha256"
echo "  - deployment_config.json provenance.hf_model_repo_revision"
echo "  - deployment_config.json operating_point.threshold  (locked_threshold"
echo "    from the run manifest — do NOT copy the old Space's value)"
echo "  - deployment_config.json research_metrics.metrics_verified"
echo
echo "app.py refuses to start while any <FILL> placeholder remains."
