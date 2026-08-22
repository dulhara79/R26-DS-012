#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/care-anxrag-validation.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

printf '%s\n' "[1/8] Running syntax, configuration, resource, and documentation audits"
python -m compileall -q "$ROOT/src" "$ROOT/tests" "$ROOT/scripts"
python -m tabnanny "$ROOT/src" "$ROOT/tests" "$ROOT/scripts"
for script in "$ROOT"/scripts/*.sh; do
  bash -n "$script"
done
python - "$ROOT" "$TMP/static-audit.json" <<'PY'
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote

import yaml

from care_anxrag.evaluation import load_benchmark
from care_anxrag.registry import load_source_registry

root = Path(__import__("sys").argv[1]).resolve()
output = Path(__import__("sys").argv[2])

pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
if pyproject["project"]["name"] != "care-anxrag":
    raise AssertionError("Unexpected project name")
if pyproject["project"]["version"] != "0.1.0":
    raise AssertionError("Unexpected project version")

source_text = (root / "config" / "sources.yaml").read_text(encoding="utf-8")
packaged_source_text = (
    root / "src" / "care_anxrag" / "resources" / "sources.yaml"
).read_text(encoding="utf-8")
if source_text != packaged_source_text:
    raise AssertionError("Packaged source registry does not match config/sources.yaml")

env_text = (root / ".env.example").read_text(encoding="utf-8")
packaged_env_text = (
    root / "src" / "care_anxrag" / "resources" / "env.example"
).read_text(encoding="utf-8")
if env_text != packaged_env_text:
    raise AssertionError("Packaged env template does not match .env.example")

sources = load_source_registry(root / "config" / "sources.yaml")
if not sources:
    raise AssertionError("Source registry is empty")
if not any(source.enabled for source in sources):
    raise AssertionError("Source registry has no enabled sources")

for yaml_path in (root / "compose.yaml", root / "CITATION.cff"):
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{yaml_path.name} did not parse as a mapping")

benchmark = load_benchmark(root / "data" / "benchmark" / "example.jsonl")
if not benchmark:
    raise AssertionError("Example benchmark is empty")

markdown_files = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
broken_links: list[str] = []
checked_links = 0
for markdown_path in markdown_files:
    text = markdown_path.read_text(encoding="utf-8")
    for raw_target in link_pattern.findall(text):
        target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target = unquote(target.split("#", 1)[0])
        if not target:
            continue
        checked_links += 1
        resolved = (markdown_path.parent / target).resolve()
        if not resolved.exists():
            broken_links.append(f"{markdown_path.relative_to(root)} -> {target}")
if broken_links:
    raise AssertionError("Broken local Markdown links: " + "; ".join(broken_links))

report = {
    "status": "passed",
    "python_compileall": "passed",
    "tabnanny": "passed",
    "shell_syntax": "passed",
    "pyproject_toml": "passed",
    "yaml_documents": ["config/sources.yaml", "compose.yaml", "CITATION.cff"],
    "source_registry_count": len(sources),
    "enabled_source_count": sum(source.enabled for source in sources),
    "packaged_resources_match": True,
    "benchmark_record_count": len(benchmark),
    "local_markdown_links_checked": checked_links,
    "ruff": {
        "status": "not_executed",
        "reason": "Ruff is not installed in the offline validation environment; compileall, tabnanny, shell parsing, tests, and package audits were executed instead.",
    },
}
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
PY

printf '%s\n' "[2/8] Running the deterministic test suite with coverage"
rm -f "$ROOT/.coverage"
coverage run --source=care_anxrag -m pytest -q --junitxml="$TMP/junit.xml"
coverage json -o "$TMP/coverage.json"
coverage report --fail-under=70

printf '%s\n' "[3/8] Running the deterministic self-check"
CARE_HOME="$TMP/selfcheck" \
CARE_DATABASE_PATH="$TMP/selfcheck/care.sqlite3" \
CARE_VECTOR_PATH="$TMP/selfcheck/vectors" \
CARE_VECTOR_BACKEND=sqlite \
CARE_EMBEDDING_PROVIDER=hash \
CARE_EMBEDDING_MODEL=hash \
CARE_EMBEDDING_DIMENSIONS=256 \
CARE_GENERATOR_PROVIDER=rule \
CARE_RERANKER_PROVIDER=heuristic \
CARE_NLI_PROVIDER=heuristic \
CARE_ALLOW_NETWORK_SYNC=false \
python -m care_anxrag.cli selfcheck --offline --project-root "$ROOT" > "$TMP/selfcheck.json"

printf '%s\n' "[4/8] Running end-to-end offline acceptance"
python "$ROOT/scripts/offline_acceptance.py" \
  --project-root "$ROOT" \
  --output "$TMP/acceptance.json" > /dev/null

printf '%s\n' "[5/8] Building source and wheel distributions"
rm -rf "$ROOT/dist" "$ROOT/build" "$ROOT/src/care_anxrag.egg-info"
if command -v uv >/dev/null 2>&1; then
  uv build --offline --no-build-isolation --project "$ROOT" > "$TMP/build.log"
elif python -c 'import build' >/dev/null 2>&1; then
  python -m build --no-isolation --outdir "$ROOT/dist" "$ROOT" > "$TMP/build.log"
else
  echo "Validation requires either uv or the Python build package (install .[dev])." >&2
  exit 1
fi
WHEEL=$(find "$ROOT/dist" -maxdepth 1 -type f -name '*.whl' | sort | head -n 1)
SDIST=$(find "$ROOT/dist" -maxdepth 1 -type f -name '*.tar.gz' | sort | head -n 1)
if [ -z "$WHEEL" ] || [ -z "$SDIST" ]; then
  echo "Both wheel and source distribution are required" >&2
  exit 1
fi
python -m zipfile -t "$WHEEL" > "$TMP/wheel-integrity.txt"
tar -tzf "$SDIST" > "$TMP/sdist-contents.txt"

printf '%s\n' "[6/8] Inspecting, installing, and invoking the built wheel"
python - "$ROOT" "$WHEEL" "$SDIST" "$TMP/package-audit.json" <<'PY'
from __future__ import annotations

import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

root = Path(__import__("sys").argv[1]).resolve()
wheel = Path(__import__("sys").argv[2]).resolve()
sdist = Path(__import__("sys").argv[3]).resolve()
output = Path(__import__("sys").argv[4])

wheel_required = {
    "care_anxrag/__init__.py",
    "care_anxrag/api.py",
    "care_anxrag/cli.py",
    "care_anxrag/resources/sources.yaml",
    "care_anxrag/resources/env.example",
}
with zipfile.ZipFile(wheel) as archive:
    wheel_names = set(archive.namelist())
    missing = sorted(wheel_required - wheel_names)
    if missing:
        raise AssertionError(f"Wheel is missing required files: {missing}")
    if not any(name.endswith(".dist-info/METADATA") for name in wheel_names):
        raise AssertionError("Wheel has no METADATA")
    if not any(name.endswith(".dist-info/entry_points.txt") for name in wheel_names):
        raise AssertionError("Wheel has no entry_points.txt")
    packaged_sources = archive.read("care_anxrag/resources/sources.yaml").decode("utf-8")
    packaged_env = archive.read("care_anxrag/resources/env.example").decode("utf-8")
    if packaged_sources != (root / "config" / "sources.yaml").read_text(encoding="utf-8"):
        raise AssertionError("Wheel source registry does not match the release registry")
    if packaged_env != (root / ".env.example").read_text(encoding="utf-8"):
        raise AssertionError("Wheel env template does not match the release template")

with tarfile.open(sdist, "r:gz") as archive:
    sdist_names = set(archive.getnames())
    required_suffixes = {
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "src/care_anxrag/resources/sources.yaml",
        "src/care_anxrag/resources/env.example",
        "config/sources.yaml",
        "docs/VALIDATION.md",
        "scripts/validate.sh",
        "data/benchmark/example.jsonl",
        "examples/local_corpus/synthetic_panic_guidance.md",
        "Dockerfile",
        "compose.yaml",
        "MANIFEST.in",
    }
    missing_suffixes = [
        suffix
        for suffix in sorted(required_suffixes)
        if not any(name.endswith("/" + suffix) for name in sdist_names)
    ]
    if missing_suffixes:
        raise AssertionError(f"Source distribution is missing: {missing_suffixes}")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

report = {
    "status": "passed",
    "wheel": {
        "name": wheel.name,
        "size_bytes": wheel.stat().st_size,
        "sha256": sha256(wheel),
        "file_count": len(wheel_names),
        "required_files_present": True,
    },
    "source_distribution": {
        "name": sdist.name,
        "size_bytes": sdist.stat().st_size,
        "sha256": sha256(sdist),
        "file_count": len(sdist_names),
        "required_files_present": True,
    },
}
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
PY

python -m pip install \
  --disable-pip-version-check \
  --no-deps \
  --target "$TMP/wheel-site" \
  "$WHEEL" > "$TMP/install.log"
PYTHONPATH="$TMP/wheel-site" python -c 'import care_anxrag; import care_anxrag.api; import care_anxrag.cli'
PYTHONPATH="$TMP/wheel-site" python -m care_anxrag --help > "$TMP/cli-help.txt"

WHEEL_PROJECT="$TMP/wheel-project"
mkdir -p "$WHEEL_PROJECT"
PYTHONPATH="$TMP/wheel-site" \
CARE_HOME="$WHEEL_PROJECT/var" \
CARE_DATABASE_PATH="$WHEEL_PROJECT/var/care.sqlite3" \
CARE_VECTOR_PATH="$WHEEL_PROJECT/var/vectors" \
CARE_SOURCE_REGISTRY="$WHEEL_PROJECT/config/sources.yaml" \
CARE_VECTOR_BACKEND=sqlite \
CARE_EMBEDDING_PROVIDER=hash \
CARE_EMBEDDING_MODEL=hash \
CARE_EMBEDDING_DIMENSIONS=256 \
CARE_GENERATOR_PROVIDER=rule \
CARE_RERANKER_PROVIDER=heuristic \
CARE_NLI_PROVIDER=heuristic \
CARE_ALLOW_NETWORK_SYNC=false \
python -m care_anxrag.cli init --project-root "$WHEEL_PROJECT" > "$TMP/wheel-init.json"
PYTHONPATH="$TMP/wheel-site" \
CARE_HOME="$WHEEL_PROJECT/var" \
CARE_DATABASE_PATH="$WHEEL_PROJECT/var/care.sqlite3" \
CARE_VECTOR_PATH="$WHEEL_PROJECT/var/vectors" \
CARE_SOURCE_REGISTRY="$WHEEL_PROJECT/config/sources.yaml" \
CARE_VECTOR_BACKEND=sqlite \
CARE_EMBEDDING_PROVIDER=hash \
CARE_EMBEDDING_MODEL=hash \
CARE_EMBEDDING_DIMENSIONS=256 \
CARE_GENERATOR_PROVIDER=rule \
CARE_RERANKER_PROVIDER=heuristic \
CARE_NLI_PROVIDER=heuristic \
CARE_ALLOW_NETWORK_SYNC=false \
python -m care_anxrag.cli selfcheck --offline --project-root "$WHEEL_PROJECT" > "$TMP/wheel-selfcheck.json"
python - "$TMP/wheel-init.json" "$TMP/wheel-selfcheck.json" "$WHEEL_PROJECT" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

init = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
selfcheck = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
project = Path(sys.argv[3])
if init["health"]["status"] != "ok":
    raise AssertionError(f"Installed wheel init health is not ok: {init}")
if selfcheck["health"]["status"] != "ok":
    raise AssertionError(f"Installed wheel selfcheck health is not ok: {selfcheck}")
if selfcheck["database_integrity"] != "ok":
    raise AssertionError("Installed wheel SQLite integrity check failed")
for required in (project / "config" / "sources.yaml", project / ".env.example"):
    if not required.exists():
        raise AssertionError(f"Installed wheel did not scaffold {required}")
PY

printf '%s\n' "[7/8] Generating the machine-readable release report"
python - "$ROOT" "$TMP" "$WHEEL" "$SDIST" <<'PY'
from __future__ import annotations

import hashlib
import json
import platform
import sqlite3
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

root = Path(sys.argv[1]).resolve()
tmp = Path(sys.argv[2]).resolve()
wheel = Path(sys.argv[3]).resolve()
sdist = Path(sys.argv[4]).resolve()

xml_root = ET.parse(tmp / "junit.xml").getroot()
suites = list(xml_root.iter("testsuite"))
if not suites:
    raise RuntimeError("Could not parse pytest JUnit report")

def total(attribute: str) -> int:
    return sum(int(float(suite.attrib.get(attribute, 0))) for suite in suites)

coverage = json.loads((tmp / "coverage.json").read_text(encoding="utf-8"))
acceptance = json.loads((tmp / "acceptance.json").read_text(encoding="utf-8"))
selfcheck = json.loads((tmp / "selfcheck.json").read_text(encoding="utf-8"))
static_audit = json.loads((tmp / "static-audit.json").read_text(encoding="utf-8"))
package_audit = json.loads((tmp / "package-audit.json").read_text(encoding="utf-8"))
wheel_init = json.loads((tmp / "wheel-init.json").read_text(encoding="utf-8"))
wheel_selfcheck = json.loads((tmp / "wheel-selfcheck.json").read_text(encoding="utf-8"))

def source_tree_sha256() -> str:
    excluded_parts = {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "build",
        "dist",
        "care_anxrag.egg-info",
    }
    excluded_names = {".coverage", "validation-report.json", "SHA256SUMS"}
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in excluded_parts for part in relative.parts):
            continue
        if path.name in excluded_names or path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

report = {
    "generated_at": datetime.now(tz=UTC).isoformat(),
    "status": "passed",
    "project": {"name": "care-anxrag", "version": "0.1.0"},
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sqlite": sqlite3.sqlite_version,
    },
    "static_audit": static_audit,
    "tests": {
        "total": total("tests"),
        "failures": total("failures"),
        "errors": total("errors"),
        "skipped": total("skipped"),
    },
    "coverage": {
        "percent_covered": coverage["totals"]["percent_covered"],
        "covered_lines": coverage["totals"]["covered_lines"],
        "missing_lines": coverage["totals"]["missing_lines"],
        "num_statements": coverage["totals"]["num_statements"],
        "required_minimum_percent": 70,
    },
    "selfcheck": selfcheck,
    "offline_acceptance": acceptance,
    "package": {
        **package_audit,
        "isolated_target_install": "passed",
        "import_and_cli": "passed",
        "installed_wheel_init": wheel_init,
        "installed_wheel_selfcheck": wheel_selfcheck,
    },
    "release_source_tree_sha256": source_tree_sha256(),
    "not_executed_in_this_offline_environment": [
        "live NIMH/WHO/PubMed/PMC/NICE synchronization",
        "licensed NICE endpoint authentication",
        "real Chroma persistence with chromadb",
        "real Ollama embedding and generation",
        "real Sentence Transformers reranker/NLI inference",
        "clinical correctness, clinical safety, or regulatory validation",
        "online dependency resolution and vulnerability scanning",
    ],
}
if report["tests"]["failures"] or report["tests"]["errors"]:
    raise AssertionError("Test report contains failures or errors")
if report["coverage"]["percent_covered"] < 70:
    raise AssertionError("Coverage is below the release floor")
(root / "validation-report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2, ensure_ascii=False))
PY

printf '%s\n' "[8/8] Writing cryptographic checksums"
(
  cd "$ROOT"
  sha256sum \
    "dist/$(basename "$WHEEL")" \
    "dist/$(basename "$SDIST")" \
    validation-report.json > SHA256SUMS
)

printf '%s\n' "Validation passed. Report: $ROOT/validation-report.json"
printf '%s\n' "Checksums: $ROOT/SHA256SUMS"
