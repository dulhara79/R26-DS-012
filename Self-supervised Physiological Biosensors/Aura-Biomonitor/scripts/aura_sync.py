#!/usr/bin/env python3
"""Fail-closed rolling mirror of source main plus compatible same-repo PRs.

This script must execute from the trusted DEFAULT branch of Aura-Biomonitor,
never from a pull request head. Source files are handled as data, not executed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen

SOURCE = "DewduSendanayake/Aura-Biomonitor"
DESTINATION = "dulhara79/R26-DS-012"
TARGET = Path("Self-supervised Physiological Biosensors/Aura-Biomonitor")
BRANCH = "sync/aura-biomonitor"
STATE_FILE = ".aura-sync-state.json"
SCHEMA = 1


class SyncError(RuntimeError):
    pass


def git(*args: str, cwd: Path | None = None, allow_failure: bool = False) -> str:
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", *args],
        cwd=cwd, text=True, capture_output=True, check=False,
    )
    if result.returncode and not allow_failure:
        raise SyncError(f"git {args[0]} failed: {result.stderr.strip()[-1500:]}")
    return result.stdout.strip()


def cmd(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(list(args), cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise SyncError(f"{args[0]} failed: {result.stderr.strip()[-1500:]}")
    return result.stdout.strip()


def source_prs() -> list[dict]:
    """Return stable, eligible open PR metadata using the source read-only token."""
    token = os.environ["SOURCE_READ_TOKEN"]
    found = []
    for page in range(1, 11):
        url = (
            f"https://api.github.com/repos/{SOURCE}/pulls"
            f"?state=open&base=main&per_page=100&page={page}"
        )
        req = Request(url, headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "aura-rolling-sync",
        })
        with urlopen(req, timeout=25) as response:
            items = json.load(response)
        for pr in items:
            head_repo = (pr.get("head", {}).get("repo") or {}).get("full_name")
            if head_repo != SOURCE or pr.get("draft"):
                continue  # Never use fork code or draft proposals in a privileged job.
            found.append({
                "number": int(pr["number"]),
                "sha": pr["head"]["sha"],
                "url": pr["html_url"],
            })
        if len(items) < 100:
            return sorted(found, key=lambda x: x["number"])
    raise SyncError("More than 1,000 open source PRs; refusing partial inventory")


def checked_source_files(root: Path) -> dict[str, str]:
    """Inventory source files without following links or including root .github."""
    result: dict[str, str] = {}
    for current, dirs, files in os.walk(root, followlinks=False):
        relative = Path(current).relative_to(root)
        if relative == Path("."):
            dirs[:] = [d for d in dirs if d not in (".git", ".github")]
        for directory in dirs:
            if (Path(current) / directory).is_symlink():
                raise SyncError(f"Source symlink directory forbidden: {directory}")
        for filename in files:
            path = Path(current) / filename
            name = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.is_file():
                raise SyncError(f"Non-regular source file forbidden: {name}")
            if name == STATE_FILE:
                raise SyncError(f"Source conflicts with reserved {STATE_FILE}")
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not result or "README.md" not in result or not any(
        p.startswith("ChestStrap_V3/") for p in result
    ):
        raise SyncError("Source firmware appears empty or incomplete; refusing sync")
    return dict(sorted(result.items()))


def validate_existing_target(target: Path) -> None:
    """Reject local edits, additions, missing files, or a pre-existing unowned folder."""
    if not target.exists() and not target.is_symlink():
        return
    if target.is_symlink() or not target.is_dir():
        raise SyncError("Destination target is not a normal directory")
    state_path = target / STATE_FILE
    if state_path.is_symlink() or not state_path.is_file():
        raise SyncError("Target exists without a valid ownership manifest; refusing overwrite")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        raise SyncError(f"Invalid ownership manifest: {ex}") from ex
    if (
        state.get("schema") != SCHEMA
        or state.get("source") != SOURCE
        or state.get("destination_path") != TARGET.as_posix()
        or not isinstance(state.get("files"), dict)
    ):
        raise SyncError("Destination manifest provenance does not match this mirror")
    inventory: dict[str, str] = {}
    for current, dirs, files in os.walk(target, followlinks=False):
        for directory in dirs:
            if (Path(current) / directory).is_symlink():
                raise SyncError("Destination contains an unexpected symbolic link")
        for filename in files:
            path = Path(current) / filename
            relative = path.relative_to(target).as_posix()
            if relative == STATE_FILE:
                continue
            if path.is_symlink() or not path.is_file():
                raise SyncError(f"Unexpected destination file type: {relative}")
            inventory[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if inventory != state["files"]:
        added = sorted(set(inventory) - set(state["files"]))
        missing = sorted(set(state["files"]) - set(inventory))
        modified = sorted(
            p for p in inventory.keys() & state["files"].keys()
            if inventory[p] != state["files"][p]
        )
        raise SyncError(
            "Destination firmware diverged from the last approved mirror. "
            f"Extra={added[:8]}, missing={missing[:8]}, changed={modified[:8]}. "
            "Resolve changes in the source repository first."
        )


def write_snapshot(source: Path, destination: Path, main_sha: str,
                   prs: list[dict]) -> None:
    source_files = checked_source_files(source)
    target = destination / TARGET
    validate_existing_target(target)
    # Only operate on our exclusive path, in a throwaway clone; never on remote main.
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for relative in source_files:
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, output)
    metadata = {
        "schema": SCHEMA,
        "source": SOURCE,
        "destination_path": TARGET.as_posix(),
        "base_main_sha": main_sha,
        "included_open_prs": [{"number": p["number"], "sha": p["sha"]} for p in prs],
        "files": source_files,
    }
    (target / STATE_FILE).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def branch_head(repo: Path) -> str | None:
    output = git("ls-remote", "--heads", "origin", f"refs/heads/{BRANCH}", cwd=repo)
    if not output:
        return None
    return output.split()[0]


def existing_pr() -> dict | None:
    payload = cmd(
        "gh", "pr", "list", "--repo", DESTINATION, "--state", "open",
        "--head", BRANCH, "--base", "main", "--json", "number,url", "--limit", "100",
    )
    prs = json.loads(payload)
    if len(prs) > 1:
        raise SyncError("Multiple open destination PRs use the same sync branch")
    return prs[0] if prs else None


def perform() -> None:
    if not os.environ.get("SOURCE_READ_TOKEN"):
        raise SyncError("Missing SOURCE_READ_TOKEN")
    if not os.environ.get("GH_TOKEN"):
        raise SyncError("Missing GH_TOKEN destination credential")

    with tempfile.TemporaryDirectory(prefix="aura-sync-") as workspace:
        work = Path(workspace)
        source = work / "source"
        destination = work / "destination"

        # Source PR content is ONLY merged as inert files; no source code or
        # PR-supplied workflows are executed with the destination credential.
        git("clone", "--quiet", "--branch", "main",
            f"https://github.com/{SOURCE}.git", str(source))
        git("config", "user.name", "Aura sync integration", cwd=source)
        git("config", "user.email", "aura-sync@users.noreply.github.com", cwd=source)
        main_sha = git("rev-parse", "HEAD", cwd=source)
        prs = source_prs()
        for pr in prs:
            git("fetch", "--quiet", "origin", f"refs/pull/{pr['number']}/head", cwd=source)
            fetched = git("rev-parse", "FETCH_HEAD", cwd=source)
            if fetched != pr["sha"]:
                raise SyncError(f"Source PR #{pr['number']} moved during reconciliation; retry")
            try:
                git("merge", "--no-ff", "--no-edit", fetched, cwd=source)
            except SyncError as ex:
                raise SyncError(
                    f"Source PR #{pr['number']} conflicts with main or another open PR. "
                    "Previous destination PR remains untouched. Resolve the source "
                    f"conflict and rerun. Details: {ex}"
                ) from ex

        # Source main advancing mid-run must not generate a stale snapshot.
        remote_main = git("ls-remote", "origin", "refs/heads/main", cwd=source).split()[0]
        if main_sha != remote_main:
            raise SyncError("Source main advanced while reconciling; retry")

        git("clone", "--quiet", "--branch", "main",
            f"https://github.com/{DESTINATION}.git", str(destination))
        write_snapshot(source, destination, main_sha, prs)
        git("add", "-A", "--", TARGET.as_posix(), cwd=destination)
        staged = git("diff", "--cached", "--name-only", "--", cwd=destination)
        paths = [p for p in staged.splitlines() if p]
        prefix = TARGET.as_posix() + "/"
        if any(not p.startswith(prefix) for p in paths):
            raise SyncError("Unexpected file outside the owned component scope")

        active = existing_pr()
        if not paths:
            if active:
                cmd("gh", "pr", "close", str(active["number"]), "--repo",
                    DESTINATION, "--comment",
                    "The source snapshot now matches destination main; closing this "
                    "no-longer-needed sync PR. Nothing was deleted from main.")
            print("Destination main already matches source; no changes required")
            return

        # Branch belongs to automation. Overwrite only under an expected SHA lease.
        previous = branch_head(destination)
        git("switch", "-c", BRANCH, cwd=destination)
        git("config", "user.name", "github-actions[bot]", cwd=destination)
        git("config", "user.email",
            "41898282+github-actions[bot]@users.noreply.github.com", cwd=destination)
        git("commit", "-m",
            f"chore(aura): reconcile main {main_sha[:12]} and {len(prs)} open PR(s)",
            cwd=destination)

        # Detect dest main moving after checkout. Scheduled reconciliation recovers.
        remote_dest_main = git(
            "ls-remote", "origin", "refs/heads/main", cwd=destination
        ).split()[0]
        local_dest_main = git("rev-parse", "main", cwd=destination)
        if remote_dest_main != local_dest_main:
            raise SyncError("Destination main advanced during sync; retry")

        dest_ref = f"refs/heads/{BRANCH}"
        if previous:
            git("push", f"--force-with-lease={dest_ref}:{previous}",
                "origin", f"HEAD:{dest_ref}", cwd=destination)
        else:
            git("push", "origin", f"HEAD:{dest_ref}", cwd=destination)

        lines = [
            "## Automated Aura-Biomonitor rolling mirror",
            "",
            f"**Source:** https://github.com/{SOURCE}",
            f"**Path:** \`{TARGET.as_posix()}/\`",
            f"**Source main:** \`{main_sha}\`",
            "",
            "Included non-draft, same-repository source PRs (ascending PR order):",
        ]
        lines += [f"- {pr['url']} (\`{pr['sha'][:12]}\`)" for pr in prs] or [
            "- None; only source main is included."
        ]
        lines += [
            "", "### Review safeguards",
            "- No direct push or auto-merge into destination main.",
            "- Files outside the exclusive Aura-Biomonitor directory are untouched.",
            "- Source conflicts and local destination divergence abort the sync.",
            "- Deletes inside the owned directory are visible in this PR.",
            "- The branch is regenerated from the latest destination main.",
            "- Root source .github workflows and Git metadata are intentionally omitted.",
            "- Inspect the included proposals before merging: these may not yet",
            "  be approved in the source repository.",
        ]
        body = "\n".join(lines)
        if active:
            cmd("gh", "api", "-X", "PATCH",
                f"repos/{DESTINATION}/pulls/{active['number']}",
                "-f", f"body={body}")
            print(f"Updated destination PR: {active['url']}")
        else:
            url = cmd(
                "gh", "pr", "create", "--repo", DESTINATION, "--base", "main",
                "--head", BRANCH, "--title",
                "chore(aura): rolling firmware mirror from Aura-Biomonitor",
                "--body", body,
            )
            print(f"Created destination PR: {url}")


if __name__ == "__main__":
    try:
        perform()
    except (SyncError, OSError, KeyError, ValueError) as error:
        print(f"::error::{error}", file=sys.stderr)
        sys.exit(1)
