# Aura-Biomonitor → R26-DS-012 synchronization

The workflow in \`.github/workflows/mirror-to-r26-ds-012.yml\` continuously mirrors a **review proposal**, not a direct write to destination \`main\`.

## Setup (required before this can run)

1. Review and merge the source implementation PR into \`DewduSendanayake/Aura-Biomonitor\` \`main\`.
2. In that source repository, create an Actions secret called \`AURA_DESTINATION_TOKEN\`. Use a **fine-grained personal access token** owned by an account with write access to \`dulhara79/R26-DS-012\` and scoped ONLY to the destination repository, with **Contents: Read and write** and **Pull requests: Read and write**. An installation token from a narrowly configured GitHub App is preferred for a longer-lived deployment. Do not commit a token into either repository.
3. Ensure the destination PR can be reviewed under branch protection rules. The destination guard workflow and CODEOWNERS update are supplied by a separate destination PR; merge those before enabling routine syncs.
4. Run the source workflow using **Actions → Mirror Aura firmware to R26-DS-012 → Run workflow**. Check for a PR in destination \`main\` from \`sync/aura-biomonitor\`.
5. Require at least one independent reviewer on destination \`main\`. Do not enable automatic merge for this sync PR.

## Behavior

- Triggered by source PR openings, updates, close/merge, and draft transitions; hourly schedule and manual dispatch heal missed events.
- Starts from source \`main\` and merges compatible, open, non-draft **same-repository** PRs in ascending PR number. Fork PRs and drafts are excluded.
- If candidate PRs conflict, the workflow fails **before updating the destination branch**, leaving any older destination PR intact. Fix the conflict in the source first.
- Records exact source \`main\` SHA, included PR SHAs and SHA-256 hashes in \`Self-supervised Physiological Biosensors/Aura-Biomonitor/.aura-sync-state.json\`.
- Checks destination \`main\` against the last approved mirror manifest before writing. Manual changes or unknown files **inside the owned target** cause failure, not silent deletion.
- Rebuilds the rolling destination sync branch from current destination \`main\`. Push uses \`--force-with-lease\` on the designated automation-owned branch only. Nobody should add commits to this branch; apply changes to source instead.
- Removes files only **within the isolated owned directory** when they are removed from source, and displays those removals in the destination PR. Other component directories, root files, and workflows remain untouched.
- Source root \`.github/\` and \`.git/\` metadata are omitted; physical hardware firmware and source README stay together in the imported folder.
- A source PR is **not** considered reviewed/approved just because it appears in destination. Destination maintainers should check which open source PRs are included before merging the rolling PR.

## Recovery and governance

- Unmerged destination PR: next run updates the **same** branch and PR.
- Stale/conflicting PR: fix conflicts in source, then rerun manually. This never fabricates a resolved merge.
- Source PR closed without merge: next run removes its proposed changes unless they are also present on main or another candidate.
- Destination independent edits under Aura path: move them upstream first and resolve intentionally; the mirror fails closed.
- Destination PR already merged: next changes make a new PR on the same recurring branch.
- Avoid including secrets, collected human physiological data, patient data, firmware signing keys, or other sensitive research artifacts in the public source repository: mirrored content will also be public.
- GitHub branch protection and secret configuration are repository settings and must be verified by an administrator; workflows alone cannot stop an administrator bypassing protection.

## Design limitations

This is a **preview mirror**: compatible open source PRs are included before source merge, as specifically requested. Consequently source CI and human approval may still be pending. A destination merge may publish a candidate that is later closed or revised upstream. For release-grade archival provenance, prefer using *merged source main only* and triggering on the source merge event. Avoid treating this mirror as a clinical or safety-validated release.
