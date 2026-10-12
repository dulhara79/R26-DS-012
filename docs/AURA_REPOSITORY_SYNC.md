# Aura-Biomonitor repository mirror

The source repository \`DewduSendanayake/Aura-Biomonitor\` owns ESP32-C3 chest-strap firmware. A source-side GitHub Action proposes updates under:

\`Self-supervised Physiological Biosensors/Aura-Biomonitor/\`

It never writes to destination \`main\` directly. The PR branch is \`sync/aura-biomonitor\`. It is strictly bot-owned and rebuilt from latest destination \`main\`, so **do not put manual commits on this branch**.

The importer requires an ownership manifest \`.aura-sync-state.json\` recording imported files and their hashes. If target files are manually edited, if extra files appear there, or if source open PRs conflict, reconciliation fails instead of overwriting those changes. This restriction is limited to the Aura directory; other Component 1 files remain owned by their normal maintainers.

Before production use, enable branch protection on \`main\` (reviews, checks, no force pushes). Add the \`Guard Aura firmware mirror scope\` job as a required check if practical. Configure \`AURA_DESTINATION_TOKEN\` in the **source** repository with least privilege. The source must merge its workflow PR before any trigger can run reliably. The source implementation and setup details live at \`docs/REPOSITORY_SYNC.md\`.
