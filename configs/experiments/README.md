# Recorded experiment configurations

These small JSON manifests were exported from the local research runs on November 6, 2026. They preserve recorded commands, recipes, seed selection, source commit IDs, and the blend definition. They contain no trained weights, raw rows, or prediction arrays.

They are evidence and replay references, **not inputs consumed by a JSON configuration loader**. Use `src/history_upgrade.py --help` to translate a recipe into CLI flags. A recorded command may stop at development because confirmation and final fitting were subsequently resumed with the same model settings.

- `t3-budget-seeds30.json`: 30-seed, 3,000-update recipe.
- `t4-tree-refit.json`: the same representation with fitting-period tree selection. Its recorded parent field is inconsistent with its hypothesis; see the research summary.
- `claude-s3000-init005-seeds20.json`: 20-seed component.
- `s14-seeds50-stale24h.json`: historical 50-seed auxiliary-context component; above the current launcher's 30-seed search cap.
- `blend-refit-seeds-stale.json`: four component IDs, equal logit weights, and submission checksum.

Source commit IDs describe the original local research history and may not resolve on GitHub. See [results and reproduction limits](../../docs/experiments/research-summary.md).
