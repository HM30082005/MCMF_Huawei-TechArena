#!/usr/bin/env bash
# One-time setup of the autonomous research lab on this machine.
#
#   research/install_lab.sh MAIN_CHECKOUT
#
# MAIN_CHECKOUT holds the data, encoded cache, .venv and baseline artifacts. Creates:
#   MAIN_CHECKOUT/lab                  git worktree on branch hermes/research
#   ~/.hermes/techarena/harness        frozen, read-only copy of the evaluator
#   ~/.hermes/techarena/store          append-only scores and generated leaderboard
#   ~/.hermes/scripts/techarena_monitor.py   per-tick monitor (read-only)
# and protects the baseline and strategy-2 feature caches by making them read-only.
set -euo pipefail
main=$(cd "${1:?main checkout required}" && pwd)
src=$(cd "$(dirname "$0")/.." && pwd)
branch_base=$(git -C "$src" rev-parse HEAD)
lab=$main/lab
home=$HOME/.hermes/techarena
harness=$home/harness
store=$home/store
py=$main/.venv/bin/python
monitor=$HOME/.hermes/scripts/techarena_monitor.py

chmod -R a-w "$main/artifacts/history_upgrade" "$main/artifacts/strategy2"

if [[ ! -d $lab ]]; then
  git -C "$main" worktree add -q -b hermes/research "$lab" "$branch_base"
fi
grep -qx '/lab/' "$(git -C "$main" rev-parse --git-common-dir)/info/exclude" 2>/dev/null \
  || echo '/lab/' >> "$(git -C "$main" rev-parse --git-common-dir)/info/exclude"
mkdir -p "$lab/artifacts" "$lab/runs" "$lab/analysis"
ln -sfn "$main/analysis/.cache" "$lab/analysis/.cache"
ln -sfn "$main/train.csv" "$lab/train.csv"
ln -sfn "$main/evaluation.csv" "$lab/evaluation.csv"
ln -sfn "$main/artifacts/strategy2" "$lab/artifacts/strategy2"
ln -sfn "$main/artifacts/history_upgrade" "$lab/artifacts/history_upgrade"
ln -sfn "$main/.venv" "$lab/.venv"

# Frozen evaluator: a read-only copy that the monitor runs; the lab's own copy is not used for scores.
mkdir -p "$harness/research" "$harness/analysis" "$store" "$(dirname "$monitor")"
chmod -R u+w "$harness" 2>/dev/null || true
cp "$src/research/evaluate.py" "$harness/research/evaluate.py"
ln -sfn "$main/analysis/.cache" "$harness/analysis/.cache"
chmod u+w "$monitor" 2>/dev/null || true
cp "$src/research/monitor.py" "$monitor"
chmod a-w "$harness/research/evaluate.py" "$harness/research" "$monitor"

"$py" - "$home/config.json" <<EOF
import hashlib, json, sys
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
official = json.load(open('$main/results/official_scores.json'))['scores'][0]
json.dump(dict(lab='$lab', store='$store', harness='$harness', python='$py', job_name='techarena-research',
               disk_budget_gb=100, baseline_dir='$main/artifacts/history_upgrade',
               baseline_csv='$main/' + official['submission_file'], baseline_sha256=official['submission_sha256'],
               harness_manifest={'research/evaluate.py': sha('$harness/research/evaluate.py')}),
          open(sys.argv[1], 'w'), indent=2)
EOF

# Reference run: reproduce the baseline from saved trees and score it through the frozen evaluator.
if [[ ! -f $lab/runs/baseline-0.816323/run.json ]]; then
  (cd "$lab" && "$py" research/reproduce_baseline.py --runs-dir "$lab/runs" --store "$store")
fi
"$py" "$harness/research/evaluate.py" score "$lab/runs/baseline-0.816323" --confirm --store "$store" --copy-to "$lab/LEADERBOARD.md"
touch "$lab/runs/baseline-0.816323/SCORED"
echo "Lab ready at $lab"
