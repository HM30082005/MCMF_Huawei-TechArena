#!/usr/bin/env bash
# Launch one experiment as a detached background job with a recorded identity.
#
#   research/launch.sh RUN_ID "one-sentence hypothesis" -- COMMAND [ARGS...]
#   research/launch.sh --resume RUN_ID -- COMMAND [ARGS...]
#
# The command runs from the repository root with RUN_DIR=runs/RUN_ID exported; pass
# --output-dir "$RUN_DIR" (or the literal path) to the pipeline. Only one job may
# run at a time (the GPU is shared). The Mac is kept awake while the job runs.
set -euo pipefail
cd "$(dirname "$0")/.."
resume=0
if [[ "${1:-}" == --resume ]]; then resume=1; shift; fi
run_id=${1:?run id required}; shift
if [[ $resume == 0 ]]; then hypothesis=${1:?hypothesis required}; shift; fi
[[ "${1:-}" == -- ]] || { echo "usage: launch.sh [--resume] RUN_ID [HYPOTHESIS] -- COMMAND..." >&2; exit 2; }
shift
[[ $run_id =~ ^[a-z0-9][a-z0-9._-]{2,80}$ ]] || { echo "run id must be lowercase [a-z0-9._-]" >&2; exit 2; }
run_dir=runs/$run_id
lock=runs/.job.lock
mkdir -p runs
if [[ -f $lock ]] && kill -0 "$(cut -d' ' -f1 "$lock")" 2>/dev/null; then
  echo "Another job is running: $(cut -d' ' -f2 "$lock")" >&2; exit 3
fi
if [[ $resume == 1 ]]; then
  [[ -f $run_dir/run.json ]] || { echo "$run_dir does not exist" >&2; exit 2; }
  rm -f "$run_dir/EXIT" "$run_dir/SCORED"
else
  [[ ! -e $run_dir ]] || { echo "$run_dir already exists; choose a new run id" >&2; exit 2; }
  mkdir -p "$run_dir"
  python3 - "$run_dir/run.json" "$run_id" "$hypothesis" "$(git rev-parse HEAD)" \
    "$(git status --porcelain | wc -l | tr -d ' ')" "$*" <<'EOF'
import json, sys, datetime
path, run_id, hypothesis, commit, dirty, command = sys.argv[1:]
json.dump(dict(id=run_id, hypothesis=hypothesis, parent='baseline-0.816323', git_commit=commit,
               uncommitted_files=int(dirty), command=command,
               created=datetime.datetime.now(datetime.timezone.utc).isoformat()), open(path, 'w'), indent=2)
EOF
fi
printf '%s\n' "$*" >> "$run_dir/commands.txt"
export RUN_DIR=$run_dir
nohup bash -c '"$@"; code=$?; echo $code > "$RUN_DIR/EXIT"; rm -f runs/.job.lock; exit $code' _ "$@" \
  >> "$run_dir/log.txt" 2>&1 < /dev/null &
pid=$!
echo "$pid $run_id $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$lock"
echo "$pid" > "$run_dir/PID"
command -v caffeinate >/dev/null && nohup caffeinate -i -w "$pid" >/dev/null 2>&1 &
echo "Launched $run_id (pid $pid); log: $run_dir/log.txt"
