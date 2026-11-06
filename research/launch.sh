#!/usr/bin/env bash
# Launch one experiment as a detached background job with a recorded identity.
#
#   research/launch.sh --parent RUN RUN_ID "one-sentence hypothesis" -- COMMAND [ARGS...]
#   research/launch.sh --clone-from RUN RUN_ID "one-sentence hypothesis" -- COMMAND [ARGS...]
#   research/launch.sh --resume RUN_ID -- COMMAND [ARGS...]
#
# research/preflight.py checks the command first, and a refused launch creates nothing.
#
# The command runs from the repository root with RUN_DIR=runs/RUN_ID exported; pass
# --output-dir "$RUN_DIR" (or the literal path) to the pipeline. --clone-mf copies the
# baseline's MF models and parts into the new run as APFS clones, for experiments that
# keep the default MF recipe; --clone-from does the same from another run, for tree-only
# experiments on that run's MF recipe. Only one job may run at a time (the GPU is shared).
# Jobs run with TECHARENA_HIDE_CONFIRMATION=1: only the shared evaluator reports Jan 26.
#
# The job runs in its own process session, so it survives the shell or agent turn that
# started it. The Mac is kept awake while it runs. The script waits a few seconds and
# exits non-zero if the job has already died.
set -euo pipefail
cd "$(dirname "$0")/.."
resume=0 clone=0 clone_from= parent= combine=0
while [[ "${1:-}" == --* && "${1:-}" != -- ]]; do
  case $1 in
    --resume) resume=1 ;;
    --clone-mf) clone=1 ;;
    --clone-from) clone_from=${2:?run id required}; shift ;;
    --parent) parent=${2:?run id required}; shift ;;
    --combine) combine=1 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
run_id=${1:?run id required}; shift
if [[ $resume == 0 ]]; then hypothesis=${1:?hypothesis required}; shift; fi
[[ "${1:-}" == -- ]] || { echo "usage: launch.sh [--parent RUN|--clone-mf|--clone-from RUN|--resume] [--combine] RUN_ID [HYPOTHESIS] -- COMMAND..." >&2; exit 2; }
shift
[[ $# -gt 0 ]] || { echo "no command given" >&2; exit 2; }
[[ $run_id =~ ^[a-z0-9][a-z0-9._-]{2,80}$ ]] || { echo "run id must be lowercase [a-z0-9._-]" >&2; exit 2; }
run_dir=runs/$run_id
lock=runs/.job.lock
mkdir -p runs
if [[ -f $lock ]] && kill -0 "$(cut -d' ' -f1 "$lock")" 2>/dev/null; then
  echo "Another job is running: $(cut -d' ' -f2 "$lock")" >&2; exit 3
fi
check=(.venv/bin/python research/preflight.py)
[[ $resume == 0 ]] || check+=(--resume)
[[ $combine == 0 ]] || check+=(--combine)
[[ $clone == 0 ]] || check+=(--clone-mf)
[[ -z $clone_from ]] || check+=(--clone-from "$clone_from")
[[ -z $parent ]] || check+=(--parent "$parent")
"${check[@]}" "$run_id" -- "$@"
if [[ $resume == 1 ]]; then
  rm -f "$run_dir/EXIT" "$run_dir/SCORED"
else
  if [[ -n $clone_from ]]; then
    [[ -d runs/$clone_from/models && -d runs/$clone_from/parts ]] || { echo "runs/$clone_from has no MF models and parts" >&2; exit 2; }
  fi
  mkdir -p "$run_dir"
  if [[ -n $clone_from ]]; then
    cp -c -R "runs/$clone_from/models" "runs/$clone_from/parts" "$run_dir/"
    [[ ! -d runs/$clone_from/features ]] || cp -c -R "runs/$clone_from/features" "$run_dir/"
    chmod -R u+w "$run_dir"
  elif [[ $clone == 1 ]]; then
    cp -c -R artifacts/history_upgrade/models artifacts/history_upgrade/parts "$run_dir/"
    chmod -R u+w "$run_dir"
  fi
  python3 - "$run_dir/run.json" "$run_id" "$hypothesis" "$(git rev-parse HEAD)" \
    "$(git status --porcelain | wc -l | tr -d ' ')" "$clone" "$clone_from" "${parent:-${clone_from:-baseline-0.816323}}" "$*" <<'EOF'
import json, sys, datetime
path, run_id, hypothesis, commit, dirty, clone, clone_from, parent, command = sys.argv[1:]
json.dump(dict(id=run_id, hypothesis=hypothesis, parent=parent, git_commit=commit,
               uncommitted_files=int(dirty), cloned_baseline_mf=clone == '1', cloned_from=clone_from or None, command=command,
               created=datetime.datetime.now(datetime.timezone.utc).isoformat()), open(path, 'w'), indent=2)
EOF
fi
printf '%s\n' "$*" >> "$run_dir/commands.txt"
export RUN_DIR=$run_dir TECHARENA_HIDE_CONFIRMATION=1
# start_new_session detaches the job from this shell's session and process group.
pid=$(python3 - "$run_dir" "$@" <<'EOF'
import os, subprocess, sys
run_dir, command = sys.argv[1], sys.argv[2:]
script = '"$@"; code=$?; echo $code > "$RUN_DIR/EXIT"; rm -f runs/.job.lock; exit $code'
with open(os.path.join(run_dir, 'log.txt'), 'ab') as log:
    job = subprocess.Popen(['bash', '-c', script, '_', *command], stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
subprocess.Popen(['caffeinate', '-i', '-w', str(job.pid)], stdin=subprocess.DEVNULL,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
print(job.pid)
EOF
)
echo "$pid $run_id $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$lock"
echo "$pid" > "$run_dir/PID"
sleep 5
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Job $run_id died within 5 s (exit $(cat "$run_dir/EXIT" 2>/dev/null || echo '?')). Log tail:" >&2
  tail -20 "$run_dir/log.txt" >&2
  exit 4
fi
echo "Launched $run_id (pid $pid, detached). Log: $run_dir/log.txt"
tail -3 "$run_dir/log.txt"
