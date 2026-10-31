"""Per-tick lab monitor for the Hermes research job (runs before each agent turn).

Scores finished runs with the frozen evaluator, honours confirmation requests,
checks the integrity of the baseline and harness, sends Telegram alerts, and
prints a status block. Hermes skips the agent turn when this output is unchanged,
so the output is stable while a job runs and changes when there is work to do.

Configuration: ~/.hermes/techarena/config.json (written by research/install_lab.py).
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

CONFIG = json.loads((Path.home() / '.hermes' / 'techarena' / 'config.json').read_text())
LAB, STORE, HARNESS = Path(CONFIG['lab']), Path(CONFIG['store']), Path(CONFIG['harness'])
RUNS = LAB / 'runs'
PY = CONFIG['python']
STATE = STORE / 'monitor_state.json'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def notify(state, key, text):
    """Send each distinct alert once."""
    if state['sent'].get(key) == text:
        return
    try:
        sent = subprocess.run(['hermes', 'send', '-t', 'telegram', text], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        sent = False  # A failed alert must never stop scoring; retried on the next tick.
    if sent:
        state['sent'][key] = text


def integrity():
    problems = []
    if sha256(CONFIG['baseline_csv']) != CONFIG['baseline_sha256']:
        problems.append('baseline submission CSV changed')
    if os.access(CONFIG['baseline_dir'], os.W_OK):
        problems.append('baseline artifact directory is writable again')
    for rel, digest in CONFIG['harness_manifest'].items():
        path = HARNESS / rel
        if not path.exists() or sha256(path) != digest:
            problems.append('frozen harness file changed: ' + rel)
    return problems


def evaluate(*args):
    result = subprocess.run([PY, str(HARNESS / 'research' / 'evaluate.py'), *args, '--store', str(STORE),
                             '--copy-to', str(LAB / 'LEADERBOARD.md')],
                            capture_output=True, text=True, timeout=1800)
    return result.returncode, (result.stdout + result.stderr).strip().splitlines()[-1:] or ['']


def running_job():
    lock = RUNS / '.job.lock'
    if not lock.exists():
        return None
    pid, run_id, started = lock.read_text().split()[:3]
    try:
        os.kill(int(pid), 0)
    except OSError:
        return None
    return run_id, started


def main():
    STORE.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {'sent': {}}
    out = []
    problems = integrity()
    if problems:
        out.append('INTEGRITY FAILURE: ' + '; '.join(problems))
        out.append('Stop all work. Do not launch or score anything. The job has been paused for the user.')
        notify(state, 'integrity', 'TechArena lab INTEGRITY FAILURE, job paused: ' + '; '.join(problems))
        subprocess.run(['hermes', 'cron', 'pause', CONFIG['job_name']], capture_output=True, timeout=60)
        STATE.write_text(json.dumps(state))
        print('\n'.join(out))
        return

    events = []
    for run in sorted(RUNS.glob('*/')):
        exit_file, scored = run / 'EXIT', run / 'SCORED'
        if exit_file.exists() and not scored.exists():
            code = exit_file.read_text().strip()
            if code != '0':
                events.append(f'FAILED {run.name}: exit {code}; see runs/{run.name}/log.txt')
                scored.write_text('failed\n')
                continue
            if (run / 'development_predictions.npz').exists():
                rc, tail = evaluate('score', str(run))
                events.append(f'SCORED {run.name} (development): ' + (tail[0] if rc == 0 else 'ERROR ' + tail[0]))
            else:
                events.append(f'FINISHED {run.name}: no development_predictions.npz to score')
            scored.write_text('ok\n')
        request = run / 'REQUEST_CONFIRMATION'
        if request.exists():
            rc, tail = evaluate('score', str(run), '--confirm')
            events.append(f'CONFIRMATION {run.name}: ' + (tail[0] if rc == 0 else 'REFUSED ' + tail[0]))
            request.unlink()

    board = LAB / 'LEADERBOARD.md'
    if not board.exists() and (STORE / 'scores.jsonl').exists():
        evaluate('leaderboard')
    candidates = []
    if board.exists():
        text = board.read_text()
        for line in text.splitlines():
            if 'SUBMIT CANDIDATE' in line:
                run_id = line.split('`')[1]
                csvs = sorted((RUNS / run_id).glob('submission*.csv'))
                detail = ', '.join(f'{p.name} sha256 {sha256(p)}' for p in csvs) or 'no full-data CSV yet'
                candidates.append(f'{run_id}: {detail}')
                notify(state, 'candidate:' + run_id + detail,
                       f'TechArena: new submission candidate {run_id} ({detail}). See lab/LEADERBOARD.md')
        top = next((l for l in text.splitlines() if l.startswith('| 1 |')), '')
        if top and state.get('top') != top.split('|')[2]:
            if state.get('top') is not None:
                notify(state, 'top', 'TechArena leaderboard #1 is now ' + top.split('|')[2].strip())
            state['top'] = top.split('|')[2]

    used = sum(f.stat().st_size for f in RUNS.rglob('*') if f.is_file() and not f.is_symlink()) / 1e9
    budget = CONFIG['disk_budget_gb']
    free = shutil.disk_usage(LAB).free / 1e9

    job = running_job()
    out.append('# TechArena lab status')
    out.extend(events)
    if candidates:
        out.append('Submission candidates: ' + ' | '.join(candidates))
    if used > budget:
        out.append(f'DISK BUDGET EXCEEDED: runs/ uses {used:.0f} GB of {budget} GB. Prune features/ and parts/ '
                   'of runs that are not submission candidates before launching anything.')
    if free < 30:
        out.append(f'LOW DISK: {free:.0f} GB free on the volume. Prune before launching.')
    if job:
        run_id, started = job
        hours = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(started.replace('Z', '+00:00'))).total_seconds() / 3600
        out.append(f'RUNNING {run_id} (started {started}, about {int(hours)} h elapsed). Nothing to do until it finishes.')
        if hours >= 3:
            notify(state, 'stalled:' + run_id, f'TechArena: run {run_id} has been running for {int(hours)} h; check it.')
    else:
        # Idle: include the tick time so Hermes always wakes the agent to start the next experiment.
        out.append(f'IDLE at {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC. Analyse new results, then launch the next experiment.')
    STATE.write_text(json.dumps(state))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
