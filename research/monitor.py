"""Per-tick lab monitor for the Hermes research job (runs before each agent turn).

Scores finished runs with the frozen evaluator, honours confirmation requests,
checks the integrity of the baseline and harness, sends Telegram alerts, and
prints a status block. Hermes skips the agent turn when this output is unchanged,
so the output is stable while a job runs and changes when there is work to do.

Telegram gets one message per scored, failed or confirmed run, a daily digest,
and alerts for new leaders, submission candidates, stalls and integrity failures.

Configuration: ~/.hermes/techarena/config.json (written by research/install_lab.sh).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

CONFIG = json.loads((Path.home() / '.hermes' / 'techarena' / 'config.json').read_text())
LAB, STORE, HARNESS = Path(CONFIG['lab']), Path(CONFIG['store']), Path(CONFIG['harness'])
RUNS = LAB / 'runs'
PY = CONFIG['python']
STATE = STORE / 'monitor_state.json'
DIGEST_ZONE = ZoneInfo(CONFIG.get('digest_timezone', 'Europe/Dublin'))
DIGEST_HOUR = CONFIG.get('digest_hour', 8)
sys.path.insert(0, str(HARNESS / 'research'))
import evaluate  # noqa: E402  (the frozen copy: same promotion rule as the scores)


def gain(record):
    if 'delta' not in record:
        return f"AUC {record['roc_auc']:.6f}"
    s = record.get('slices', {})
    days = ''
    if 'jan24' in s and 'jan25' in s:
        days = f"; Jan 24 {s['jan24'].get('delta', 0):+.4f}, Jan 25 {s['jan25'].get('delta', 0):+.4f}"
    return (f"AUC {record['roc_auc']:.6f} ({record['delta']:+.4f} vs baseline, "
            f"95% CI {record['delta_lower_95']:+.4f} to {record['delta_upper_95']:+.4f}{days})")


def run_message(run_id, window):
    record = evaluate.latest(STORE).get(run_id, {}).get(window)
    if not record:
        return None
    if window == 'development':
        verdict = ('eligible for Jan 26 confirmation' if evaluate.dev_candidate(record)
                   else 'below the promotion bar')
        return (f"TechArena run {run_id}\nDevelopment {gain(record)}\n→ {verdict}\n"
                f"Hypothesis: {record.get('hypothesis', '')[:200]}")
    verdict = 'SUBMIT CANDIDATE' if record.get('delta', 0) > 0 else 'no gain on Jan 26, not a candidate'
    return f"TechArena confirmation {run_id}\nJan 26 {gain(record)}\n→ {verdict}"


def digest(state, job):
    """Once a day after DIGEST_HOUR local time: runs, best result, top of the leaderboard."""
    now = dt.datetime.now(DIGEST_ZONE)
    today = now.date().isoformat()
    if now.hour < DIGEST_HOUR or state.get('digest_date') == today:
        return
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    recent = [r for r in evaluate.records(STORE) if r['run'] != evaluate.BASELINE
              and dt.datetime.fromisoformat(r['scored_at']) >= since]
    dev = [r for r in recent if r['window'] == 'development']
    conf = [r for r in recent if r['window'] == 'confirmation']
    failed = [p.parent.name for p in RUNS.glob('*/SCORED')
              if p.read_text().strip() == 'failed' and dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc) >= since]
    lines = [f'TechArena daily digest ({today})',
             f'Last 24 h: {len(dev)} runs scored, {len(conf)} confirmation looks, {len(failed)} failed.']
    if dev:
        best = max(dev, key=lambda r: r.get('delta', -1))
        lines.append(f"Best development result: {best['run']}, {gain(best)}")
    rows = evaluate.leaderboard(argparse.Namespace(store=STORE, copy_to=LAB / 'LEADERBOARD.md'))
    lines.append('Leaderboard top 3:')
    for i, row in enumerate(rows[:3], 1):
        record = row['conf'] or row['dev'] or {}
        window = 'Jan 26' if row['conf'] else 'dev'
        lines.append(f"{i}. {row['run']}: {window} AUC {record.get('roc_auc', 0):.6f}, {row['status']}")
    lines.append(f'Now running: {job[0]}' if job else 'Now: idle')
    notify(state, 'digest:' + today, '\n'.join(lines))
    if state['sent'].get('digest:' + today):
        state['digest_date'] = today


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
    baseline = Path(CONFIG['baseline_dir'])
    entries = [baseline, *baseline.rglob('*')]
    writable = sum(os.access(p, os.W_OK) for p in entries)
    if writable:
        problems.append(f'{writable} baseline artifact entries are writable again')
    current = {str(p.relative_to(baseline)): [p.stat().st_size, p.stat().st_mtime_ns]
               for p in entries if p.is_file()}
    expected = CONFIG['baseline_manifest']
    changed = sorted(k for k in set(current) | set(expected) if current.get(k) != expected.get(k))
    if changed:
        problems.append(f'{len(changed)} baseline files changed, added or removed, e.g. ' + changed[0])
    for rel, digest in CONFIG['lab_manifest'].items():
        path = LAB / rel
        if not path.exists() or sha256(path) != digest or os.access(path, os.W_OK):
            problems.append('protected lab file changed or unlocked: ' + rel)
    for rel, digest in CONFIG['harness_manifest'].items():
        path = HARNESS / rel
        if not path.exists() or sha256(path) != digest:
            problems.append('frozen harness file changed: ' + rel)
    return problems


def run_evaluator(*args):
    result = subprocess.run([PY, str(HARNESS / 'research' / 'evaluate.py'), *args, '--store', str(STORE),
                             '--copy-to', str(LAB / 'LEADERBOARD.md')],
                            capture_output=True, text=True, timeout=1800)
    return result.returncode, (result.stdout + result.stderr).strip().splitlines()[-1:] or ['']


def recorded_id(run):
    meta = run / 'run.json'
    return json.loads(meta.read_text()).get('id', run.name) if meta.exists() else run.name


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
        out.append('Stop all work. Do not launch or score anything, and do not try to repair this: '
                   'no chmod, no test writes, no edits in protected folders. The job has been paused for the user.')
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
                notify(state, 'failed:' + run.name + code, f'TechArena run {run.name} failed (exit {code}).')
                scored.write_text('failed\n')
                continue
            if (run / 'development_predictions.npz').exists():
                rc, tail = run_evaluator('score', str(run))
                events.append(f'SCORED {run.name} (development): ' + (tail[0] if rc == 0 else 'ERROR ' + tail[0]))
                message = run_message(recorded_id(run), 'development') if rc == 0 else None
                if message:
                    notify(state, 'scored:' + run.name + message, message)
            else:
                events.append(f'FINISHED {run.name}: no development_predictions.npz to score')
            scored.write_text('ok\n')
        request = run / 'REQUEST_CONFIRMATION'
        if request.exists():
            rc, tail = run_evaluator('score', str(run), '--confirm')
            events.append(f'CONFIRMATION {run.name}: ' + (tail[0] if rc == 0 else 'REFUSED ' + tail[0]))
            message = (run_message(recorded_id(run), 'confirmation') if rc == 0
                       else f'TechArena confirmation for {run.name} refused: {tail[0]}')
            if message:
                notify(state, 'confirmation:' + run.name + message, message)
            request.unlink()

    board = LAB / 'LEADERBOARD.md'
    if not board.exists() and (STORE / 'scores.jsonl').exists():
        run_evaluator('leaderboard')
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
        out.append(f'RUNNING {run_id} (started {started}, about {int(hours)} h elapsed). End your turn now; do not wait or poll. You will be woken when it finishes.')
        if hours >= 3:
            notify(state, 'stalled:' + run_id, f'TechArena: run {run_id} has been running for {int(hours)} h; check it.')
    else:
        # Idle: include the tick time so Hermes always wakes the agent to start the next experiment.
        out.append(f'IDLE at {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC. Analyse new results, then launch the next experiment.')
    try:
        digest(state, job)
    except Exception as error:  # The digest is a convenience; it must never block the status.
        out.append(f'(daily digest failed: {error})')
    STATE.write_text(json.dumps(state))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
