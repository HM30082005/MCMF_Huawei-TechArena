"""Shared out-of-time evaluator and internal leaderboard for experiment runs.

Every run directory saves raw scores for fixed first-event-per-user windows:

  development_predictions.npz   event_indices, logits (N, 10): search window
  confirmation_predictions.npz  event_indices, logits (N, 10): reserved window
  evaluation_logits.npy + submission_*.csv: full-data submission (optional)

This evaluator owns the labels and the window definitions; runs never report their
own scores. Scores are appended to a store outside the run directories, and the
leaderboard is regenerated from that store. It does not import PyTorch.

  python research/evaluate.py score RUN_DIR --store STORE [--confirm]
  python research/evaluate.py leaderboard --store STORE [--copy-to PATH]
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'analysis' / '.cache'
START = 1737331200  # 2025-01-20 00:00 UTC
DAY = 86400
BASELINE = 'baseline-0.816323'
RESAMPLES = 300
# A run must clear these on development before it may spend a confirmation look.
DEV_MIN_GAIN = .001
CONFIRMATIONS_PER_DAY = 2


def load(prefix='train'):
    keys = ['user', 'time', 'hist_len'] + (['label'] if prefix == 'train' else [])
    return {k: np.load(CACHE / f'{prefix}_{k}.npy', mmap_mode='r') for k in keys}


def first_events(data, cutoff, end=None):
    ix = np.flatnonzero((data['time'] >= cutoff) & ((data['time'] < end) if end else True))
    _, first = np.unique(data['user'][ix], return_index=True)
    return np.sort(ix[first])


def windows(data):
    return {'development': first_events(data, START + 4*DAY, START + 6*DAY),
            'confirmation': first_events(data, START + 6*DAY)}


def window_identity(data):
    """Hash of window rows and labels; changes if the cache or definitions change."""
    digest = hashlib.sha256()
    for name, ix in windows(data).items():
        digest.update(name.encode())
        digest.update(ix.astype(np.int64).tobytes())
        digest.update(np.ascontiguousarray(data['label'][ix]).tobytes())
    return digest.hexdigest()


def auc(y, score):
    y = np.asarray(y).ravel().astype(bool)
    n = int(y.sum())
    return float((rankdata(np.asarray(score).ravel())[y].sum(dtype=np.float64) - n*(n+1)/2) / (n*(len(y)-n)))


def read_predictions(path, expected):
    with np.load(path) as saved:
        ix, logits = saved['event_indices'], np.asarray(saved['logits'], dtype=np.float64)
    if not np.array_equal(ix, expected):
        raise ValueError(f'{path.name}: event indices differ from the evaluation window')
    if logits.shape != (len(expected), 10) or not np.isfinite(logits).all():
        raise ValueError(f'{path.name}: expected finite logits of shape {(len(expected), 10)}')
    return logits


def paired(y, score, baseline, seed):
    """Event-level paired bootstrap of the AUC change against the baseline."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(RESAMPLES):
        ix = rng.integers(0, len(y), len(y))
        deltas.append(auc(y[ix], score[ix]) - auc(y[ix], baseline[ix]))
    return float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))


def window_report(data, name, ix, score, baseline):
    y = np.asarray(data['label'][ix])
    report = dict(events=len(ix), roc_auc=auc(y, score))
    if baseline is not None:
        report['baseline_auc'] = auc(y, baseline)
        report['delta'] = report['roc_auc'] - report['baseline_auc']
        report['delta_lower_95'], report['delta_upper_95'] = paired(y, score, baseline, 927)
    slices = {}
    day = (np.asarray(data['time'][ix]) - START) // DAY
    hist = np.asarray(data['hist_len'][ix])
    fit_counts = np.bincount(data['user'][:int(np.searchsorted(data['time'], data['time'][ix].min()))],
                             minlength=int(data['user'].max()) + 1)
    new_user = fit_counts[data['user'][ix]] == 0
    masks = [('new_user', new_user), ('empty_history', hist == 0),
             ('history_1_9', (hist > 0) & (hist < 10)), ('history_10_plus', hist >= 10)]
    if name == 'development':
        masks = [('jan24', day == 4), ('jan25', day == 5)] + masks
    for label, mask in masks:
        if mask.any() and 0 < y[mask].sum() < y[mask].size:
            item = dict(events=int(mask.sum()), roc_auc=auc(y[mask], score[mask]))
            if baseline is not None:
                item['delta'] = item['roc_auc'] - auc(y[mask], baseline[mask])
            slices[label] = item
    report['slices'] = slices
    return report


def records(store):
    path = store / 'scores.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def latest(store):
    result = {}
    for record in records(store):
        result.setdefault(record['run'], {})[record['window']] = record
    return result


def dev_candidate(dev):
    """Promotion rule for spending a confirmation look."""
    s = dev.get('slices', {})
    return (dev.get('delta', 0) >= DEV_MIN_GAIN and dev.get('delta_lower_95', -1) > 0
            and s.get('jan24', {}).get('delta', -1) > 0 and s.get('jan25', {}).get('delta', -1) > 0)


def baseline_scores(store, name):
    path = store / 'baseline' / f'{name}_predictions.npz'
    if not path.exists():
        return None
    with np.load(path) as saved:
        return np.asarray(saved['logits'], dtype=np.float64)


def score(args):
    run = args.run_dir.resolve()
    store = args.store.resolve()
    store.mkdir(parents=True, exist_ok=True)
    data = load()
    identity = window_identity(data)
    manifest = store / 'windows.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['identity'] != identity:
            raise SystemExit('Evaluation windows or labels changed; refusing to score')
    else:
        manifest.write_text(json.dumps({'identity': identity}) + '\n')
    meta = json.loads((run / 'run.json').read_text()) if (run / 'run.json').exists() else {}
    run_id = meta.get('id', run.name)
    names = ['development'] + (['confirmation'] if args.confirm else [])
    previous = latest(store).get(run_id, {})
    written = []
    try:
        score_windows(run, store, data, meta, run_id, names, previous, written)
    finally:
        leaderboard(argparse.Namespace(store=store, copy_to=args.copy_to))


def score_windows(run, store, data, meta, run_id, names, previous, written):
    for name in names:
        path = run / f'{name}_predictions.npz'
        if not path.exists():
            raise SystemExit(f'Missing {path}')
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if previous.get(name, {}).get('predictions_sha256') == sha:
            continue
        if name == 'confirmation' and run_id != BASELINE:
            dev = next((r for r in written if r['window'] == 'development'), None) or previous.get('development')
            if not dev or not dev_candidate(dev):
                raise SystemExit('Confirmation refused: development result does not meet the promotion rule')
            today = dt.datetime.now(dt.timezone.utc).date().isoformat()
            used = sum(1 for r in records(store) if r['window'] == 'confirmation'
                       and r['run'] != BASELINE and r['scored_at'].startswith(today))
            if used >= CONFIRMATIONS_PER_DAY:
                raise SystemExit(f'Confirmation refused: {CONFIRMATIONS_PER_DAY} confirmation looks already used today (UTC)')
        ix = windows(data)[name]
        logits = read_predictions(path, ix)
        baseline = None if run_id == BASELINE else baseline_scores(store, name)
        record = dict(run=run_id, window=name, scored_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                      predictions_sha256=sha, hypothesis=meta.get('hypothesis', ''), parent=meta.get('parent', BASELINE),
                      **window_report(data, name, ix, logits, baseline))
        submission = sorted(run.glob('submission*.csv'))
        if submission:
            record['submission'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in submission}
        with (store / 'scores.jsonl').open('a') as out:
            out.write(json.dumps(record) + '\n')
        written.append(record)
        print(json.dumps({k: record[k] for k in record if k != 'slices'}))


def fmt(value, digits=6, signed=False):
    if value is None:
        return '–'
    return f'{value:+.{digits}f}' if signed else f'{value:.{digits}f}'


def leaderboard(args):
    store = args.store.resolve()
    runs = latest(store)
    rows = []
    for run_id, w in runs.items():
        dev, conf = w.get('development', {}), w.get('confirmation')
        if run_id == BASELINE:
            status = 'baseline (official 0.816323)'
        elif conf is not None:
            status = 'SUBMIT CANDIDATE' if conf.get('delta', 0) > 0 and dev_candidate(dev) else 'confirmed: no gain'
        else:
            status = 'eligible for confirmation' if dev_candidate(dev) else 'development only'
        rows.append(dict(run=run_id, status=status, dev=dev, conf=conf,
                         hypothesis=(dev or conf or {}).get('hypothesis', ''),
                         submission=(conf or dev or {}).get('submission')))
    rows.sort(key=lambda r: (r['conf'] is None, -(r['conf'] or r['dev'] or {}).get('roc_auc', 0)))
    lines = ['# Internal leaderboard', '',
             f'Generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC} by `research/evaluate.py`. '
             'Do not edit by hand. Development is the search window (first event per user, Jan 24–25); '
             'confirmation (first event per user, Jan 26) is reserved for development winners. '
             'Deltas are against the 0.816323 baseline on identical rows, with 95% paired bootstrap intervals.', '',
             '| # | Run | Status | Confirmation AUC | Δ conf | Development AUC | Δ dev [95% CI] | Δ Jan 24 / Jan 25 | Hypothesis |',
             '|---:|---|---|---:|---:|---:|---|---|---|']
    for i, r in enumerate(rows, 1):
        d, c = r['dev'] or {}, r['conf'] or {}
        s = d.get('slices', {})
        ci = (f"{fmt(d.get('delta'), signed=True)} [{fmt(d.get('delta_lower_95'), signed=True)}, "
              f"{fmt(d.get('delta_upper_95'), signed=True)}]") if 'delta' in d else '–'
        days = (f"{fmt(s.get('jan24', {}).get('delta'), signed=True)} / {fmt(s.get('jan25', {}).get('delta'), signed=True)}"
                if 'delta' in d else '–')
        lines.append(f"| {i} | `{r['run']}` | {r['status']} | {fmt(c.get('roc_auc'))} | {fmt(c.get('delta'), signed=True)} | "
                     f"{fmt(d.get('roc_auc'))} | {ci} | {days} | {r['hypothesis'][:120]} |")
    picks = [r for r in rows if r['status'] == 'SUBMIT CANDIDATE']
    lines += ['', '## Submission candidates', '']
    if not picks:
        lines.append('None yet. A candidate needs a development gain of at least '
                     f'+{DEV_MIN_GAIN} (interval above zero, positive on both days) and a positive confirmation gain.')
    for r in picks:
        files = r['submission'] or {}
        detail = ', '.join(f'`{k}` sha256 `{v}`' for k, v in files.items()) or 'full-data CSV not built yet'
        lines.append(f"- `{r['run']}`: confirmation {fmt(r['conf']['roc_auc'])} "
                     f"({fmt(r['conf']['delta'], signed=True)}); {detail}")
    text = '\n'.join(lines) + '\n'
    (store / 'LEADERBOARD.md').write_text(text)
    if args.copy_to:
        Path(args.copy_to).write_text(text)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('score')
    s.add_argument('run_dir', type=Path)
    s.add_argument('--store', type=Path, required=True)
    s.add_argument('--confirm', action='store_true', help='Also score the reserved confirmation window')
    s.add_argument('--copy-to', type=Path)
    b = sub.add_parser('leaderboard')
    b.add_argument('--store', type=Path, required=True)
    b.add_argument('--copy-to', type=Path)
    args = parser.parse_args()
    if 'torch' in sys.modules:
        raise RuntimeError('Evaluator must not import PyTorch')
    score(args) if args.command == 'score' else leaderboard(args)


if __name__ == '__main__':
    main()
