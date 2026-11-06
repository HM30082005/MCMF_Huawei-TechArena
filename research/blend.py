"""Average finished runs into one new run, with equal weights on the logit scale.

  python research/blend.py RUN_ID "one-sentence hypothesis" COMPONENT COMPONENT [...]

Choose components on development rows only. Equal weights are deliberate: tuning weights on
a window fits its noise. The new run holds the files the shared evaluator scores, so it joins
the leaderboard like any other, and it needs no GPU job. Each component must have finished its
final stage, so the blend can be scored on every window and submitted.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
WINDOWS = ('development', 'confirmation')


def blend_window(folders, window):
    """Mean logits of one window; every component must hold exactly the same rows."""
    indices, total = None, None
    for folder in folders:
        with np.load(folder / f'{window}_predictions.npz') as saved:
            ix, logits = saved['event_indices'], np.asarray(saved['logits'], dtype=np.float64)
        if not np.isfinite(logits).all():
            raise ValueError(f'{folder.name}: non-finite {window} scores')
        if indices is None:
            indices, total = ix, logits.copy()
        elif not np.array_equal(ix, indices) or logits.shape != total.shape:
            raise ValueError(f'{folder.name}: {window} rows differ from the other components')
        else:
            total += logits
    return indices, total / len(folders)


def blend_evaluation(folders):
    scores = [np.load(folder / 'evaluation_logits.npy').astype(np.float64) for folder in folders]
    if any(s.shape != scores[0].shape or not np.isfinite(s).all() for s in scores):
        raise ValueError('Evaluation scores differ in shape or contain non-finite values')
    return sum(scores) / len(scores)


def check_components(names):
    if len(set(names)) < 2 or len(set(names)) != len(names):
        raise SystemExit('A blend needs at least two different components.')
    folders = [RUNS / name for name in names]
    needed = [f'{w}_predictions.npz' for w in WINDOWS] + ['evaluation_logits.npy']
    missing = [f'{f.name}/{n}' for f in folders for n in needed if not (f / n).exists()]
    if missing:
        raise SystemExit('Finish these components first (--until final):\n  ' + '\n  '.join(missing))
    return folders


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_id')
    parser.add_argument('hypothesis')
    parser.add_argument('components', nargs='+')
    args = parser.parse_args()
    run = RUNS / args.run_id
    if run.exists():
        raise SystemExit(f'{run.relative_to(ROOT)} already exists. Choose a new run ID.')
    folders = check_components(args.components)
    # Computed before the folder exists, so a failure leaves nothing for the monitor to score.
    windows = {w: blend_window(folders, w) for w in WINDOWS}
    evaluation = blend_evaluation(folders)
    sys.path.insert(0, str(ROOT / 'src'))
    import strategy1  # Imported late: it loads PyTorch, and the checks above do not need it.
    run.mkdir(parents=True)
    for window, (indices, logits) in windows.items():
        np.savez(run / f'{window}_predictions.npz', event_indices=indices, logits=logits)
    np.save(run / 'evaluation_logits.npy', evaluation)
    submission = strategy1.write_submission(evaluation, filename=str(run / 'submission_blend.csv'))
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (run / 'run.json').write_text(json.dumps(dict(
        id=args.run_id, hypothesis=args.hypothesis, parent=args.components[0], git_commit=commit,
        blend_of=args.components, weights='equal, on the logit scale', submission=submission), indent=2) + '\n')
    (run / 'EXIT').write_text('0\n')  # Last, so the monitor only ever sees a complete blend.
    print(f"Blended {len(folders)} runs into {run.relative_to(ROOT)}; submission sha256 {submission['sha256']}")


if __name__ == '__main__':
    main()
