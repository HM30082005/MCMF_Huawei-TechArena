"""Reproduce the 0.816323 baseline from its saved trees and export it as a run.

Reloads the saved LightGBM checkpoints, re-predicts the development, confirmation
and evaluation features, checks the results against the saved scores, rebuilds the
submission CSV, and checks its SHA-256 against the official record. Nothing is
retrained and the baseline directory is only read.

  python research/reproduce_baseline.py --runs-dir RUNS --store STORE
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research'))
import evaluate  # noqa: E402

SOURCE = ROOT / 'artifacts' / 'history_upgrade'
RUN_ID = evaluate.BASELINE
OFFICIAL_SHA = json.loads((ROOT / 'results' / 'official_scores.json').read_text())['scores'][0]['submission_sha256']


def ensure_openmp():
    """LightGBM uses the OpenMP runtime bundled with the PyTorch wheel; find it without importing torch."""
    spec = importlib.util.find_spec('torch')
    runtime = str(Path(spec.origin).parent / 'lib') if spec else ''
    if runtime and runtime not in os.environ.get('DYLD_LIBRARY_PATH', ''):
        env = dict(os.environ, DYLD_LIBRARY_PATH=runtime + os.pathsep + os.environ.get('DYLD_LIBRARY_PATH', ''))
        os.execve(sys.executable, [sys.executable] + sys.argv, env)


def predict(tree, features):
    import lightgbm as lgb
    x = np.load(SOURCE / 'features' / features, mmap_mode='r')
    model = lgb.Booster(model_file=str(SOURCE / 'trees' / tree))
    return model.predict(x.reshape(-1, x.shape[-1]), raw_score=True, num_threads=8).reshape(-1, 10)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs-dir', type=Path, required=True)
    parser.add_argument('--store', type=Path, required=True)
    args = parser.parse_args()
    ensure_openmp()
    data = evaluate.load()
    windows = evaluate.windows(data)
    run = args.runs_dir.resolve() / RUN_ID
    run.mkdir(parents=True, exist_ok=True)

    dev = predict('development_LightGBM_31.txt', 'development.npy')
    recorded = json.loads((SOURCE / 'trees' / 'development_LightGBM_31.json').read_text())['development']['roc_auc']
    dev_auc = evaluate.auc(data['label'][windows['development']], dev)
    assert abs(dev_auc - recorded) < 1e-12, (dev_auc, recorded)
    np.savez(run / 'development_predictions.npz', event_indices=windows['development'], logits=dev)

    conf = predict('confirmation_LightGBM_31.txt', 'confirmation.npy')
    with np.load(SOURCE / 'confirmation_LightGBM.npz') as saved:
        np.testing.assert_array_equal(saved['event_indices'], windows['confirmation'])
        np.testing.assert_allclose(conf, saved['logits'], rtol=0, atol=1e-12)
    np.savez(run / 'confirmation_predictions.npz', event_indices=windows['confirmation'], logits=conf)

    final = predict('final_LightGBM_31.txt', 'evaluation.npy')
    np.testing.assert_allclose(final, np.load(SOURCE / 'evaluation_LightGBM_logits.npy'), rtol=0, atol=1e-12)
    np.save(run / 'evaluation_logits.npy', final)

    # The CSV writer imports PyTorch, so it runs in its own process.
    csv = run / 'submission_LightGBM.csv'
    code = ('import sys,numpy as np; sys.path.insert(0,sys.argv[1]); import strategy1;'
            'print(strategy1.write_submission(np.load(sys.argv[2]),filename=sys.argv[3])["sha256"])')
    env = {k: v for k, v in os.environ.items() if k != 'DYLD_LIBRARY_PATH'}
    sha = subprocess.run([sys.executable, '-c', code, str(ROOT / 'src'), str(run / 'evaluation_logits.npy'), str(csv)],
                         check=True, capture_output=True, text=True, env=env).stdout.split()[-1]
    if sha != OFFICIAL_SHA:
        raise SystemExit(f'Rebuilt CSV SHA-256 {sha} does not match the official record {OFFICIAL_SHA}')

    (run / 'run.json').write_text(json.dumps(dict(
        id=RUN_ID, parent=None, status='completed',
        hypothesis='Baseline: ten-seed shared-item history MF + LightGBM (31 leaves); official AUC 0.816323',
        source=str(SOURCE), reproduced=dict(development_auc=dev_auc, confirmation_logits='exact',
                                            evaluation_logits='exact', submission_sha256=sha)), indent=2) + '\n')
    store = args.store.resolve()
    (store / 'baseline').mkdir(parents=True, exist_ok=True)
    for name in ['development', 'confirmation']:
        shutil.copy2(run / f'{name}_predictions.npz', store / 'baseline' / f'{name}_predictions.npz')
    print(json.dumps(dict(run=str(run), development_auc=dev_auc, submission_sha256=sha, matches_official=True)))


if __name__ == '__main__':
    main()
