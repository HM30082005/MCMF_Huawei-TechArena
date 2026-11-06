import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
import evaluate  # noqa: E402


def synthetic():
    """Eight days of events; 40 users join each day, so both development days have first events."""
    rng = np.random.default_rng(0)
    users = np.concatenate([np.arange(d * 40, d * 40 + 300) for d in range(8)])
    days = np.repeat(np.arange(8), 300)
    time = evaluate.START + days * evaluate.DAY + rng.integers(0, evaluate.DAY, len(users))
    order = np.argsort(time, kind='stable')
    label = np.zeros((len(users), 10), np.int8)
    label[rng.random(len(users)) < .6, 0] = 1
    return {'user': users[order], 'time': time[order], 'label': label[order],
            'hist_len': rng.integers(0, 21, len(users))[order]}


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = self.root / 'store'
        self.patch = patch.object(evaluate, 'load', lambda prefix='train': self.data)
        self.patch.start()
        self.windows = evaluate.windows(self.data)
        noise = np.random.default_rng(1)
        self.base = {n: self.data['label'][ix] + noise.normal(0, 1.5, (len(ix), 10)) for n, ix in self.windows.items()}
        self.save('baseline-0.816323', self.base)
        self.score('baseline-0.816323', confirm=True)
        (self.store / 'baseline').mkdir(exist_ok=True)
        for n in self.windows:
            np.savez(self.store / 'baseline' / f'{n}_predictions.npz', event_indices=self.windows[n], logits=self.base[n])

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def save(self, run, logits, windows=None):
        directory = self.root / 'runs' / run
        directory.mkdir(parents=True, exist_ok=True)
        for n, ix in (windows or self.windows).items():
            np.savez(directory / f'{n}_predictions.npz', event_indices=ix, logits=logits[n])
        (directory / 'run.json').write_text(json.dumps({'id': run, 'hypothesis': 'test'}))
        return directory

    def score(self, run, confirm=False):
        evaluate.score(argparse.Namespace(run_dir=self.root / 'runs' / run, store=self.store,
                                          confirm=confirm, copy_to=None))

    def better(self, run):
        # Mixing in more label signal improves AUC on every day.
        self.save(run, {n: self.base[n] + .8 * self.data['label'][ix] for n, ix in self.windows.items()})

    def test_worse_run_is_refused_confirmation_but_listed(self):
        self.save('worse', {n: v + np.random.default_rng(2).normal(0, 3, v.shape) for n, v in self.base.items()})
        with self.assertRaisesRegex(SystemExit, 'promotion rule'):
            self.score('worse', confirm=True)
        runs = evaluate.latest(self.store)
        self.assertIn('development', runs['worse'])
        self.assertNotIn('confirmation', runs['worse'])
        self.assertIn('`worse`', (self.store / 'LEADERBOARD.md').read_text())

    def test_better_run_becomes_submission_candidate(self):
        self.better('better')
        self.score('better', confirm=True)
        rows = evaluate.leaderboard(argparse.Namespace(store=self.store, copy_to=None))
        self.assertEqual(rows[0]['run'], 'better')
        self.assertEqual(rows[0]['status'], 'SUBMIT CANDIDATE')

    def test_daily_confirmation_budget(self):
        for i in range(evaluate.CONFIRMATIONS_PER_DAY):
            self.better(f'better{i}')
            self.score(f'better{i}', confirm=True)
        self.better('one-too-many')
        with self.assertRaisesRegex(SystemExit, 'already used today'):
            self.score('one-too-many', confirm=True)

    def test_rows_must_match_the_window(self):
        shifted = {n: ix[1:] for n, ix in self.windows.items()}
        self.save('shifted', {n: self.base[n][1:] for n in self.windows}, windows=shifted)
        with self.assertRaisesRegex(ValueError, 'event indices differ'):
            self.score('shifted')

    def test_changed_labels_are_rejected(self):
        self.data['label'] = self.data['label'].copy()
        self.data['label'][self.windows['development'][0], 0] ^= 1
        self.better('after-change')
        with self.assertRaisesRegex(SystemExit, 'windows or labels changed'):
            self.score('after-change')


if __name__ == '__main__':
    unittest.main()
