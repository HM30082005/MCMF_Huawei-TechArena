"""--tree-refit-policy (tree_refit_policy) picks its tree count from fitting days only."""
import argparse
import sys
import unittest
from unittest.mock import patch

import numpy as np

import history_upgrade_trees as trees


class RefitPolicyTests(unittest.TestCase):
    def setUp(self):
        self.start = 1737331200
        # Tree rows for days 2, 3 and 4: two events each, ten candidates per event.
        self.times = self.start + np.repeat([2, 3, 4], 2) * 86400 + 5
        self.data = {'time': self.times}
        self.meta = {'first_event': 0, 'last_event': 6}
        self.X = np.arange(60)[:, None].astype(float)
        self.y = np.arange(60) % 2

    def run_policy(self, best_trees, through=4):
        calls = []

        def fake_fit(kind, size, X, y, names, stage, args, valid=None, trees=None):
            calls.append(dict(rows=len(X), valid_rows=len(valid[0]), stage=stage, valid_max=valid[0].max()))
            return None, dict(trees=best_trees)

        with patch.object(trees, 'fit', fake_fit):
            count = trees.refit_tree_count('LightGBM', 31, self.X, self.y, ['f'], 'final',
                                           argparse.Namespace(), self.data, self.meta, through)
        return count, calls[0]

    def test_early_stops_on_the_last_fitting_day_trained_on_the_earlier_days(self):
        count, call = self.run_policy(100)
        self.assertEqual((call['rows'], call['valid_rows']), (40, 20))
        self.assertEqual(call['stage'], 'final_inner')
        self.assertEqual(call['valid_max'], 59)
        # Days 2-3 pick the count, days 2-4 are refit: 3/2 as many trees.
        self.assertEqual(count, 150)

    def test_missing_last_day_is_an_error_rather_than_an_empty_validation_set(self):
        with self.assertRaisesRegex(ValueError, 'last fitting day'):
            self.run_policy(100, through=5)


if __name__ == '__main__':
    unittest.main()
