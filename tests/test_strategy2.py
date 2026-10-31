"""Temporal leakage and feature invariants for the boosted-tree pipeline."""
import argparse
import unittest

from catboost import CatBoostClassifier
import numpy as np

from strategy2 import PastStatistics, engineered_block, make_pool, rank_within_event, tree_params


class StrategyTwoTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            'user': np.array([0, 1, 0, 2], np.int32),
            'items': np.tile(np.arange(10, dtype=np.int32), (4, 1)),
            'hist': np.array([[0, 2] + [-1]*18, [1] + [-1]*19,
                              [0, 2] + [-1]*18, [-1]*20], np.int32),
            'hist_time': np.array([[50, 80] + [0]*18, [60] + [0]*19,
                                   [50, 80] + [0]*18, [0]*20]),
            'hist_len': np.array([2, 1, 2, 0], np.int8),
            'time': np.array([100, 150, 220, 250]),
            'bucket': np.array([0, 1, 2, 3], np.int8),
            'weekend': np.zeros(4, np.int8),
            'label': np.array([[1] + [0]*9, [0, 1] + [0]*8,
                                [0, 0, 1] + [0]*7, [0]*10], np.int8),
        }
        self.sizes = {'all_users': 3, 'all_items': 10}
        self.fit, self.query = np.array([0, 1]), np.array([2, 3])
        self.stats = PastStatistics(self.data, self.fit, self.sizes, 200)
        self.stats.build_similarity(self.data, self.fit, self.sizes)
        self.vectors = np.random.default_rng(3).normal(size=(10, 4)).astype(np.float32)
        self.scores = np.tile(np.arange(10, dtype=np.float32), (2, 1))

    def features(self, data=None, stats=None, scores=None):
        return engineered_block(data or self.data, self.query, stats or self.stats,
                    self.vectors, (self.vectors, self.vectors),
                    self.scores if scores is None else scores)

    def test_current_and_future_labels_cannot_change_features(self):
        original, names = self.features()
        changed = dict(self.data, label=self.data['label'].copy())
        changed['label'][self.query] = 1 - changed['label'][self.query]
        stats = PastStatistics(changed, self.fit, self.sizes, 200)
        stats.build_similarity(changed, self.fit, self.sizes)
        actual, actual_names = self.features(changed, stats)
        np.testing.assert_array_equal(original, actual)
        self.assertEqual(names, actual_names)
        without_labels = dict(self.data)
        del without_labels['label']
        np.testing.assert_array_equal(original, self.features(without_labels)[0])

    def test_temporal_guard_rejects_query_outcomes(self):
        with self.assertRaises(ValueError):
            PastStatistics(self.data, np.array([0, 2]), self.sizes, 200)

    def test_candidate_permutation_and_tie_ranks(self):
        perm = np.array([4, 2, 7, 3, 1, 0, 5, 9, 8, 6])
        original, names = self.features()
        changed = dict(self.data, items=self.data['items'][:, perm])
        actual, _ = self.features(changed, scores=self.scores[:, perm])
        np.testing.assert_allclose(original[:, perm], actual, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(rank_within_event(np.ones((2, 10))), .5)

    def test_no_past_events_and_empty_history_are_finite(self):
        stats = PastStatistics(self.data, np.array([], dtype=int), self.sizes, 200)
        stats.build_similarity(self.data, np.array([], dtype=int), self.sizes)
        actual, names = self.features(stats=stats)
        self.assertTrue(np.isfinite(actual).all())
        self.assertEqual(actual[1, 0, names.index('history_length')], 0)
        self.assertEqual(actual[1, 0, names.index('last_click_age_log_hours')], -1)

    def test_catboost_quantized_pool_and_target_free_counters(self):
        features, names = self.features()
        pool = make_pool(features, self.data, self.query, names, threads=2)
        params = tree_params(2, 2, argparse.Namespace(threads=2))
        params.update(allow_writing_files=False, verbose=False)
        del params['train_dir']
        model = CatBoostClassifier(**params).fit(pool)
        actual = model.predict(pool, prediction_type='RawFormulaVal')
        self.assertEqual(actual.shape, (20,))
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(all(x.startswith('Counter:') for x in model.get_all_params()['simple_ctr']))


if __name__ == '__main__':
    unittest.main()
