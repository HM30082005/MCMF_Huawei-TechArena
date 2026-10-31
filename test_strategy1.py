"""Checks for temporal boundaries, masking, target leakage, and model invariants."""
import unittest

import numpy as np
import torch

from strategy1 import HistoryFactorization, chronological_split, prepare_tensors, roc_auc, utc_stamp


class StrategyOneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.data = {
            'user': np.array([0, 1, 2]),
            'items': np.tile(np.arange(10), (3, 1)),
            'hist': np.array([[0, 2] + [-1] * 18, [1] + [-1] * 19, [-1] * 20]),
            'hist_time': np.array([[800, 900] + [0] * 18, [700] + [0] * 19, [0] * 20]),
            'hist_len': np.array([2, 1, 0]),
            'time': np.array([1000, 1000, 1000]),
            'bucket': np.array([0, 1, 2]), 'weekend': np.array([0, 0, 1]),
            'label': np.zeros((3, 10)),
        }
        self.batch = prepare_tensors(self.data, 'cpu')
        self.model = HistoryFactorization(3, 10, 4, [3, 3, 0], np.ones(10), .06)
        self.model.eval()

    def test_auc_concordance_and_ties(self):
        self.assertAlmostEqual(roc_auc([1, 1, 0, 0], [.9, .4, .6, .2]), .75)
        self.assertEqual(roc_auc([1, 0], [.5, .5]), .5)

    def test_first_post_cutoff_events_and_disjoint_windows(self):
        t = utc_stamp('2025-01-24')
        data = {'user': np.array([0, 0, 1, 0, 1, 2]),
                'time': np.array([t-1, t, t, t+60, t+86400, t+2*86400])}
        fit, valid = chronological_split(data, '2025-01-24', '2025-01-26')
        np.testing.assert_array_equal(fit, [0])
        np.testing.assert_array_equal(valid, [1, 2])

    def test_labels_cannot_change_forward_predictions(self):
        original = self.model(self.batch).detach()
        changed = dict(self.batch, label=1-self.batch['label'])
        torch.testing.assert_close(original, self.model(changed))

    def test_unknown_user_parameters_are_ignored(self):
        original = self.model(self.batch).detach()
        with torch.no_grad():
            self.model.user.weight[3].fill_(100)
            self.model.user_bias.weight[3].fill_(100)
        torch.testing.assert_close(original, self.model(self.batch))

    def test_candidates_are_permutation_equivariant(self):
        perm = torch.tensor([2, 5, 1, 9, 4, 0, 3, 7, 8, 6])
        original = self.model(self.batch).detach()
        changed = dict(self.batch, items=self.batch['items'][:, perm], repeat=self.batch['repeat'][:, perm])
        torch.testing.assert_close(original[:, perm], self.model(changed))

    def test_empty_history_and_training_gradients(self):
        self.assertTrue(torch.isfinite(self.model(self.batch)).all())
        self.assertEqual(int(self.batch['hist'][2].sum()), 0)
        logits, reg = self.model(self.batch, regularize=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, self.batch['label']) + reg
        loss.backward()
        self.assertGreater(float(self.model.item.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.history.weight.grad.abs().sum()), 0)
        torch.testing.assert_close(self.model.history.weight.grad[0], torch.zeros(4))


if __name__ == '__main__':
    unittest.main()
