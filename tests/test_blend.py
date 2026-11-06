"""Blends must be plain equal-weight means, on identical rows, of complete components."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
import blend  # noqa: E402


class BlendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runs = Path(self.tmp.name)
        self.ix = np.arange(4)
        self.a = self.make('a', 1.0)
        self.b = self.make('b', 3.0)
        patcher = patch.object(blend, 'RUNS', self.runs)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make(self, name, value, ix=None):
        folder = self.runs / name
        folder.mkdir()
        rows = self.ix if ix is None else ix
        for window in blend.WINDOWS:
            np.savez(folder / f'{window}_predictions.npz', event_indices=rows, logits=np.full((len(rows), 10), value))
        np.save(folder / 'evaluation_logits.npy', np.full((6, 10), value))
        return folder

    def test_windows_are_equal_weight_means_of_the_logits(self):
        ix, logits = blend.blend_window([self.a, self.b], 'development')
        np.testing.assert_array_equal(ix, self.ix)
        np.testing.assert_array_equal(logits, np.full((4, 10), 2.0))
        np.testing.assert_array_equal(blend.blend_evaluation([self.a, self.b]), np.full((6, 10), 2.0))

    def test_components_with_different_rows_are_rejected(self):
        other = self.make('other', 5.0, ix=np.arange(1, 5))
        with self.assertRaisesRegex(ValueError, 'rows differ'):
            blend.blend_window([self.a, other], 'confirmation')

    def test_non_finite_scores_are_rejected(self):
        np.savez(self.b / 'development_predictions.npz', event_indices=self.ix,
                 logits=np.full((4, 10), np.nan))
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            blend.blend_window([self.a, self.b], 'development')

    def test_unfinished_components_are_named(self):
        (self.b / 'evaluation_logits.npy').unlink()
        with self.assertRaisesRegex(SystemExit, 'b/evaluation_logits.npy'):
            blend.check_components(['a', 'b'])

    def test_a_blend_needs_two_different_components(self):
        for names in (['a'], ['a', 'a']):
            with self.assertRaisesRegex(SystemExit, 'two different'):
                blend.check_components(names)


if __name__ == '__main__':
    unittest.main()
