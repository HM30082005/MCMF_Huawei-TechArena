"""Each refusal here matches a launch mistake that once wasted a run."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import history_upgrade as pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
import preflight  # noqa: E402

MF = ['--steps', '3000', '--init-std', '0.005', '--seeds', '10']


def command(run, *flags):
    return ['.venv/bin/python', 'src/history_upgrade.py', '--output-dir', f'runs/{run}', *MF, *flags]


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / 'src').mkdir()
        (root / 'tests').mkdir()
        (root / 'src/history_upgrade.py').touch()
        (root / 'tests/test_x.py').write_text('query_clicks tree_refit_policy')
        self.runs = root / 'runs'
        self.runs.mkdir()
        self.parent = self.make_run('parent-run', ' '.join(command('parent-run')))
        patches = [patch.object(preflight, 'ROOT', root), patch.object(preflight, 'RUNS', self.runs),
                   patch.object(preflight, 'code_is_committed', lambda: None),
                   patch.object(preflight, 'tests_pass', lambda: None)]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(self.tmp.cleanup)

    def make_run(self, name, line):
        folder = self.runs / name
        folder.mkdir()
        (folder / 'development_predictions.npz').touch()
        (folder / 'commands.txt').write_text(line + '\n')
        return folder

    def check(self, run, cmd, parent='parent-run', clone_from=None, clone_mf=False, resume=False, combine=False):
        opts = argparse.Namespace(run_id=run, command=['--', *cmd], parent=parent, clone_from=clone_from,
                                  clone_mf=clone_mf, resume=resume, combine=combine)
        return preflight.check(opts, pipeline.build_parser(), pipeline.recipe)

    def refused(self, pattern, *args, **kwargs):
        with self.assertRaisesRegex(preflight.Refused, pattern):
            self.check(*args, **kwargs)
        self.assertFalse((self.runs / args[0]).exists() and args[0] != 'parent-run' and not kwargs.get('resume'))

    def test_one_tested_change_passes_and_is_reported(self):
        lines = self.check('new-run', command('new-run', '--query-clicks'))
        self.assertIn('query_clicks: False -> True', lines[0])

    def test_launcher_option_after_the_separator_is_refused(self):
        # stale-context-p12 and t4-refit-durham-retry-fixed died this way.
        self.refused('go before the run ID', 'new-run', command('new-run', '--clone-from', 'parent-run'))
        self.refused('unrecognized', 'new-run', command('new-run', '--candidates', 'LightGBM:31'))

    def test_missing_output_dir_cannot_reach_the_baseline_folder(self):
        # t3-asymmetric-steps overwrote the baseline's recipe.json this way.
        self.refused('--output-dir runs/new-run', 'new-run', ['.venv/bin/python', 'src/history_upgrade.py', *MF])
        self.refused('--output-dir runs/new-run', 'new-run', command('other-run'))

    def test_code_from_another_checkout_is_refused(self):
        self.refused("this checkout's code", 'new-run', ['python', '/elsewhere/src/history_upgrade.py'])

    def test_a_parent_is_required(self):
        self.refused('--parent RUN', 'new-run', command('new-run'), parent=None)

    def test_several_changes_from_the_parent_are_refused_unless_combining(self):
        # t4-refit-fixed tested a tree policy with a different MF recipe from its parent.
        cmd = ['.venv/bin/python', 'src/history_upgrade.py', '--output-dir', 'runs/new-run', '--steps', '5000',
               '--init-std', '0.05', '--seeds', '10', '--tree-refit-policy']
        self.refused('changes 3 things', 'new-run', cmd)
        self.assertTrue(self.check('new-run', cmd, combine=True))

    def test_an_option_without_a_test_is_refused(self):
        # An option that silently does nothing wastes the run and gives a false result.
        self.refused('--stale-context-hours has no test', 'new-run', command('new-run', '--stale-context-hours', '24'))
        (Path(self.tmp.name) / 'tests/test_x.py').write_text('stale_context_hours')
        self.assertTrue(self.check('new-run', command('new-run', '--stale-context-hours', '24')))

    def test_clones_must_keep_the_parent_mf_recipe(self):
        self.refused('only fit the parent', 'new-run', command('new-run', '--dim', '64'), clone_from='parent-run',
                     parent=None)
        self.assertTrue(self.check('new-run', command('new-run', '--tree-refit-policy'), clone_from='parent-run',
                                   parent=None))

    def test_existing_run_ids_are_refused(self):
        self.refused('already exists', 'parent-run', command('parent-run', '--query-clicks'))

    def test_a_parent_marked_invalid_or_without_predictions_is_refused(self):
        bad = self.make_run('bad-run', ' '.join(command('bad-run')))
        (bad / 'SCORED').write_text('invalid: removed no clicks\n')
        self.refused('cannot be a parent', 'new-run', command('new-run'), parent='bad-run')
        (bad / 'SCORED').unlink()
        (bad / 'development_predictions.npz').unlink()
        self.refused('no development predictions', 'new-run', command('new-run'), parent='bad-run')

    def test_resume_must_keep_the_recipe(self):
        args = pipeline.build_parser().parse_args(command('parent-run')[2:])
        (self.parent / 'run.json').write_text('{}')
        (self.parent / 'recipe.json').write_text(json.dumps({'recipe': pipeline.recipe(args)}))
        self.assertTrue(self.check('parent-run', command('parent-run', '--until', 'confirmation'), resume=True))
        changed = ['.venv/bin/python', 'src/history_upgrade.py', '--output-dir', 'runs/parent-run', '--steps', '5000']
        self.refused('do not match the recipe', 'parent-run', changed, resume=True)

    def test_a_finished_unscored_run_blocks_the_next_launch(self):
        # One turn chained three runs this way, and none was scored until it ended.
        (self.parent / 'EXIT').write_text('0\n')
        self.refused('not scored yet', 'new-run', command('new-run', '--query-clicks'))
        (self.parent / 'SCORED').write_text('ok\n')
        self.assertTrue(self.check('new-run', command('new-run', '--query-clicks')))

    def test_seed_counts_above_the_flat_part_of_the_curve_are_refused(self):
        cmd = ['.venv/bin/python', 'src/history_upgrade.py', '--output-dir', 'runs/new-run', '--steps', '3000',
               '--init-std', '0.005', '--seeds', '75']
        self.refused('over the limit', 'new-run', cmd)

    def test_baseline_can_be_a_parent(self):
        self.assertIn('steps: 5000 -> 3000', self.check(
            'new-run', ['python', 'src/history_upgrade.py', '--output-dir', 'runs/new-run', '--steps', '3000'],
            parent='baseline-0.816323')[0])


if __name__ == '__main__':
    unittest.main()
