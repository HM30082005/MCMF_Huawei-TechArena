"""Check an experiment before launch.sh creates anything, so a bad command costs nothing.

  python research/preflight.py [--parent RUN] [--clone-from RUN] [--clone-mf]
                               [--resume] [--combine] RUN_ID -- COMMAND [ARGS...]

Each check below is here because skipping it once wasted a run: wrong flags, code run from
another checkout, the baseline as output folder, uncommitted code, a recipe that differs from
its parent in several ways, and a new option with no test showing that it does anything.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
BASELINE = 'baseline-0.816323'
# Settings that do not change what is being tested.
NOT_RECIPE = {'output_dir', 'until', 'device', 'threads', 'checkpoint_every', 'tree_candidates'}
# Long-standing numeric settings, already covered by the baseline-identity test.
NEEDS_NO_TEST = {'seeds', 'steps', 'batch_events', 'dim', 'l2', 'lr', 'aux_weight', 'negatives',
                 'history_limit', 'init_std'}
TRACKED_CODE = ['src', 'tests', 'research/*.py', 'research/*.sh']
MAX_SEEDS = 30  # The development curve is flat beyond this, and cost grows with every seed.


class Refused(Exception):
    pass


def split_pipeline(command):
    """Return the arguments after history_upgrade.py, or None for any other command."""
    for i, word in enumerate(command):
        if word.endswith('history_upgrade.py'):
            if (ROOT / word).resolve() != (ROOT / 'src/history_upgrade.py').resolve():
                raise Refused(f'Run this checkout\'s code: use src/history_upgrade.py, not {word}.')
            return command[i + 1:]
    return None


def parse(parser, argv):
    errors = io.StringIO()
    try:
        with contextlib.redirect_stderr(errors):
            return parser.parse_args(argv)
    except SystemExit:
        last = errors.getvalue().strip().splitlines()[-1] if errors.getvalue().strip() else 'invalid arguments'
        raise Refused(f'{last}\nLauncher options (--parent, --clone-from, --clone-mf, --combine) go before the '
                      'run ID, not after --. See the valid pipeline flags with: '
                      '.venv/bin/python src/history_upgrade.py --help') from None


def settings(args):
    return {k: v for k, v in vars(args).items() if k not in NOT_RECIPE}


def parent_args(parser, parent):
    """The parent's recipe, from the last pipeline command it ran."""
    if parent == BASELINE:
        return parser.parse_args([])
    folder = RUNS / parent
    if not (folder / 'development_predictions.npz').exists():
        raise Refused(f'Parent {parent} has no development predictions, so nothing can be compared with it. '
                      f'Valid parents are in LEADERBOARD.md.')
    marker = folder / 'SCORED'
    if marker.exists() and marker.read_text().startswith(('invalid', 'failed')):
        raise Refused(f'Parent {parent} is marked "{marker.read_text().strip()}", so it cannot be a parent.')
    lines = (folder / 'commands.txt').read_text().splitlines() if (folder / 'commands.txt').exists() else []
    for line in reversed(lines):
        words = shlex.split(line)
        if any(w.endswith('history_upgrade.py') for w in words):
            start = next(i for i, w in enumerate(words) if w.endswith('history_upgrade.py'))
            try:
                return parse(parser, words[start + 1:])
            except Refused:
                break
    raise Refused(f'Cannot read the recipe of parent {parent} from its commands.txt. Choose another parent.')


def differences(args, parent):
    new, old = settings(args), settings(parent)
    return {k: (old.get(k), new[k]) for k in new if new[k] != old.get(k)}


def tests_text():
    return '\n'.join(p.read_text() for p in sorted((ROOT / 'tests').glob('*.py')))


def check_changes(changes, combine):
    if len(changes) > 1 and not combine:
        listed = ', '.join(f'{k} {a!r} -> {b!r}' for k, (a, b) in changes.items())
        raise Refused(f'This run changes {len(changes)} things from its parent: {listed}. Change one at a time, '
                      'or pick a closer parent. Pass --combine only to join changes already confirmed separately.')
    text = tests_text()
    for name in changes:
        if name not in NEEDS_NO_TEST and name not in text:
            raise Refused(f'--{name.replace("_", "-")} has no test. Add one to tests/ that shows on a small example '
                          'that the option really changes what it claims. An option that silently does nothing '
                          'wastes the run and gives a false result.')


def finished_runs_are_scored(current):
    """Scoring happens between turns, so this also stops one turn from running experiment after experiment."""
    waiting = [p.name for p in sorted(RUNS.iterdir()) if p.name != current and (p / 'EXIT').exists()
               and not (p / 'SCORED').exists()]
    if waiting:
        raise Refused(f'{", ".join(waiting)} finished but is not scored yet. End your turn now: the monitor scores it '
                      'between turns and wakes you with the result.')


def code_is_committed():
    changed = subprocess.run(['git', 'status', '--porcelain', '--', *TRACKED_CODE], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
    if changed:
        raise Refused('Commit your code first, so the run records what actually ran:\n' + changed)


def tests_pass():
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-t', '.'], cwd=ROOT,
                            capture_output=True, text=True)
    if result.returncode:
        raise Refused('Tests fail. Fix your change; never edit an existing test to make it pass.\n'
                      + '\n'.join(result.stderr.strip().splitlines()[-12:]))


def check(opts, parser, recipe):
    command = opts.command[1:] if opts.command[:1] == ['--'] else opts.command
    if not command:
        raise Refused('No command given after --.')
    if '=' in command[0]:
        raise Refused(f'Put {command[0]} before research/launch.sh, not after --.')
    run = RUNS / opts.run_id
    if opts.resume and not (run / 'run.json').exists():
        raise Refused(f'{run.relative_to(ROOT)} has no run.json, so it cannot be resumed. Start it without --resume.')
    if not opts.resume and run.exists():
        raise Refused(f'{run.relative_to(ROOT)} already exists. Choose a new run ID.')
    parent = opts.clone_from or (BASELINE if opts.clone_mf else opts.parent)
    if not opts.resume and not parent:
        raise Refused('Name the run to compare with: --parent RUN or --clone-from RUN, before the run ID.')
    finished_runs_are_scored(opts.run_id)
    code_is_committed()
    argv = split_pipeline(command)
    report = ['Custom command: it must write the run contract files in research/README.md.']
    if argv is not None:
        args = parse(parser, argv)
        if (ROOT / args.output_dir).resolve() != run.resolve():
            raise Refused(f'Pass --output-dir runs/{opts.run_id}. Without it the pipeline targets the baseline folder.')
        if args.steps_per_seed and len(args.steps_per_seed) != args.seeds:
            raise Refused('--steps-per-seed needs one value per seed.')
        if opts.resume:
            saved = run / 'recipe.json'
            if saved.exists() and json.loads(saved.read_text())['recipe'] != recipe(args):
                raise Refused(f'These flags do not match the recipe in {saved.relative_to(ROOT)}. Resume with the same '
                              'flags, changing only --until.')
            report = ['Resuming with the same recipe.']
        else:
            if args.seeds > MAX_SEEDS:
                raise Refused(f'--seeds {args.seeds} is over the limit of {MAX_SEEDS}: the curve is flat beyond that. Compare '
                              'an idea with its parent at the same seed count; if it loses, reject it and move on.')
            parsed_parent = parent_args(parser, parent)
            changes = differences(args, parsed_parent)
            check_changes(changes, opts.combine)
            if opts.clone_from or opts.clone_mf:
                if recipe(args) != recipe(parsed_parent):
                    raise Refused(f'Cloned MF models only fit the parent\'s MF recipe. Use the parent\'s MF flags, or '
                                  f'use --parent {parent} without cloning so the MF is refitted.')
            listed = ', '.join(f'{k}: {a!r} -> {b!r}' for k, (a, b) in changes.items()) or 'none'
            report = [f'Changes from {parent}: {listed}.']
    tests_pass()
    return report


def main():
    # The pipeline imports PyTorch, so it stays out of this module until a check needs it.
    sys.path.insert(0, str(ROOT / 'src'))
    try:
        import history_upgrade as pipeline
    except Exception as error:
        print(f'PREFLIGHT REFUSED, nothing was created:\nsrc/history_upgrade.py does not import: {error!r}. '
              'Fix it, or undo your change with: git checkout -- src/history_upgrade.py', file=sys.stderr)
        sys.exit(5)
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('run_id')
    cli.add_argument('--parent')
    cli.add_argument('--clone-from')
    cli.add_argument('--clone-mf', action='store_true')
    cli.add_argument('--resume', action='store_true')
    cli.add_argument('--combine', action='store_true')
    cli.add_argument('command', nargs=argparse.REMAINDER)
    opts = cli.parse_args()
    try:
        for line in check(opts, pipeline.build_parser(), pipeline.recipe):
            print('Preflight OK. ' + line)
    except Refused as refusal:
        print('PREFLIGHT REFUSED, nothing was created:\n' + str(refusal), file=sys.stderr)
        sys.exit(5)


if __name__ == '__main__':
    main()
