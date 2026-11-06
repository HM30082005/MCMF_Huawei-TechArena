# Research harness

Tools for running and comparing experiments against the 0.816323 baseline.
They are used by people and by the autonomous Hermes research job alike.

## Evaluation protocol

| Window | Rows | Model fitting cutoff | Use |
|---|---|---|---|
| `development` | First event per user, Jan 24–25 (43,955 events) | Before Jan 24 | Search: compare every idea here |
| `confirmation` | First event per user, Jan 26 (35,138 events) | Before Jan 26 | Reserved: only development winners |

A run is **eligible for confirmation** when its development AUC beats the
baseline by at least +0.001, the 95% paired bootstrap interval of that gain is
above zero, and the gain is positive on Jan 24 and on Jan 25 separately. At
most two confirmation looks are allowed per UTC day. A confirmed run with a
positive confirmation gain is a **submission candidate**. Candidates get a
full-data refit and a submission CSV, and a person decides whether to upload.

January 26 has been reused in earlier development, so it is a sanity check,
not an untouched test set. Official uploads are never automated.

## Run contract

Each run lives in `runs/<run_id>/` and writes, using the stage names of
`src/history_upgrade.py --output-dir runs/<run_id>`:

- `development_predictions.npz`: `event_indices`, `logits` of shape (N, 10)
- `confirmation_predictions.npz`: same, after `--until confirmation`
- `evaluation_logits.npy` and `submission_*.csv`: after `--until final`

Any pipeline can take part by writing these files for exactly the window rows.
Runs never report their own scores.

## Tools

| Tool | Purpose |
|---|---|
| `evaluate.py score RUN --store S [--confirm]` | Score a run; appends to `S/scores.jsonl`, regenerates `S/LEADERBOARD.md` |
| `evaluate.py leaderboard --store S` | Regenerate the leaderboard |
| `reproduce_baseline.py` | Re-predict the baseline from saved trees, check the official CSV hash, and export it as the reference run |
| `launch.sh --parent RUN \| --clone-from RUN \| --clone-mf RUN_ID "hypothesis" -- CMD` | Start one detached job (one at a time) with a recorded identity; `--clone-from` and `--clone-mf` reuse an existing MF. Hides Jan 26 results from pipeline output |
| `preflight.py` | Run by `launch.sh`: refuses bad flags, wrong folders, uncommitted code, failing tests, untested options and more than one change from the parent, before anything is created |
| `blend.py RUN_ID "hypothesis" A B ...` | Equal-weight logit blend of finished runs, written as a new run and scored like any other |
| `monitor.py` | Per-tick lab monitor used by the Hermes job: scoring, integrity checks, alerts |
| `install_lab.sh MAIN_CHECKOUT` | One-time lab setup on a machine with the data and baseline artifacts |

A typical development-only search run:

```sh
research/launch.sh --parent baseline-0.816323 shorter-fit-3000 "3,000 MF updates generalise better than 5,000" -- \
  .venv/bin/python src/history_upgrade.py --output-dir runs/shorter-fit-3000 \
  --steps 3000 --until development --tree-candidates LightGBM:31
```

## Public source snapshot

This directory retains the original experiment infrastructure and historical
notebooks. For a concise, corrected account of the results, start with
[the research summary](../docs/experiments/research-summary.md). The generated
[leaderboard snapshot](../docs/experiments/research-leaderboard.md) is archived
separately; a clean clone does not contain live runs or an evaluator store.

`evaluate.py` and the synthetic unit tests can be used without Hermes. The
launcher and installer preserve the original **macOS/Hermes-specific** setup:
APFS clone copies, `caffeinate`, a shared local environment, and saved baseline
artifacts. `install_lab.sh` changes local artifact permissions and creates a
worktree and files under `~/.hermes`; it is not part of the standard setup in
the root README. `monitor.py` integrates with Hermes notifications and requires
that local configuration. Do not run the installer just to review the code.

The archived notebooks and agent instructions describe the historical research
process. Some notebook claims were later corrected; they are not a substitute
for evaluator evidence. The current launcher limits new searches to 30 seeds,
so it will refuse archived higher-seed recipes. Direct CLI reproduction is
documented in the root README.
