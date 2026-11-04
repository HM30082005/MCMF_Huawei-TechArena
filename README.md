# MCMF – Huawei TechArena 2026

Click prediction for the TechArena 2026 Phase 1 challenge: score each of the
ten candidate items shown in an event with a click probability, using opaque
user/item IDs, recent click histories, and event context. Submissions are
ranked by global row-level ROC AUC. The full task contract is in
[INSTRUCTIONS.md](INSTRUCTIONS.md).

## Current best

**Official ROC AUC 0.816323**: ten-seed shared-item history matrix
factorisation, followed by LightGBM ([`src/history_upgrade.py`](src/history_upgrade.py)).
The exact submission checksum and score provenance are in
[`results/official_scores.json`](results/official_scores.json).

**[`docs/PROGRESS.md`](docs/PROGRESS.md) lists every direction tried, with scores
and improvements.** The best model since this score reaches 0.8247 locally; its
official score is unknown until uploaded.

| Approach | Local AUC (Jan 26) | Official AUC | Report |
|---|---:|---:|---|
| Starter solution | 0.6141 | – | [`src/starter_solution.py`](src/starter_solution.py) |
| Strategy 1: user/history factorisation | 0.7062 | – | [strategy1](docs/experiments/strategy1.md) |
| FuxiCTR DIN / DeepFM / DCNv2 | 0.6727 / 0.7120 / 0.7204 | – | [ctr-benchmark](docs/experiments/ctr-benchmark.md) |
| Strategy 2: behavioural CatBoost | 0.7268 | – | [strategy2](docs/experiments/strategy2.md) |
| History upgrade + CatBoost | 0.8223 | – | [history-upgrade](docs/experiments/history-upgrade.md) |
| **History upgrade + LightGBM** | **0.8225** | **0.816323** | [history-upgrade](docs/experiments/history-upgrade.md) |

Local AUC is measured on each user's first event on January 26, 2025, with
models fitted only on earlier days. That window has been reused for
comparisons, so it is not an untouched test set. Local and official scores
are not directly comparable.

**Start with [`docs/HANDOFF.md`](docs/HANDOFF.md)**: current state, temporal
validation protocol, what has already failed, and the prioritised plan.

## Repository layout

```text
src/            pipeline scripts (run from the repository root)
tests/          unit tests
analysis/       data parsing/profiling scripts and small JSON audits
docs/           handoff, strategy notes, reference study
docs/experiments/  one report per experiment: settings, results, reproduction
requirements/   pinned dependency sets (each includes the one it builds on)
results/        official_scores.json: append-only competition score record
```

Not in Git: the raw CSVs, `analysis/.cache/` (encoded arrays), `artifacts/`
(models, features, predictions, submissions), `external/` (reference clones)
and `.venv/`.

## Setup

1. **Data.** Download the three `.csv.gz` assets from the
   [dataset-v1 release](https://github.com/HM30082005/MCMF_Huawei-TechArena/releases/tag/dataset-v1)
   into the repository root and decompress them. The release lists SHA-256
   checksums for both compressed and original files (about 7.7 GB in total).

   ```sh
   gunzip train.csv.gz evaluation.csv.gz example_submission.csv.gz
   ```

2. **Environment.** Python 3.12. With [uv](https://docs.astral.sh/uv/):

   ```sh
   uv venv --python 3.12 .venv
   uv pip install --python .venv/bin/python -r requirements/history-upgrade.txt
   ```

3. **Encoded cache.** Parses the CSVs into memory-mapped arrays under
   `analysis/.cache/`:

   ```sh
   .venv/bin/python analysis/profile_data.py
   ```

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -t . -v
```

One test reads `analysis/.cache/`, so run step 3 first.

## Reproducing the best pipeline

```sh
# Builds the strategy-2 behavioural feature caches the upgrade depends on.
OPENBLAS_NUM_THREADS=2 .venv/bin/python src/strategy2.py prepare --device auto
OPENBLAS_NUM_THREADS=2 .venv/bin/python src/history_upgrade.py --device auto
```

This writes to `artifacts/history_upgrade/`, **overwriting** any earlier run
there. Back that directory up first. Full training took about 25 minutes on
an Apple M5 Pro (MPS for the factorisation, CPU for the trees). On macOS, the
tree worker uses the OpenMP runtime bundled with PyTorch; `history_upgrade.py`
sets this up for its child process. See
[docs/experiments/history-upgrade.md](docs/experiments/history-upgrade.md)
for details and verification.

The dependency-free starter baseline:

```sh
python3 src/starter_solution.py --train train.csv --evaluation evaluation.csv --out submission.csv
```

## Conventions

- **Validate out of time.** Fit only on days before the scored window and
  score each user's first event, like the evaluation set. Never use random
  row splits.
- **Record official scores** in `results/official_scores.json` against the
  exact SHA-256 of the uploaded CSV. The file is append-only; never edit or
  remove earlier entries.
- **Write up each experiment** in `docs/experiments/`: hypothesis, settings,
  matched-row local results, and reproduction commands.
- **Do not commit** datasets, artifacts, submissions or credentials.
