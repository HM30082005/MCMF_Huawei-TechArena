# Historical-click model upgrade

**Official result recorded October 31, 2026:** the user reports **0.816323 ROC
AUC** for `artifacts/history_upgrade/submission_LightGBM.csv`. The local file's
SHA-256 was verified as
`6137599051cfac43a71ec5ed62a96c27deaba4085e0ecaf23252bc46430cb184`.
The competition page was not independently checked; the submission ID, upload
time, and leaderboard partition were not supplied. Durable provenance is in
`analysis/official_scores.json`; the continuation plan is in `CLAUDE_HANDOFF.md`.
No official CatBoost score has been reported.

The selected pipeline scored **0.822252 ROC AUC** on the shared January 26
holdout, up from strategy 2's **0.726809**, an absolute gain of **0.095443**.
Its log loss is **0.190848**, versus 0.213617 previously. The revised MF alone
scored **0.802552 AUC** before the tree stage. These are local validation
results. The separately uploaded LightGBM model has the official result above.

| Model | Shared holdout ROC AUC | Log loss |
|---|---:|---:|
| **Upgraded CatBoost, selected on development** | **0.822252** | **0.190848** |
| Upgraded LightGBM comparison | 0.822508 | 0.190739 |
| Upgraded MF alone | 0.802552 | 0.201567 |
| Previous strategy-2 CatBoost | 0.726809 | 0.213617 |
| Previous strategy-1 MF | 0.706187 | 0.254958 |

LightGBM is numerically higher on confirmation by 0.000256; the submission
retains CatBoost because it was selected before confirmation. The two tree
learners were essentially tied on development.

The selected model's approximate 95% AUC interval is **0.819647–0.824864**.
The paired 95% interval for improvement over strategy 2 is
**+0.092201 to +0.098099**. These use 100 bootstrap resamples of whole
first-event-per-user groups, seed 927. They describe sampling uncertainty
conditional on these trained models, excluding training-seed and temporal
uncertainty.

| Holdout slice | Events | Selected model AUC |
|---|---:|---:|
| Known users | 34,543 | 0.823405 |
| New users | 595 | 0.744032 |
| Empty supplied history | 198 | 0.544556 |
| Supplied history length 1–9 | 7,446 | 0.776727 |
| Supplied history length 10–20 | 27,494 | 0.833390 |

History slices use the original supplied history length for comparability,
although the revised model can use a reconstructed longer history. The small
empty-history slice remains weak and has considerable sampling uncertainty.

This implementation follows the representation and training ideas studied in
[the ATU reference solution](https://github.com/jancastachova/techarena2026-atu/tree/1ed9fd299e4a04352279272bd6be02ed1d08ceec).
It is implemented in `history_upgrade.py` and `history_upgrade_trees.py` and
preserves the earlier strategy-1 and strategy-2 models and submissions.

## Changes implemented

- Remove learned user preference vectors. A user's preference is the mean of
  clicked-item vectors; retain scalar user and item biases plus an intercept.
- Use one shared 32-dimensional item table for candidate and history roles,
  with explicit regularization coefficient 0.05 under the reference model's
  normalization. Initialization is random normal with standard deviation 0.05.
- Reconstruct up to 100 previous clicks from a log deduplicated by user, item,
  and timestamp. Repeats at different times remain distinct. Each event's own
  supplied history is merged into its inference inputs. Current/future clicks
  are excluded from its history.
- Add a historical-click classification objective with weight 0.1 and nine
  sampled alternatives, drawn according to fitting-period item impression
  frequencies. The positive occurrence is excluded from its context, and a
  sampled copy of the positive is masked. Auxiliary context can include other
  clicks after the auxiliary target, provided all precede the prediction-day
  cutoff. Sampled alternatives are not treated as observed non-click exposures.
- Train ten independent parameter sets and average prediction components.
  Independent seeds run together on the Apple GPU; a unit check confirms that
  batching preserves single-model forward values and gradients.
- Replace all old learned-MF features with the new score, item/user biases,
  history dot product, event-relative scores, seed uncertainty, and extended
  history summaries. Both tree learners receive the same 70 numeric features.

The impression loss is binary cross-entropy over actual ten-candidate events,
including no-click events. Each seed receives independently sampled complete
events with replacement, 820 events per update, for 5,000 Adam updates at
learning rate 0.003. This sampling implementation differs from the reference's
permuted passes. Embedding coordinates are never averaged between seeds.

## Development selection

MF fitting uses January 20–23. Tree fitting uses January 22–23; the first two
days provide MF fitting data and counter statistics but are a tree burn-in
period. Development scoring is the first event per user in January 24–25:
43,955 events / 439,550 candidate rows. Prediction-time statistics are frozen
at January 24. This is the same development scoring window used previously.

| Number of MF seeds averaged | Development MF ROC AUC |
|---|---:|
| 1 | 0.757883 |
| 3 | 0.773054 |
| 5 | 0.775772 |
| **10** | **0.778515** |

The old strategy-1 MF scored 0.673415 on this scoring window. Ten seeds were
selected using development labels before confirmation scoring.

| Tree configuration | Selected trees | Development ROC AUC |
|---|---:|---:|
| **CatBoost depth 6** | **1,000** | **0.798399** |
| CatBoost depth 8 | 465 | 0.798323 |
| LightGBM 31 leaves | 272 | 0.798361 |
| LightGBM 63 leaves | 139 | 0.798342 |

The previous strategy-2 pipeline scored 0.698839 on this development window.
The tree-learner differences above are tiny; this comparison supports the
representation changes, not a broad claim of CatBoost superiority.

Each tree trial had a 1,200-tree limit, learning rate 0.05, seed 42, and
development early stopping after 80 non-improving iterations. CatBoost uses
L2 penalty 20, 128 numeric borders, and 0.8 row / 0.85 feature subsampling.
LightGBM uses minimum leaf size 200, L2 penalty 1, and 0.8 row / feature
subsampling. Raw user/item IDs are excluded from both tree feature matrices.

## Temporal protocol

Each training day receives MF predictions from a model fitted strictly before
that day. Both development scoring days use the model fitted before January
24. Confirmation uses the first event per user on January 26: the exact
35,138 events / 351,380 rows and 22,266 positives from our prior comparisons.
Its MF fits January 20–25; its tree fits January 22–25. Selected tree counts
are fixed from development, with no confirmation-based early stopping.

The overall tree learner, each learner's best configuration, and seed count
are recorded in `selection_locked.json` before confirmation fitting/scoring.
Both learners are reported on confirmation, but the development winner remains
the selected model for the final submission.

The final MF uses all January 20–26 training data; the final tree uses January
22–26. Evaluation outcomes are unavailable. Historical-click fitting uses only
fitting-period records; query histories remain inference inputs. This is
stricter than the reference's use of additional pre-cutoff clicks recovered
from held-out/evaluation histories to fit its item representations.

The January 26 window has been examined in earlier experiments, so it is a
shared comparative holdout rather than a new untouched test set. Official
evaluation has a different time/history distribution and requires CSV upload.

## Reproduction and artifacts

Both tree models were refitted on the full available training period using
their settings selected on development. The final MF uses 686,824 events and
1,009,701 unique clicks; its reconstructed training histories average 26.3
clicks. The final tree stage uses 489,807 events / 4,898,070 rows after burn-in.

| Upload file, under `artifacts/history_upgrade/` | Role |
|---|---|
| `submission_history_upgrade.csv` | Primary CatBoost model, selected on development |
| `submission_LightGBM.csv` | LightGBM comparison model |

Each CSV has exactly 529,390 unique evaluation row IDs and finite probabilities
in [0, 1]. The writer checks user/item/time alignment against the raw evaluation
data, then independently re-reads every output row. The agent generated these
files without uploading them; the user subsequently reported the LightGBM
competition score above. SHA-256 checksums:

- CatBoost: `e5540cf95fb11130d350a854ffb96e9576c267d3970b0288b42d013e4be1a022`.
- LightGBM: `6137599051cfac43a71ec5ed62a96c27deaba4085e0ecaf23252bc46430cb184`.

Both confirmation and full-data tree checkpoints were reloaded and reproduced
their predictions. All six daily MF models passed the strict fitting-event and
click-timestamp cutoff checks. Holdout indices and labels match the strategy-2
baseline exactly, and AUC was independently checked against scikit-learn.

The largest CatBoost feature importances belong to the new history dot product
(33.05%), MF score (16.94%), and within-event MF softmax share (14.08%). These
are model-importance measurements, not causal attribution or independent
ablations of the individual changes.

```sh
uv pip install --python .venv/bin/python -r requirements-history-upgrade.txt
.venv/bin/python analysis/profile_data.py
# Required only if the strategy-2 base feature caches are absent:
OPENBLAS_NUM_THREADS=2 .venv/bin/python strategy2.py prepare --device auto
.venv/bin/python -m unittest test_strategy1 test_strategy2 test_history_upgrade -v
OPENBLAS_NUM_THREADS=2 .venv/bin/python history_upgrade.py --device auto
```

The run uses PyTorch 2.15.0 with MPS, CatBoost 1.2.10, and LightGBM 4.7.0 on
CPU. Tree fitting runs in a separate process without importing PyTorch. On
macOS it uses the OpenMP library bundled with the installed PyTorch wheel,
without requiring a system-wide OpenMP installation.

Models, per-seed components, feature caches, locked settings, validation
predictions, numerical results, and verification records are under
`artifacts/history_upgrade/`. The execution log is
`artifacts/history_upgrade_run.log`. Large artifacts stay local and ignored
by Git. Completed daily models/features are reused; incomplete MF fits have
optimizer/RNG checkpoints every 500 updates. CatBoost also saves snapshots;
an incomplete LightGBM fit may restart.

Twenty unit checks passed, including exact CPU interruption/resume and
independent-seed gradient checks. A saved development MF checkpoint reloaded
on CPU matched GPU components on 512 varied events within
0.000000954 absolute difference. Both tree learners passed actual fitting and
checkpoint-reload smoke checks in a process without PyTorch. Additional final
scoring and submission checks are recorded in the run's verification files.
