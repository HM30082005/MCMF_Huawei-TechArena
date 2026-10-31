# Strategy 2: behavioral features and CatBoost

Strategy 2 scored **0.726809 ROC AUC** on the same January 26 chronological
holdout as the previous models. Its log loss was **0.213617**. These are local
validation results; `evaluation.csv` has no labels and its official AUC requires
uploading the generated submission to the competition page.

| Model | Shared holdout ROC AUC |
|---|---:|
| **Strategy 2: CatBoost** | **0.726809** |
| DCNv2 | 0.720385 |
| DeepFM | 0.712022 |
| Strategy 1: history factorization | 0.706187 |
| DIN | 0.672707 |
| Original starter | 0.614071 |

The improvement is **+0.006424 over DCNv2** and **+0.020622 over strategy 1**.
The approximate 95% AUC interval is **0.723862–0.730722**, from 100 bootstrap
resamples of whole first-event-per-user groups. Paired comparison intervals are
recorded in `artifacts/strategy2/comparison_statistics.json`: **+0.003991 to
+0.008804 versus DCNv2**, and **+0.019146 to +0.021875 versus strategy 1**.
These intervals
describe sampling variability conditional on the trained models, excluding
training-seed and temporal-distribution uncertainty.

## Implementation and temporal handling

[`src/strategy2.py`](../../src/strategy2.py) implements 66 numeric features and categorical
user/item IDs. The numeric features cover smoothed item/user click rates and
exposure counts, time-bucket and recent fitting-day rates, history length and
diversity, repetition counts and matching-click ages, history timing, sparse
collaborative-filtering similarity, strategy 1's score, and spectral/learned
candidate-history similarities. Candidate scores also have within-event ranks,
centered values, and event summary statistics. Candidate position is excluded.

Each training day's outcome-derived features use only preceding days. All ten
candidates are excluded together from their event's fitting data. Separate
128-dimensional, two-epoch strategy 1 models supply predictions made out of
time; existing matching checkpoints are reused. Historical graph edges are
binary user-item relationships. Impression-based CTR denominators use actual
labelled exposures, with shrinkage, rather than warm-up history counts.

January 20 has no earlier labelled fitting events: its default CTR is 0.05,
embedding features are zero, and a flag marks the missing fitting history.
Supplied click histories remain available as prediction inputs. No evaluation
outcomes are used.

CatBoost uses target-free `Counter` categorical statistics with `SkipTest`,
maximum categorical combination complexity 1, and `has_time=True`. This avoids
row-level target encoding across sibling candidates in an event. Explicit CTR
features remain chronologically fitted. See the official documentation for
[categorical transformations](https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic)
and [counter settings](https://catboost.ai/docs/en/references/training-parameters/ctr).

## Training and selection

Development fitting uses January 20–23: **392,831 events / 3,928,310 rows**.
Development scoring uses the first event per user in January 24–25:
**43,955 events / 439,550 rows**. Both scoring days' features are frozen at
January 24; January 25 scoring features do not incorporate January 24 outcomes.

| Depth | Selected trees | Development AUC |
|---|---:|---:|
| 6 | 400 | 0.698318 |
| **8** | **399** | **0.698839** |

Both trials had a 400-tree limit, learning rate 0.08, and early stopping after
50 non-improving rounds. Other settings: CatBoost 1.2.10, CPU, 12 threads,
seed 42, Logloss objective, AUC selection, 64 numeric borders, L2 penalty 20,
Bernoulli subsampling 0.8, and feature subsampling 0.85. This is a bounded
comparison: improvement near the tree limit leaves room for further development
tuning, and does not establish globally optimal settings.

Depth and tree count were saved before confirmation scoring. The confirmation
model fits January 20–25: **588,873 events / 5,888,730 rows**. It scores the
first event per user on January 26: **35,138 events / 351,380 rows**, including
**22,266 clicks**. Its features are frozen at January 26. The final recipe is
then refitted on all **686,824 training events / 6,868,240 rows**; evaluation
features are frozen at January 27.

The January 26 window has already been used for earlier model comparisons; it
is not an untouched test set. Official evaluation spans a longer period with
older/shorter histories and more new users. Local gains do not guarantee the
same gains on the official data.

| Holdout slice | Events | ROC AUC |
|---|---:|---:|
| Known users | 34,543 | 0.727745 |
| New users | 595 | 0.667316 |
| Empty history | 198 | 0.512757 |
| History length 1–9 | 7,446 | 0.686271 |
| History length 10–20 | 27,494 | 0.736680 |

The empty-history slice is small and remains weak; its score is not a precise
estimate of future performance for that segment.

## Outputs, checks, and reproduction

The upload file is `artifacts/strategy2/submission_strategy2.csv`. Numerical
results, feature importances, predictions, selected settings, checkpoints,
feature provenance, bootstrap comparisons, and verification records are saved
under `artifacts/strategy2/`. The execution log is
`artifacts/strategy2_run.log`. These large artifacts remain local and ignored by
Git.

The CSV contains **529,390 unique evaluation row IDs**, with finite probabilities
between **0.001687 and 0.720224**. Its SHA-256 is
`3cbc777dca951a21d6bbb0e0e98662aff99b7521419dcded1872844add3321e0`.
No competition upload was performed.

The pipeline verifies holdout indices and labels against strategy 1, checks AUC
against scikit-learn, reloads the saved confirmation checkpoint, and validates
submission row alignment, unique IDs, and finite probabilities. Eleven unit
checks cover both strategies, including target independence, strict temporal
boundaries, candidate permutation behavior, empty histories, and the actual
CatBoost quantized fitting path.

```sh
uv pip install --python .venv/bin/python -r requirements/strategy2.txt
.venv/bin/python analysis/profile_data.py
.venv/bin/python -m unittest tests.test_strategy1 tests.test_strategy2 -v
OPENBLAS_NUM_THREADS=6 .venv/bin/python src/strategy2.py all --device auto
```

Completed feature files and tree models are reused on rerun. Feature files and
new checkpoints use temporary files followed by replacement; CatBoost also
saves fitting snapshots every 30 seconds. An interrupted short embedding fit
may need to restart. To train exclusively from completed feature caches, use
`src/strategy2.py train --device cpu`. No paid compute or LLM endpoint is required.
