# DIN, DeepFM, and DCNv2 benchmark

Benchmarked locally on October 30, 2026. **DCNv2 was best in this configured comparison, with 0.72038499 ROC AUC.**

All scores below use the exact same January 26 first-event-per-user holdout as strategy 1: **35,138 events, 351,380 candidate rows, and 22,266 clicked rows**. These are local validation scores. The official evaluation dataset has no labels, so its AUC remains unknown until competition upload.

## Holdout results

| Model | ROC AUC | Approx. 95% AUC interval | Change from strategy 1 | Log loss |
|---|---:|---|---:|---:|
| DCNv2 | **0.720385** | 0.7171–0.7236 | +0.014198 | 0.247483 |
| DeepFM | **0.712022** | 0.7088–0.7150 | +0.005835 | 0.266176 |
| DIN | **0.672707** | 0.6695–0.6759 | -0.033480 | 0.294247 |
| Strategy 1 (previous custom model) | 0.706187 | 0.7034–0.7097 | — | 0.254958 |
| Original starter formula | 0.614071 | — | — | — |

The intervals use 100 paired bootstrap resamples of whole first-event-per-user groups with seed 927. They describe sampling variability on this holdout conditional on the trained models; they do not include training-seed or temporal-distribution uncertainty. The paired 95% intervals for the AUC changes from strategy 1 are:

- DCNv2: +0.011691 to +0.016302.
- DeepFM: +0.003120 to +0.008905.
- DIN: -0.037008 to -0.030451.

## Training and selection

- Development fitting: January 20–23, 2025 UTC; 392,831 events / 3,928,310 rows.
- Development scoring: the first event per user in January 24–25; 43,955 events / 439,550 rows.
- Confirmation fitting: January 20–25; 588,873 events / 5,888,730 rows.
- Confirmation scoring: first event per user on January 26; 35,138 events / 351,380 rows.
- Final refit: all January 20–26 training data; 686,824 events / 6,868,240 rows. Predict all 529,390 unlabelled evaluation rows.

Each architecture received four development trials: 32, 64, and 128 dimensions with fitting-only spectral item initialization, plus 64 dimensions with the original random initialization. The highest development AUC selected the dimension, initialization, and epoch. All selections were written to `artifacts/ctr_benchmark/selection_locked.json` before any confirmation model was scored.

| Model | Selected dimensions | Selected epochs | Selected initialization | Development AUC |
|---|---:|---:|---|---:|
| DCNv2 | 128 | 1 | spectral | 0.710946 |
| DeepFM | 128 | 5 | spectral | 0.664396 |
| DIN | 64 | 2 | spectral | 0.637532 |

Common settings: seed 42, AdamW with learning rate 0.001 and weight decay 0.00001, 1,024 events / up to 10,240 candidate rows per batch, binary cross-entropy on actual exposed positives and negatives, gradient clipping at 5, 10% network dropout, and 20% user dropout. Maximum 18 epochs with development-AUC early stopping after 3 non-improving epochs. Confirmation and final models were freshly initialized on their own fitting data and trained for the selected fixed number of epochs.

## Downloaded implementation and inputs

Used the actual, unchanged classes from [FuxiCTR](https://github.com/reczoo/FuxiCTR), pinned to commit `183da1c944ba8930fcdcd54114837c081959d748`. Model-source hashes were checked before and after the run. The adapter handles MPS device placement, batching, the common optimizer loop, and our temporal protocol; it does not replace the public model forward implementations.

Inputs are user/item IDs, time bucket, weekday/weekend, click-history item IDs, history length, age since the latest click, empty-history indicator, candidate repetition indicator/count, and age since the candidate was last clicked. IDs are categorical keys. DIN applies its upstream candidate-conditioned attention; DeepFM and DCNv2 use masked mean history pooling. Candidate and history item embeddings are shared. Unknown fitting-period users/items map to zero.

Spectral initialization follows strategy 1 and uses only fitting-period histories and clicks, with a binary user-item graph collapsing duplicate historical records and repeat edges. Validation and evaluation histories are prediction inputs only. No fitting statistics or initialization use their target labels.

All three selected models used spectral initialization. The from-scratch runs were development comparisons and did not win selection. These results describe this feature setup and bounded search, rather than universal or fully optimized performance for each architecture.

## Verified submission files

All three selected models were refitted on the full training data. Each CSV has exactly `rowid,predicted_probability`, all **529,390 unique evaluation row IDs**, and finite probabilities in [0, 1]. The writer checks alignment against evaluation user/item IDs and timestamps, then independently re-reads the completed CSV.

- `artifacts/ctr_benchmark/submissions/DCNv2.csv` — SHA-256 `47510c2e0f071e334662e8b4ae707b1857f32fdc83e00c993ef23d879b298339`.
- `artifacts/ctr_benchmark/submissions/DeepFM.csv` — SHA-256 `778101e50c62616b825a5a404f75116a22285697725c6322bd6184f269ea7f36`.
- `artifacts/ctr_benchmark/submissions/DIN.csv` — SHA-256 `e04d9e98fe489b63a9d43c2e4806cd98ed123caf570b1a4d4df9a56a16a3f809`.

Checkpoint reloads reproduced all holdout predictions. ROC AUC was independently cross-checked against scikit-learn. Smoke checks cover finite empty-history output, ignoring target labels during forward prediction, row permutation invariance, nonzero training gradients, and adapter alignment. The saved holdout event indices and labels match strategy 1 exactly.

No competition upload was performed. These files are ready for the user to upload. The January 26 window was previously used for exploratory comparison, and the official evaluation covers a longer period with shorter/older histories and more new users; the local scores are not official-score forecasts.

## Reproduce

Implementation: `benchmark_ctr.py`. Numerical records, curves, model-source hashes, selected settings, checkpoint paths, scores by user/history segment, CSV metadata, and bootstrap statistics are in `artifacts/ctr_benchmark/`. Large artifacts and the downloaded vendor repository are ignored by Git.

```sh
uv pip install --python .venv/bin/python -r requirements-ctr-benchmark.txt
git clone https://github.com/reczoo/FuxiCTR.git artifacts/ctr_benchmark/vendor/FuxiCTR
git -C artifacts/ctr_benchmark/vendor/FuxiCTR checkout 183da1c944ba8930fcdcd54114837c081959d748
.venv/bin/python analysis/profile_data.py
OPENBLAS_NUM_THREADS=2 .venv/bin/python benchmark_ctr.py --device auto
```

Skip cloning if the pinned repository is already present. The existing project virtual environment was used for this run; no paid compute or LLM endpoint was required.
