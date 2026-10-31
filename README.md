# MCMF – Huawei TechArena 2026

Phase 1 click prediction starter code and competition instructions.

**Latest recorded official ROC AUC: 0.816323**, reported by the user on
October 31, 2026, for the historical-click upgrade's LightGBM submission.
The exact CSV checksum and score provenance are recorded in
[`analysis/official_scores.json`](analysis/official_scores.json).
See [`CLAUDE_HANDOFF.md`](CLAUDE_HANDOFF.md) for the complete development handoff,
implementation details, previous experiments, and prioritized improvement plan.

- [Competition instructions](INSTRUCTIONS.md)
- [Starter solution](starter_solution.py)
- [Download all three CSV files](https://github.com/HM30082005/MCMF_Huawei-TechArena/releases/tag/dataset-v1)

## Dataset

The original CSV files total approximately 7.7 GB. They are provided as losslessly
compressed `.csv.gz` assets in the `dataset-v1` release because the training and
evaluation files exceed GitHub's regular Git file size limit. Download all three
assets into this folder, then decompress them:

```sh
gunzip train.csv.gz evaluation.csv.gz example_submission.csv.gz
```

This restores the original CSV files byte for byte. Dataset files stay local and
are excluded from Git commits. The release includes SHA-256 checksums for both
the compressed downloads and the original CSV files.

## Run the starter solution

Requires Python 3 and uses only the standard library.

```sh
python3 starter_solution.py --train train.csv --evaluation evaluation.csv --out submission.csv
```

Read `INSTRUCTIONS.md` for the task, validation guidance, and submission format.

## Supervised strategy 1

The trained history-aware factorization model is implemented in
[`strategy1.py`](strategy1.py). See [`STRATEGY1_RESULTS.md`](STRATEGY1_RESULTS.md)
for reproduction commands, validation results, and the generated submission.
It scored **0.7062 ROC AUC** on the January 26 chronological confirmation split;
the official score on the unlabelled evaluation dataset requires uploading the
submission to the competition page.

## Public CTR model benchmark

[`benchmark_ctr.py`](benchmark_ctr.py) trains the downloaded FuxiCTR DIN, DeepFM,
and DCNv2 implementations with the same chronological development and
confirmation windows. DCNv2 scored **0.7204 ROC AUC**, DeepFM **0.7120**, and DIN
**0.6727** on the shared January 26 holdout. See
[`CTR_BENCHMARK_RESULTS.md`](CTR_BENCHMARK_RESULTS.md) for settings, validation,
reproduction commands, and the three full-data submission files.

## Behavioral strategy 2

[`strategy2.py`](strategy2.py) trains CatBoost over 66 behavioral and embedding
features with predictions and outcome-derived statistics made out of time. It
scored **0.7268 ROC AUC** on the same January 26 holdout. See
[`STRATEGY2_RESULTS.md`](STRATEGY2_RESULTS.md) for training settings, temporal
checks, reproduction commands, and the full-data submission file.

## Historical-click model upgrade

[`history_upgrade.py`](history_upgrade.py) replaces the learned user preference
vectors with shared item vectors pooled over extended histories, adds an
auxiliary historical-click loss, and averages ten model runs. CatBoost over
the revised features scored **0.8223 ROC AUC** on the shared January 26 holdout,
up from strategy 2's 0.7268. See
[`HISTORY_UPGRADE_RESULTS.md`](HISTORY_UPGRADE_RESULTS.md) for the implementation,
development selection, validation protocol, and reproduction commands.
The LightGBM comparison scored 0.8225; both full-data upload CSVs are listed in
the results report. Its **user-reported official competition AUC is 0.816323**;
the CatBoost submission has no reported official score.
