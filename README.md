# MCMF – Huawei TechArena 2026

Phase 1 click prediction starter code and competition instructions.

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
