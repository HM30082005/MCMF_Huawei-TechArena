# Strategy 1: trained supervised history factorization

Implemented in `strategy1.py` and trained locally on October 30, 2026 using the
Mac's Apple M5 Pro GPU. The selected model has **128-dimensional embeddings** and
uses **two supervised epochs** after fitting-only spectral initialization.

**Local chronological confirmation ROC AUC: 0.70618675.** The original starter
formula scored 0.61407149 on the same first-event-per-user January 26 split,
an improvement of 0.09211526 AUC. This is a local measurement, not an official
score on `evaluation.csv`.

## What was implemented

- Separate trainable candidate-item and historical-item vectors.
- Regularized trainable user vectors, exposure-based shrinkage, and 20% user
  dropout so history remains useful when a user ID is absent from fitting data.
- Mean pooling of the user's provided click history, with a learned history
  score scale and explicit empty-history masking.
- Global, user, and item biases.
- Direct repeat indicator, repeat count, and time since the candidate's previous
  click, plus history length and time since any prior click.
- Small item-by-time-bucket and item-by-weekend interactions.
- Binary cross-entropy on every displayed candidate, including all-zero events;
  AdamW optimization, embedding/user regularization, and gradient clipping.
- A comparison of 32, 64, and 128 dimensions, with early stopping by development
  ROC AUC, followed by a locked confirmation run and a full-data refit.

Spectral initialization uses only the fitting period's labels and histories.
Its binary user-item graph collapses duplicate historical records and repeat
edges, so a click copied into many histories is not counted repeatedly in the
initialization. Warm-up clicks enter that graph without inventing exposure
denominators or non-click labels. The supervised model retains real repeated
clicks as part of each event's history.

## Validation and selection

All dates below refer to dataset timestamps in UTC. Every validation user
contributes the first event within the stated window, with all ten candidates.

| Stage | Fitting period | Scoring period | Fitting events | Scored events |
|---|---|---|---:|---:|
| Development | Jan 20–23 | Jan 24–25 | 392,831 | 43,955 |
| Confirmation | Jan 20–25 | Jan 26 | 588,873 | 35,138 |
| Final model | Jan 20–26 | Unlabelled evaluation | 686,824 | 52,939 |

| Embedding dimensions | Selected development epoch | Development ROC AUC |
|---|---:|---:|
| 32 | 6 | 0.6445 |
| 64 | 3 | 0.6574 |
| **128** | **2** | **0.6734** |

Dimension, epoch count, learning rate, and regularization were fixed before
scoring the supervised model on the January 26 confirmation set. Its labels
were not used for that selection. The January 26 split had appeared in the
earlier exploratory baseline study, so this is a confirmation comparison rather
than a claim of a completely untouched final test.

The full-data model was initialized afresh on all labelled training data and
trained for the selected two epochs. Evaluation histories were used only as
prediction inputs. There are no evaluation labels to fit or score against.

## Confirmation measurements

| Measurement | Value |
|---|---:|
| Global ROC AUC | **0.70618675** |
| Approximate 95% bootstrap interval | 0.7034–0.7097 |
| Log loss | 0.25495826 |
| Scored candidate rows | 351,380 |
| Positive labels | 22,266 |
| Known-user AUC | 0.7075 |
| New-user AUC | 0.6402 |
| Histories with 1–9 clicks AUC | 0.6689 |
| Histories with 10–20 clicks AUC | 0.7178 |
| Empty-history AUC | 0.4718 |

The interval uses 100 bootstrap resamples of complete first-event-per-user
groups. Empty-history performance is a weakness, measured on only 198 events
and 90 positive candidate rows. Shorter histories also score lower. Evaluation
contains more empty/short histories and spans seven days, whereas confirmation
spans one day; the local AUC should not be presented as a forecast of the
official score. The model was selected for AUC; probability calibration remains
an improvement opportunity given its log loss and extreme predictions.

## Submission and verification

`submission_strategy1.csv` contains **529,390 predictions and unique row IDs**.
The header is exactly `rowid,predicted_probability`. Every output row was checked
against the original evaluation row ID, encoded user/item, and timestamp, and
the finished CSV was independently re-read to check its keys and finite values
in [0, 1]. Float64 sigmoid and 12 significant digits preserve ranking precision;
applying the same output precision to the confirmation scores leaves AUC
unchanged at the reported precision.

SHA-256 of the CSV:

```text
4da97e4ceca5570127159012b3c819ec847386606184233133a92f76f62a6f3f
```

Saved checkpoints were independently reloaded on CPU, with their predictions
compared against GPU results. Six automated tests cover chronological splitting,
target-independent predictions, unknown-user masking, candidate permutation
invariance, empty histories/gradients, and AUC including ties. Encoded-input
fingerprints protect checkpoint and initialization reuse from stale mappings.

The user will upload the CSV to the competition page. No official submission
was made, and **official evaluation AUC is unknown**.

## Files and reproduction

- `strategy1.py`: training, validation, model selection, prediction, CSV checks.
- `test_strategy1.py`: focused correctness tests.
- `requirements-strategy1.txt`: exact installed numerical/training dependencies.
- `artifacts/strategy1/final_d128.pt`: trained full-data checkpoint.
- `artifacts/strategy1/results.json`: settings, epoch curves, slices, CSV checksum.
- `artifacts/strategy1/verification.json`: checkpoint reload verification.
- `artifacts/strategy1/confirmation_predictions.npz`: local holdout predictions.
- `artifacts/strategy1/user_ids.json` and `item_ids.json`: encoding maps.

Large model files, encoded arrays, and generated submissions are ignored by Git.
The new implementation and results are local; they have not been pushed.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-strategy1.txt
.venv/bin/python analysis/profile_data.py
.venv/bin/python -m unittest -v test_strategy1
OPENBLAS_NUM_THREADS=2 .venv/bin/python strategy1.py train --device auto
```

To regenerate the submission from the saved full-data checkpoint and matching
encoded arrays without training again:

```sh
.venv/bin/python strategy1.py predict --checkpoint artifacts/strategy1/final_d128.pt
```

PyTorch's official documentation describes
[embedding layers](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding)
and the local [MPS GPU backend](https://docs.pytorch.org/docs/stable/notes/mps.html).
