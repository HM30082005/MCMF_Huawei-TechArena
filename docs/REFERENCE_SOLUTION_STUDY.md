# Study of the ATU TechArena reference solution

Inspected on October 30, 2026. The repository was cloned separately into
`external/techarena2026-atu/` at commit
`1ed9fd299e4a04352279272bd6be02ed1d08ceec`. The clone is ignored by our parent
repository and its source files were left unchanged. No model training or
competition upload was performed during this study.

## Score evidence

The authors' detailed [results and upload log](https://github.com/jancastachova/techarena2026-atu/blob/1ed9fd299e4a04352279272bd6be02ed1d08ceec/experiments/RESULTS.md)
report **0.815842 public ROC AUC** for experiment 11. This is a reported
leaderboard result, not an independently reproduced score.

| Experiment | Incremental change | Reported public AUC |
|---|---|---:|
| 03 | Initial chronological MF + LightGBM pipeline | 0.727931 |
| 04 | History-based preference vector; remove learned user preference vectors | 0.766866 |
| 05 | Auxiliary historical-click training | 0.777089 |
| 06 | Average five MF runs | 0.795989 |
| 07 | Average ten MF runs | 0.798027 |
| 09 | Reconstruct histories beyond the supplied 20 clicks | 0.804933 |
| 10 | Train a historical click using the user's other known clicks as context | 0.812971 |
| 11 | Share history and candidate item vectors | **0.815842** |

These sequential comparisons include model interactions and do not establish
that each improvement will transfer independently to our models.

The newer [simplified notes](https://github.com/jancastachova/techarena2026-atu/blob/1ed9fd299e4a04352279272bd6be02ed1d08ceec/experiments/resultsSIMPLE.md)
mention approximately 0.818 public AUC for experiment 13 and a 0.83571 local
baseline. The detailed upload log stops at experiment 11; the newer claims have
less accompanying score provenance. They are not evidence of 0.83571 official
AUC. Experiment 15 adds another set of random seeds and a submission average.

Our current 0.726809 is a January 26 local score. It cannot be directly subtracted
from their public leaderboard scores. Their default local test starts January
25 and includes first events per user through January 26, which also differs
from our one-day confirmation window.

## What the model actually learns

The preference model is compact:

\[
s(u,i)=b_0+b_u+b_i+
\left(\frac{1}{|H_u|}\sum_{h\in H_u}v_h\right)^T v_i.
\]

There is one shared 32-dimensional vector per item. A user has a scalar bias,
but no independently learned preference vector. The history average supplies
their preference direction. Empty histories contribute zero to the dot product.
This differs from our 128-dimensional model with separate item/history tables
and a learned user preference vector.

The [history model](https://github.com/jancastachova/techarena2026-atu/blob/1ed9fd299e4a04352279272bd6be02ed1d08ceec/experiments/04_history_mf/mf_stage.py)
uses explicit embedding regularization, with coefficient 0.05 and its own batch
normalization. That coefficient is not directly comparable with our differently
normalized penalties. Experiment 11 trains each seed for 5,000 optimizer steps,
using Adam at 0.003 and batches of 820 complete events.

The [click augmentation stage](https://github.com/jancastachova/techarena2026-atu/blob/1ed9fd299e4a04352279272bd6be02ed1d08ceec/experiments/05_click_aug/mf_stage.py)
adds a sampled classification loss with weight 0.1. Each historical click is
the positive among nine sampled alternatives. Sampling follows fitting-period
candidate impression frequencies, rather than uniform item sampling. A sampled
copy of the positive is masked. These alternatives are auxiliary contrastive
examples, not additional observed non-click impressions.

The [long-history stage](https://github.com/jancastachova/techarena2026-atu/blob/1ed9fd299e4a04352279272bd6be02ed1d08ceec/experiments/09_full_history/mf_stage.py)
reconstructs a deduplicated click log keyed by user, item, and timestamp. It
combines histories across events and observed fitting clicks. Each event uses
only clicks strictly before its timestamp, capped at the most recent 100.
Repeated clicks on the same item at different times remain separate records.

For auxiliary training, [experiment 10](https://github.com/jancastachova/techarena2026-atu/blob/1ed9fd299e4a04352279272bd6be02ed1d08ceec/experiments/10_other_clicks/mf_stage.py)
uses the user's other known clicks as context, excluding the target occurrence.
Context can follow the auxiliary target's timestamp, but every context click
must precede the day being predicted. This is representation fitting on past
data, rather than using a scored event's future click.

[Experiment 11](https://github.com/jancastachova/techarena2026-atu/blob/1ed9fd299e4a04352279272bd6be02ed1d08ceec/experiments/11_tied_items/mf_stage.py)
ties the candidate table to the history table. It averages prediction components
over ten independently initialized runs; it does not average embedding vectors,
whose coordinates can differ between runs.

The downstream LightGBM takes 39 numeric features: 28 behavioral features, four
MF components, and seven event-relative features. It has no raw user/item ID
categorical inputs. The MF components include the total score, item bias, user
bias, and history dot product. Relative features include score ranks, candidate
softmax shares, and the event log-sum-exp. Default LightGBM settings use binary
loss, 63 leaves, minimum leaf size 200, and learning rate 0.05.

## Temporal handling and reproducibility

Static inspection of the main training paths found these safeguards:

- Label-based counters update at training-day boundaries and remain frozen
  after the fitting cutoff. All ten event candidates stay together.
- A fitting day's MF features come from a model fitted on earlier days.
- Test events are the first event per user after the cutoff; later events are
  discarded. Held-out click labels are stored for scoring and do not enter the
  fitting click log or outcome counters in the inspected paths.
- The tree stage starts at day 2, after an MF burn-in period. Tree count is
  chosen on the last fitting day, scaled by the added fitting days, then refit.
- The click log includes pre-cutoff clicks recovered from held-out/evaluation
  histories. Our existing pipelines have kept those histories as inference
  inputs only. Reproduction must explicitly account for this additional use of
  supplied, unlabelled inputs. The code checks that the log precedes the cutoff.

This is a source review, not a full execution-based proof against leakage.
Earlier experiments 01/02 used an optimistic validation procedure that the
authors explicitly superseded.

The newer self-contained [experiment 13](https://github.com/jancastachova/techarena2026-atu/blob/1ed9fd299e4a04352279272bd6be02ed1d08ceec/experiments/13_retune_7days/retune.py)
uses 15 seeds, 3,000 steps, and random initialization standard deviation 0.005.
It is easier to inspect than experiment 11's chain of imports and runtime class
replacement. However, its defaults were tuned using January 26 as a fitting
check for the official run; that day is our existing confirmation window.
Treat a comparison with those published settings as diagnostic, not a new
untouched test.

Some experiment-13 help examples reference `12_standalone/standalone.py`, which
is absent from this commit. Use the actual experiment-13 script path. The
requirements pin NumPy, pandas, SciPy, PyTorch, and LightGBM; no LLM endpoint is
involved. The upstream pipeline separates PyTorch and LightGBM into processes
because of a reported macOS OpenMP conflict.

## What this changes for our next experiment

The strongest concrete reproduction target is **experiment 11**, whose score
has the clearest log. Evaluate its published recipe at cutoff 6 (January 26)
against our exact held-out events before adapting individual components.

```sh
# From external/techarena2026-atu, with its dependencies installed separately:
.venv/bin/python experiments/11_tied_items/tied_items.py --cutoff 6 \
  --train ../../train.csv --evaluation ../../evaluation.csv
```

For our own model, prioritize a history-based preference representation, a
deduplicated extended history, the auxiliary sampled-click loss, shared item
vectors, and prediction averaging. Keep scalar activity biases. Introduce
changes in controlled comparisons, starting with one seed and then increasing
the seed count. Compare tree learners only after the representation is fixed.

The authors report that a choice loss, candidate-aware attention, and extra
recency weighting did not improve their final pipeline. That lowers their
priority relative to the concrete history/model changes above. It does not
establish that those alternatives can never help our setup.

## Local checks completed

All 27 Python source files parsed successfully without executing the training
scripts. Both committed submissions `11_tied_items.csv` and
`15_avg_13_15.csv` have the exact two-column header, all **529,390 unique row
IDs** from our actual `evaluation.csv`, matching order, and finite probabilities
in [0, 1]. The experiment-11 CSV SHA-256 is
`6af3e202d0de1c0e5d3a0ae5a8992f56f5f221422856190e141abe5db6a3e899`.

Machine-readable checks are in `analysis/external_repo_audit.json`. No upstream
training was run and no official leaderboard result was independently verified.
