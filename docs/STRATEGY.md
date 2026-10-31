# Dataset study and five modeling strategies

This recommendation is based on a full scan of both datasets and exploratory
chronological validation, rather than the column descriptions alone. The five
approaches below are promising candidates; identifying the best final model
requires controlled comparisons. No evaluation labels or official leaderboard
scores are available.

## Findings from every row

| Finding | Measurement | Modeling implication |
|---|---:|---|
| Training size | 6,868,240 rows; 686,824 events | Stream CSV once and cache compact event arrays. |
| Evaluation size | 529,390 rows; 52,939 events/users | Evaluate one first post-cutoff event per user. |
| Training users / candidate items | 53,003 / 50,000 | Many sparse categorical identities; learn shared relationships. |
| Positive labels | 432,023; 6.2902% of rows | Use actual exposed negatives and an appropriate probabilistic objective. |
| No-click events | 37.0984% | Explicitly model the possibility of choosing none. |
| Median training events per user | 8 | Regularize user embeddings; history must support sparse and new users. |
| Median impressions per item | 107 | Item-by-context target rates need substantial shrinkage. |
| Candidate already in history | 2.5855% of training candidates | Direct repetition is useful but covers a minority of candidates. |
| Click rate for candidate in history | 29.3104% | Include repeat indicator, count, and elapsed time since that item's last click. |
| Click rate for candidate outside history | 5.6792% | Similarity to other items is essential for predicting new choices. |
| Evaluation users absent from training | 1,453; 2.7447% | Supply a history-only route without learned user IDs. |
| Evaluation candidates absent from training | 0 | Every evaluation candidate has training exposure. |
| Empty histories, train / evaluation | 0.3267% / 1.2562% | Keep a pooled item/context fallback. |
| Median history length, train / evaluation | 20 / 15 | Validation should reflect shorter, sparser evaluation histories. |
| Median last-click age, train / evaluation | 8.45 / 42.55 hours | Random training-row validation would misrepresent recency. |
| Maximum-length histories, train / evaluation | 67.5326% / 39.2074% | A full-history-only solution would miss much of evaluation. |
| Training events containing warm-up clicks | 84.9200% | Histories supply useful clicks from before labelled impressions begin. |
| Evaluation events containing warm-up clicks | 88.2884% | Preserve historical information even for infrequent users. |

Training spans 2025-01-20 through 2025-01-26 UTC. Evaluation spans
2025-01-27 through 2025-02-02 UTC. Only 35,269 of the 52,939 evaluation events
are on January 27; the rest are first events on later days. The oldest history
click is January 6. All evaluation history timestamps precede January 27.

The scan checked all events have ten distinct candidates, shared event context,
binary labels with at most one click, aligned histories of at most 20 entries,
strictly prior and ordered history timestamps, and nonempty fields. Both files
are chronologically sorted and events are contiguous. Candidate-position click
rates are approximately equal; positions and opaque ID characters should not be
invented as preference features.

## Exploratory validation results

For each cutoff, fit exclusively on earlier events, then score the first
available event per user at or after the cutoff. Keep all ten rows together and
calculate ROC AUC over all validation rows globally.

| Model | Jan 25 cutoff: 43,964 events | Jan 26 cutoff: 35,138 events |
|---|---:|---:|
| Smoothed item click rate | 0.5293 | 0.5251 |
| Smoothed item + user click rates | 0.5488 | 0.5465 |
| Starter formula, reimplemented with sparse matrix algebra | 0.6110 | 0.6141 |
| Starter + fixed repeat bonus of 2 log-odds units | 0.6137 | 0.6155 |
| 64-dimensional history embeddings + priors, fixed weight 10 | **0.6371** | **0.6504** |

The embedding probe builds a binary user-item matrix from fitting-period
histories and observed clicks, normalizes for user/item frequency, obtains a
randomized low-rank decomposition, and compares each candidate with the mean
embedding of that event's history. It removes the leading graph component and
uses seed 42. This is an initial spectral model, not a fully trained supervised
factorization or attention model.

These folds overlap for some users/events and are primarily weekend events;
they are diagnostic checks, not independent final tests. Feature ideas and
weight sweeps also make these exploratory results. They do not establish an
official leaderboard score or statistically proven superiority. AUC was checked
against a known 0.75 example, a tie, a perfect ranking, and an inverted ranking.

A fixed 24-hour half-life on the starter's history similarity reduced AUC
(best tested weights: 0.5907 / 0.5930). Item-by-time-bucket CTR plus the original
history score also reduced AUC (0.6093 / 0.6124). Neither result rules out learned
time interactions, but both argue against assuming that stronger recency or
extra context automatically helps.

## 1. Make learned item and history embeddings the main model

This has the strongest immediate empirical support: the small spectral probe
already exceeds the starter on both chronological checks. Start with that
reproducible 64-dimensional model, then compare a supervised logistic
factorization model with 32, 64, and 128 dimensions. Include item and user biases,
a regularized user representation, pooled history embeddings, direct repeat
features, and small context interactions. Train against the labelled candidate
impressions, including all-zero events. Factorization models learn interactions
between sparse categorical variables through shared latent vectors; see the
[original Factorization Machines paper](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Rendle2010FM.pdf).

For auxiliary history pretraining, deduplicate historical clicks by
`(user_id, item_id, timestamp)`. Repeated copies of a historical click across
events are not separate outcomes. Warm-up clicks have no corresponding exposure
denominators, so do not mix them directly into impression-based CTR estimates or
invent non-click labels for every unseen item. Fit all representations from the
fold's fitting data; the held-out event's own supplied history is an inference
input only.

## 2. Train a boosted-tree model over explicit behavioral features

Use CatBoost as the first tree candidate. Include categorical user/item IDs,
smoothed user/item CTR, fitting-period counts, history length and diversity,
candidate repeat count, time since the last matching click, time since any
click, baseline collaborative-filtering score, and the embedding score from
strategy 1. Add maximum/mean candidate-history similarity, and each candidate's
score relative to the other nine candidates. CatBoost supports categorical
feature statistics and combinations; see its
[categorical feature documentation](https://catboost.ai/docs/en/features/categorical-features).

Create label-derived training features from strictly earlier events or
chronological out-of-fold predictions. Never let a row or another row from its
event contribute its outcome to that event's features. Validation aggregates
remain fitted on the fitting period. Tune chronological handling and categorical
regularization, and retain time features only when they improve held-out AUC.
This is the fastest practical companion to the embedding model, and a natural
way to learn how repetition, similarity, and activity interact.

## 3. Learn which parts of history matter for each candidate

Compare a small candidate-conditioned attention model, following the
[Deep Interest Network paper](https://arxiv.org/abs/1706.06978), against mean history
pooling. Each candidate queries the past clicked items; the network learns which
past clicks matter for that candidate. Include relative time gaps and history
position, with proper masks for padding and empty histories.

The reason to test this is the history signal, not an assumption that a larger
neural network will win. Start small, use the same folds and actual candidate
negatives, and require gains over the simpler embedding model. Compare learned
recency against no decay, and avoid committing to a fixed short half-life given
the failed 24-hour experiment and older evaluation histories.

## 4. Model the ten-item choice plus a no-click option

Compare independent binary prediction with a shared candidate scorer and an
event-level softmax over ten candidates plus a learned outside option:

\[
P(i\mid e)=\frac{\exp(s_i)}{\exp(s_{\mathrm{none}})+\sum_{j=1}^{10}\exp(s_j)}.
\]

This architecture is a hypothesis motivated by the verified at-most-one-click
structure and 37.1% no-click rate. It lets a candidate's probability reflect its
competitors and the user's propensity to click anything. The candidate scorer
must share parameters across positions; arbitrary CSV position is not a class
identity. Output only the ten item probabilities in the submission. Retain the
model only if global row-level AUC improves: better within-event ranking or
better probability calibration alone does not establish success on this metric.

## 5. Blend complementary models with sparse-user fallbacks

Blend the embedding, tree, and any successful attention/choice models using
predictions made out of time. Start with a small weighted average; compare
simple rank blending if score scales differ. Choose weights on development
folds and require stability across cutoffs. An ensemble is useful only when its
members make sufficiently different mistakes and the measured gain persists.

For users absent from fitting data, prioritize the history-only embedding path;
for empty histories, use regularized item/context scores and known-user activity
when available. Evaluate these segments separately, but select the final model
on pooled global AUC. Do not normalize every user's ten scores to an identical
range: differences in event-level click propensity matter for cross-event AUC.

## Shared validation and execution plan

1. Use first-event-per-user chronological folds for primary selection, plus all
   later events as a secondary diagnostic. Add a weekday development cutoff to
   the weekend probes and reserve a genuinely untouched later check before
   extensive hyperparameter selection. Report the unavoidable shorter history
   window and weekday differences from the official setup.
2. Keep ten-row events intact. Fit CTRs, similarities, embeddings, and learned
   preprocessing only on the fitting period. Training encodings must exclude
   the current event's labels. Avoid random row splits and label-derived
   aggregate features computed over the full training set.
3. Record global ROC AUC and slices for new users, empty/short histories, and
   activity levels. Check last-click ages and history-length distribution
   against evaluation. Use user/event bootstrap intervals for close decisions.
4. First strengthen and retain the proven spectral baseline, then train
   supervised factorization and CatBoost. Test attention and event-choice
   modeling as controlled alternatives. Blend only complementary winners.
5. Refit the chosen recipe on all labelled training events, produce one finite
   probability for every evaluation `rowid`, verify the submission, and use the
   limited leaderboard submissions for a few confirmed candidates. Preserve
   a reproducible local experiment log rather than tuning on public scores.

No LLM endpoint is needed for these opaque-ID recommendation models. All probes
used local NumPy and SciPy; no new packages, paid compute, or external inference
services were required.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=2 python3 analysis/profile_data.py
OPENBLAS_NUM_THREADS=2 python3 analysis/study.py
OPENBLAS_NUM_THREADS=2 python3 analysis/spectral_probe.py
```

The full CSV parser writes compact arrays into ignored `analysis/.cache/`.
Measurements are saved in `dataset_profile.json`, `history_profile.json`, and
`spectral_probe.json`. `history_profile.json` was generated as an additional
one-off timestamp diagnostic. These analysis files are local additions; this
request did not publish them to GitHub or submit competition predictions.
