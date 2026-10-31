# TechArena click prediction — development handoff for Claude

Prepared **2026-10-31**, Europe/Dublin. This is a continuation brief for the
existing project, including measured results, implementation constraints,
and proposed experiments. Experiments in sections 7–8 are proposals unless
explicitly marked as completed. No further models were trained to prepare
this handoff.

## 1. Start here

**The current official baseline is ROC AUC 0.816323 for our LightGBM CSV.**
The user reported this competition result on October 31, 2026. We verified
the local file checksum, but did not inspect the competition page. The
submission ID, upload timestamp, and public/private partition were not given.
The instructions say that only public leaderboard scores update during Phase
1; do not silently convert this report into a verified private score.

| Baseline identity | Value |
|---|---|
| Pipeline | Ten independently initialized shared-item history models, then LightGBM |
| Submission | `artifacts/history_upgrade/submission_LightGBM.csv` |
| SHA-256 | `6137599051cfac43a71ec5ed62a96c27deaba4085e0ecaf23252bc46430cb184` |
| Evaluation rows | 529,390 |
| Final tree checkpoint | `artifacts/history_upgrade/trees/final_LightGBM_31.txt` |
| Official AUC, user reported | **0.816323** |
| January 26 local confirmation AUC | **0.8225077648886067** |
| Local confirmation log loss | 0.1907389450310171 |
| Durable official-score record | `analysis/official_scores.json` |

The local score exceeds the official score by 0.006184765. This is one
comparison across different samples and time distributions, not a correction
factor to subtract from future local results. Reaching approximately 0.83
official AUC is the user's aspiration, not an established achievable result.

**Preserve this baseline before experimentation.** The current training
scripts have a fixed output directory and incomplete cache identities. First
make experiments use separate directories and preserve the exact CSV, models,
features, configuration, and predictions above. Do not overwrite them while
trying a new seed count or training duration.

Project root on the original computer:

```text
/Users/mihailharetonenko/Hackathons/Huwawei Tech Arena
```

Paths in code blocks and artifact names below are relative to that root.
Local hyperlinks point to the original computer; adjust them on another
machine. The main GitHub repository is
[HM30082005/MCMF_Huawei-TechArena](https://github.com/HM30082005/MCMF_Huawei-TechArena).
**The new model code, reports, and tests are still local, uncommitted files at
handoff time. A clone of GitHub alone does not contain the completed work.**
The checked-out committed revision is
`600cb58baa398293d7458e73a7716707ba8b9d1b`; inspect `git status` and transfer the
working files as well. No commit, push, or new competition upload was made
while preparing this document.

## 2. The challenge and the data

This is synthetic recommendation data. Predict a click probability for each
candidate item, using opaque user/item identifiers, past clicks, and event
context. An item ID identifies an individual item, not an item category.
There are no item descriptions, semantic categories, or user profiles to give
to a pretrained language model. An Ollama chat model is therefore not a
drop-in click predictor for these inputs.

| Property | Training | Evaluation |
|---|---:|---:|
| Candidate rows | 6,868,240 | 529,390 |
| Events | 686,824 | 52,939 |
| Users with candidate events | 53,003 | 52,939 |
| Candidates per event | 10 distinct items | 10 distinct items |
| Time range, UTC | January 20–26, 2025 | January 27–February 2, 2025 |
| Click labels | Available | Unavailable |
| Supplied history | Latest 0–20 earlier clicks | Latest 0–20 earlier clicks |
| Median supplied history length | 20 | 15 |
| Empty supplied histories | 0.327% | 1.256% |
| Median last-click age | 8.45 hours | 42.55 hours |

Training contains 432,023 clicked rows: row CTR 0.06290156. Exactly zero or
one candidate is clicked per event; 37.10% of events have no click. Histories
are clicks only, ordered oldest to newest, with aligned timestamps. They can
contain repeated items. Warm-up clicks reach back to January 6, before the
labelled training period; their impression denominators are unavailable.

Evaluation contains each user's **first event at or after January 27**.
There are 1,453 evaluation users without training events, about 2.745%.
No evaluation candidate item is new relative to training candidate coverage.
All clicks appearing in supplied evaluation histories were verified to be
before January 27. Evaluation's older, shorter histories and activity
selection matter more than a random train/test split would reveal.

Columns are `rowid`, `event_id`, `user_id`, `item_id`, `timestamp`,
`time_bucket`, `day_type`, `history_item_ids`, `history_timestamps`, plus
training-only `clicked`. Use CSV and JSON parsers. Do not derive features
from random row/event IDs or numerical interpretations of opaque IDs.

**Metric:** global row-level ROC AUC, not an average of user/event AUCs.
Correct cross-event ranking matters. Event probabilities need not sum to
one. Submit exactly `rowid,predicted_probability`, one finite probability in
[0, 1] for every evaluation row. Public/private rows are randomly partitioned;
final ranking uses private AUC. The stated allowance is ten uploads per team
per day. See the actual
[competition instructions](</Users/mihailharetonenko/Hackathons/Huwawei Tech Arena/INSTRUCTIONS.md>)
before changing the task assumptions.

## 3. What has actually been run

All local scores in this table use the same January 26 first-event-per-user
confirmation rows. The official column contains only results reported for
our own submissions, not the external authors' results.

| Our implementation | Local confirmation AUC | Our reported official AUC |
|---|---:|---:|
| Original starter | 0.614071 | Unknown |
| Strategy 1: learned user/history factorization | 0.706187 | Unknown |
| FuxiCTR DIN | 0.672707 | Unknown |
| FuxiCTR DeepFM | 0.712022 | Unknown |
| FuxiCTR DCNv2 | 0.720385 | Unknown |
| Strategy 2: behavioral CatBoost + old MF | 0.726809 | Unknown |
| Revised shared-item history MF, ten seeds, alone | 0.802552 | Unknown |
| Revised history MF + CatBoost | 0.822252 | Unknown |
| **Revised history MF + LightGBM** | **0.822508** | **0.816323** |

DIN/DeepFM/DCNv2 were downloaded and actually trained using upstream FuxiCTR
classes, pinned to commit `183da1c944ba8930fcdcd54114837c081959d748`.
There were four development trials per architecture, including spectral
initialization. This was a bounded benchmark, not proof that those model
families cannot perform better with the new historical-click supervision.

Strategy 1 had separate candidate/history embeddings, learned user preference
vectors, and spectral initialization. It overfit strongly. Strategy 2 added
66 numeric behavioral/embedding features and two native categorical IDs.
The large latest gain came from changing representation and supervision,
not simply substituting LightGBM for CatBoost.

The revised **CatBoost** was selected on development before confirmation;
its local AUC is 0.8222522602992265, log loss 0.19084769502859142. LightGBM
was a near-tied comparison and happened to score 0.000256 higher on
confirmation. Preserve the historical selection record. For future official
comparisons, the LightGBM file with the known 0.816323 result is the anchor.

The primary CatBoost file remains
`artifacts/history_upgrade/submission_history_upgrade.csv`, SHA-256
`e5540cf95fb11130d350a854ffb96e9576c267d3970b0288b42d013e4be1a022`.
Its official score is unknown; do not assign 0.816323 to both files.

## 4. Current model and temporal protocol

### Representation and objective

The main implementation is
[history_upgrade.py](</Users/mihailharetonenko/Hackathons/Huwawei Tech Arena/history_upgrade.py>),
with trees fitted by
[history_upgrade_trees.py](</Users/mihailharetonenko/Hackathons/Huwawei Tech Arena/history_upgrade_trees.py>).
For one model run:

```text
score(user, item) = intercept + scalar_user_bias + scalar_item_bias
                   + dot(mean(shared_item_vector[h] for h in history),
                         shared_item_vector[item])
```

There is no learned user preference vector. A single 32-dimensional item
table serves both history and candidate roles. Empty histories give a zero
history vector. A fitting click log combines supplied fitting histories and
clicked fitting candidates, deduplicated by `(user, item, timestamp)`;
repeat clicks at different times remain distinct. Inference reconstructs the
latest 100 clicks strictly before each event, merging the query's own
supplied history. It never inserts the query's target click into that history.

The loss combines binary cross-entropy on the ten real impressions with an
auxiliary sampled classification loss over historical clicks. The auxiliary
target is one real click versus nine alternatives sampled from fitting-period
item impression frequencies. Sampled copies of the positive item are masked.
Other clicks of that user supply context, excluding the exact target
occurrence; context may be later than the auxiliary target but must still
precede the prediction-day fitting cutoff. This is representation learning
from information already available before the scored day, not a causal
next-click objective. Auxiliary negatives are sampled alternatives, not
observed non-click impressions.

Current recipe: ten independent parameter sets, 32 dimensions, initial
embedding standard deviation 0.05, Adam learning rate 0.003, regularization
coefficient 0.05 with the implementation's normalization, auxiliary weight
0.1, 5,000 updates, and 820 complete events per update per seed. Events are
independently sampled with replacement. Our sampling differs from the
reference's permuted passes. Independent seeds are batched on MPS; forward
values and gradients were checked against independently run single models.
Average score components, never embedding coordinates across seeds.

The tree input is 70 numeric features: 54 retained behavioral/spectral/
collaborative features plus 16 new MF, event-relative, seed-disagreement, and
extended-history features. All twelve old learned-MF features were removed.
Neither current tree learner receives raw user/item IDs. Seed disagreement
and candidate repeat counts are already included; they are not new ideas.

Final LightGBM settings: binary objective, 31 leaves, 272 trees, learning
rate 0.05, minimum leaf size 200, L2 penalty 1, row/feature fractions 0.8,
bagging frequency 1, seed 42, deterministic CPU fitting, column-wise
histograms, twelve threads. Tree counts came from development early stopping.
CatBoost uses depth 6, 1,000 trees, learning rate 0.05, L2 20, 128 numeric
borders, 0.8 row/0.85 feature subsampling, CPU fitting, and time ordering.

### Folds and out-of-time tree features

Day 0 is January 20, 2025, UTC. Each training day's MF features come from
models fitted strictly before that day; outcome counters are day-lagged too.
The first two days are a tree burn-in period, but still fit MF/counters.

| Stage | MF fitting data | Tree fitting events | Scoring events |
|---|---|---|---|
| Development | January 20–23 | January 22–23, 195,814 events | First event/user in January 24–25, 43,955 events |
| Confirmation | January 20–25 | January 22–25, 391,856 events | First event/user on January 26, 35,138 events |
| Final submission | January 20–26 | January 22–26, 489,807 events | All 52,939 evaluation events |

Both development scoring days use the January 24 frozen model/statistics.
Confirmation has 351,380 rows and 22,266 positives. Its indices and labels
were checked against earlier baselines exactly. Final MF fits 686,824
events and 1,009,701 deduplicated clicks; reconstructed fitting histories
average 26.3 clicks. Final trees use 4,898,070 rows.

Ten-seed development MF AUC was 0.778515 versus 0.757883 for one seed.
Development tree AUCs: CatBoost depth 6, 0.798398704; CatBoost depth 8,
0.798323350; LightGBM 31 leaves, 0.798360918; LightGBM 63 leaves,
0.798342044. These tiny learner differences are weak selection evidence.
Settings were locked before confirmation in `selection_locked.json`.

**January 26 is a reused comparison window, not an untouched test set.** It
was examined in previous development and in the external repo's tuning.
Never describe later experiments on it as independent final evaluation.
Official evaluation labels remain unavailable locally.

### Diagnostics already available

For the selected CatBoost, approximate whole-user/event bootstrap AUC 95%
interval: 0.819647–0.824864; improvement over strategy 2: +0.095443, paired
interval +0.092201 to +0.098099. These use 100 resamples and describe
conditional sampling uncertainty, not seed or temporal uncertainty.

| CatBoost confirmation slice | Events | AUC |
|---|---:|---:|
| Known users | 34,543 | 0.823405 |
| Users without fitting events | 595 | 0.744032 |
| Empty supplied history | 198 | 0.544556 |
| Supplied history length 1–9 | 7,446 | 0.776727 |
| Supplied history length 10–20 | 27,494 | 0.833390 |

These slice numbers are **CatBoost**, not LightGBM, and use supplied history
length rather than reconstructed length. Measure LightGBM slices and rare-item
support before prioritizing a specialist. CatBoost's largest importances were
history dot product 33.05%, MF score 16.94%, and MF softmax share 14.08%.
These are importance measurements, not isolated causal ablations.

## 5. Findings from the external repository

Studied clone: `external/techarena2026-atu`, pinned at
`1ed9fd299e4a04352279272bd6be02ed1d08ceec`, from
[jancastachova/techarena2026-atu](https://github.com/jancastachova/techarena2026-atu/tree/1ed9fd299e4a04352279272bd6be02ed1d08ceec).
The clone is ignored by our parent Git repository and was left unchanged.
Our numerical implementation is separate; the upstream pipeline was studied,
not retrained as-is. See our detailed
[source study](</Users/mihailharetonenko/Hackathons/Huwawei Tech Arena/analysis/REFERENCE_SOLUTION_STUDY.md>).
That study predates implementation of our upgrade; its suggested main changes
have now been completed.

Their [detailed experiment/upload log](https://github.com/jancastachova/techarena2026-atu/blob/1ed9fd299e4a04352279272bd6be02ed1d08ceec/experiments/RESULTS.md)
reports these public AUCs; these are author-reported, not independently
reproduced or scores of our code:

| Reference experiment | Change | Reported public AUC |
|---|---|---:|
| 03 | Initial MF + LightGBM | 0.727931 |
| 04 | Replace learned user preferences with pooled history | 0.766866 |
| 05 | Add historical-click supervision | 0.777089 |
| 06 | Average five MF seeds | 0.795989 |
| 07 | Average ten MF seeds | 0.798027 |
| 09 | Reconstruct longer histories | 0.804933 |
| 10 | Other known clicks as auxiliary context | 0.812971 |
| 11 | Share item tables | 0.815842 |

Our current upgrade implements the main ideas through experiment 11. Do not
present those same changes as still pending. Sequential upstream gains
interact, and are not promises of additive improvements here.

Important remaining differences:

1. Their self-contained
   [experiment 13](https://github.com/jancastachova/techarena2026-atu/blob/1ed9fd299e4a04352279272bd6be02ed1d08ceec/experiments/13_retune_7days/retune.py)
   uses **3,000 updates, fifteen seeds, and initialization std 0.005**.
   Our baseline still uses 5,000 updates, ten seeds, and std 0.05. Their
   six-day-fit step sweep favored 3,000 over 5,000 updates; this is a
   concrete next experiment, not a generic call to train longer.
2. Their fitting click log also recovers pre-cutoff clicks from the supplied
   held-out/evaluation histories. Ours learns representations only from
   fitting histories/labels, using each query's history only at inference.
   The size and value of the additional unique click set are **not measured
   in our current pipeline**. Audit it before assuming a large gain.
3. Their tree matrix has 39 features versus our 70. Their tree iteration
   count is selected on the last fitting day, scaled for the added data,
   then refit. Ours uses fixed counts chosen on the earlier development
   window. Neither policy is automatically preferable.
4. Their default local cutoff is January 25; ours is January 26. Some later
   reference results are fitting-period diagnostics, not full validations.
   Do not compare their local 0.83571 directly with our local 0.822508.

Their [simplified notes](https://github.com/jancastachova/techarena2026-atu/blob/1ed9fd299e4a04352279272bd6be02ed1d08ceec/experiments/resultsSIMPLE.md)
claim approximately 0.818 public AUC for experiment 13 and a 0.83571 **local**
baseline. The detailed upload log stops at experiment 11, so the newer claims
have weaker accompanying provenance. Experiment 15 adds fifteen new seeds and
averages submissions, effectively extending ensemble diversity; it does not
establish a verified 0.83 official result.

Read experiment 13 for a self-contained reference. The earlier implementation
chains imports and runtime replacements across experiments 04/05/09/10/11.
Some help examples mention `12_standalone/standalone.py`, absent at this pinned
commit. No license file was found during the review; preserve attribution and
check licensing before redistributing upstream source.

## 6. Ideas already tried, or evidence that lowers their priority

| Idea | Evidence/status | Consequence |
|---|---|---|
| More learned user preference vectors | Our old MF overfit; reference removal gave a large gain; regularized residual vectors also failed there | Keep scalar biases; infer preferences from history unless a new constrained hypothesis warrants a test |
| More epochs/updates by default | Our earlier MF overfit; reference longer runs degraded, including 8,000+ updates | Study validation step curves and ensemble efficiency |
| Larger item dimension | Reference 64 dimensions often worse than 32 | A small controlled check, not the first major experiment |
| Strong recency, sequence order, learned position weights | Mostly failed in the reference; several overfit | Current evidence favors pooled taste; require temporal residual evidence before sequence models |
| Attention/Transformers | No clear reference gain; our bounded DIN benchmark weak | Only revisit with the stronger current data/objective and a specific hypothesis |
| Choice-only loss or event-group ranking | Reference choice MF improved but final trees/public did not; LambdaRank/rank_xendcg worse | Preserve global row-level objective and event propensity |
| Normalize predictions within each event | Our strategy-2 development sweep of log-sum-exp adjustment chose zero adjustment; no gain | Do not promise improvement by forcing sums to one |
| Forty-nine auxiliary negatives instead of nine | Reference worse | Changing negative distribution/false-negative handling is a distinct hypothesis; more negatives alone is not |
| Stronger regularization selected from one seed | Several apparent reference gains disappeared at three seeds | Confirm under an ensemble and the intended fit length |
| Shared co-click spectral initialization | Reference one-seed gain disappeared under averaging; shared starts reduce diversity | A new graph score may add information; identical initialization is not established progress |
| Candidate position, shown-before-without-click, other candidates added to user preference | Reference flat/noisy or worse | Low priority; random candidate ordering has no reliable signal |
| More tree tuning, extra old MF features, tiny blends | Usually around 0.0002–0.0003 upstream | Require repeatable out-of-time gains before spending uploads |
| Empty-history fallback alone | Weak slice, but only 1.256% of evaluation events; baseline already has item/activity features | Worth a small targeted test, not a credible sole route to +0.014 official AUC |

The reference authors infer that the simulator largely uses an unordered
preference direction. Treat that as an empirical working hypothesis, not
access to the unknown data generator.

## 7. Technical improvements: strengthen the existing approach

These improve experiment reliability, fitting, or use of existing signals.
They do not assume access to new ground-truth labels. Priorities indicate
recommended order, not guaranteed score gains.

### T1 — Versioned experiments and trustworthy caches — prerequisite

Implement a run configuration and `--output-dir`/run ID shared by MF and tree
workers. Separate prepare, MF fit, score-components, tree fit, predict, and
export stages. Preserve the original run unchanged. Export schema, feature
order, code revision or source hashes, dependencies, data hashes, cutoffs,
seed IDs, and full hyperparameters with every run.

Current specific hazards:

- Both upgrade scripts hard-code `artifacts/history_upgrade`. Different
  recipes collide with existing outputs; some paths reject mismatches,
  while others rebuild under the same names.
- `strategy1.cache_fingerprint()` hashes training arrays and ID maps, but
  **not evaluation array contents**. Feature/cache identities do not fully
  bind source code, base-feature contents, query availability policy, and
  all input arrays. Do not assume a cache is correct just because it loads.
- The new feature builder depends on strategy-2 feature caches, including
  an expensive old-MF preparation path whose twelve outputs it discards.
  Extract a reusable behavior-only builder. Compare its retained feature
  arrays with the baseline before accepting the refactor.
- Re-running the current exporter regenerates `results.json` and can reset
  official-score fields. `analysis/official_scores.json` is the durable
  source of truth; reattach a score only when the exported CSV hash matches.

Add focused checks for these contracts. Do not change model math while
refactoring experiment plumbing. A predict-only run should reproduce the
saved baseline probabilities before any new modeling claim.

### T2 — Better validation and error analysis — highest priority

Add expanding chronological cutoffs, for example January 24, 25, and 26,
with first-event-per-user scoring. These folds overlap and are correlated;
do not treat them as independent trials. Keep a fixed query window and a
frozen fitting cutoff per fold. Distinguish short one-day diagnostic folds
from first-event-over-all-remaining-days folds, which resemble evaluation
more closely. Report both rather than silently changing the baseline split.

Use earlier windows for configuration selection and treat January 26 as
reused confirmation. There is no unused labelled day beyond it. Freeze a
small experiment plan before inspecting results and reserve official
uploads for finalists. Respect the fact that public scores are themselves
development feedback.

Measure pooled AUC/log loss plus slices by supplied/reconstructed history
length, history age, training-user support, candidate click/impression
support, repeats, event date/time bucket, and whether an event has any click.
Count positives/negatives and undefined-AUC slices. For candidate rarity,
derive support from fitting data only. Evaluate cross-slice ranking as well
as within-slice AUC; global AUC includes pairs from different slices.

Use paired user/event resampling for changes on identical rows and multiple
training seeds for optimization variability. A 100-resample interval is
coarse for tiny gains; increase resamples only for finalists. Do not use
random candidate-row splits, which share users, events, and future signals.

### T3 — Reproduce the reference's newer convergence recipe — high priority

Run a small, explicit factorial comparison after T1:

| Run | Updates | Init std | Seeds | Question |
|---|---:|---:|---:|---|
| Preserved baseline | 5,000 | 0.05 | 10 | Existing anchor |
| Shorter fit | 3,000 | 0.05 | 10 | Does stopping earlier generalize better? |
| Smaller initialization | 5,000 | 0.005 | 10 | Does the smaller start improve averaged representations? |
| Combined | 3,000 | 0.005 | 10 | Do the changes interact? |
| Budget comparison | 3,000 | 0.005 | 15 | Is compute better spent on more independent seeds? |

Fifteen × 3,000 is 45,000 seed-updates versus the baseline's 50,000;
actual runtime/memory need measurement. Thirty seeds, as suggested by the
reference's later averaging, are a follow-up only if ensemble learning
curves still improve. Evaluate the full tree pipeline, not MF alone. Keep
base features, data policy, and tree search budget fixed for causal comparison.
Pilot with a few seeds to reject failures, but do not select a final setting
from a single lucky seed. Do not use published January 26 tuning as evidence
of an untouched confirmation result.

### T4 — Tree fitting policy, feature ablations, and modest tuning

Compare the existing fixed iteration counts with an inner last-fitting-day
early-stop/refit policy. Any scaling of iterations after adding data must be
declared and tested, not assumed valid. Keep outer confirmation labels out
of iteration selection. Inspect whether trees overfit MF quality differences
across early training days before introducing day weights or source-day
features; reference tests found little benefit from these.

Ablate feature groups: 70-feature baseline, a compact reference-like set,
old spectral/co-click features removed, and extended-history features removed.
Keep the upgraded MF fixed. Then test a bounded leaf/min-leaf/L2 search;
there is no evidence that a huge search is a better use of time than data
and representation experiments. LightGBM's own
[tuning guide](https://lightgbm.readthedocs.io/en/v4.5.0/Parameters-Tuning.html)
identifies leaf count and minimum leaf size as key complexity controls.

Global monotonic probability calibration does not improve AUC by itself.
Calibration matters for log loss and for mixing different models; segment-
specific corrections can change rankings but require honest out-of-time
selection. Avoid arbitrary rebalancing of the observed click rate.

### T5 — Ensembles with measured diversity

Use paired out-of-time predictions to test a few fixed CatBoost/LightGBM
blend weights, then freeze the chosen rule. Select weights on development
only, not January 26. Compare averaging probabilities versus averaging
logits; neither is automatically optimal. The two current tree models share
nearly all information, so the plausible gain is limited unless their errors
are complementary. Diverse MF recipes or a genuinely different graph score
may be more useful than more near-identical tree seeds.

Never fit a second-stage stacker on in-sample predictions. Build its training
features from chronological out-of-time predictions. Compare gain per unit
compute and record how many effective independent models are being added.

### T6 — Performance and reproducibility without changing predictions

Reuse saved per-seed components for prefix ensemble tests and cached features
for tree-only tests. Keep memory mapping and chunked CSV handling. Log
training/auxiliary losses separately, validation curves, history/support
coverage, elapsed time, and peak memory. Check numerical equivalence before
mixed precision or changed vectorization; MPS and CPU need not be bitwise
identical. Keep resumable optimizer/RNG state and atomic saves.

**Keep tree fitting in a Torch-free child process on this Mac.** The project
uses the OpenMP runtime bundled with the PyTorch wheel via the child's
`DYLD_LIBRARY_PATH`; importing Torch in that worker defeats the isolation.
Cross-platform dependency packaging should make this conditional and avoid
hard-coded paths to someone else's environment.

## 8. Principal experiments: change the data or learning assumptions

These are distinct scientific hypotheses. Each needs a separately versioned
dataset/availability policy, controlled baseline, and out-of-time comparison.
None is implemented merely because it appears below.

### P1 — Recover additional real historical clicks — highest data priority

The main unclosed reference difference is its use of pre-cutoff clicks from
the **supplied histories of held-out/evaluation queries** during representation
fitting. Those are observed historical positives, not the hidden evaluation
candidate labels. Audit two click sets for each fold:

```text
A: histories of fitting events + clicked fitting candidates (current)
B: A + allowable pre-cutoff clicks from the supplied scoring-query histories
deduplication key: (user, item, timestamp)
```

Count new unique clicks, new users/items, support gains for rare items, and
changes to reconstructed histories. Test B first as an auxiliary-only
training extension, keeping impression denominators unchanged. Extend graph
or long-history fitting separately so that effects can be attributed.

The availability contract is essential: use only the exact query histories
available in that evaluation scenario; filter **every click and auxiliary
context** before the MF fitting cutoff; never use held-out `clicked` values
or histories from later discarded events to reconstruct scored outcomes.
For daily out-of-time features, apply that day's cutoff, not the final
January 27 cutoff. Own-history inference must still exclude current/future
clicks relative to the individual event.

Timestamp filtering alone does not establish when a record would have been
available in a live system. Distinguish competition batch/transductive access
to all supplied inputs from per-user online prediction. Confirm the
competition's allowed use of supplied unlabelled inputs and mirror the same
policy locally; do not label an inconsistent transductive comparison as a
strictly inductive gain. Our current code deliberately takes the stricter
approach. If B adds few unique clicks, deprioritize it on that evidence.

Do not fabricate exposure counts for warm-up clicks or classify every
unobserved item as a factual negative. Historical counts and impression CTRs
must remain separate quantities.

### P2 — More useful training views, especially for sparse histories

The current auxiliary objective already uses other clicks and up to 100
context items. Test several **views of those same known clicks**: random
context subsets, context dropout, and length caps such as 5/10/20/50/100.
Keep the exact target occurrence out of its own context. This could make
the model more robust to shorter evaluation histories without inventing
new ground truth. It is a hypothesis, not a proven augmentation benefit.

Compare event-uniform versus user-balanced auxiliary sampling, with explicit
weights so highly active users or duplicate contexts do not silently dominate.
Distinguish actual additional click records from repeated views of the same
positive. Keep provenance and effective sample weights. If testing causal
next-click views, use only earlier context; do not confuse that objective
with the current allowed other-click representation objective.

### P3 — Better negatives and exposure-aware supervision

Baseline alternatives follow impression popularity, which is important
because candidate exposure is not uniform. Compare small mixtures of that
distribution with uniform or smoothed-popularity sampling, and separately
test hard negatives from real earlier candidate sets. Mask positive-item
collisions and examine false negatives among other known liked items.

Changing the sampler or number of alternatives changes the objective and
the scale of auxiliary learning. Derive any sampling correction rather than
blindly subtracting log-popularity, and retest the auxiliary weight under
the new normalization. Actual non-clicked impressions carry different
evidence from unobserved sampled items. Forty-nine alternatives already
failed upstream; nine with better-controlled semantics is a more defensible
first test than merely increasing K.

### P4 — Rare-item sharing through a second collaborative representation

First verify our own performance by fitting-period candidate support. If
rare-item errors dominate, build a sparse item-neighbor score or a graph
representation from allowable pre-cutoff clicks. Add its scores to the
trees instead of replacing the strong baseline immediately. Use shrinkage
toward item/global priors or regularized collaborative groups so rare items
borrow strength without memorizing arbitrary IDs.

Candidate approaches include top-k co-click similarity, a regularized linear
item model inspired by [EASE](https://arxiv.org/abs/1905.03375), or a small
[LightGCN](https://arxiv.org/abs/2002.02126) using neighborhood aggregation.
These papers motivate distinct collaborative signals; they do not establish
that the methods will beat our model here. Existing spectral/co-click
features and failed shared PPMI initialization make an explicit incremental
ablation necessary. A separate prediction signal is a different test from
initializing every ensemble seed identically.

Budget sparse implementations first. With 50,000 items, one dense 50,000²
float32 matrix alone takes 10 GB decimal, before inverses and workspaces;
float64 doubles that. Do not blindly run a dense EASE inverse on this laptop.
All graph edges and fitting statistics must follow the same temporal and
query-history policy as the corresponding MF feature.

### P5 — Better pooling, cold-user inference, and support shrinkage

Test frequency versus unique-item pooling, bounded repeat weights,
popularity-adjusted pooling, and shrinkage of short-history vectors toward
a learned low-capacity prior. Consider several fixed context caps before
complex attention. Scalar history length and repeats are already tree
features, so require evidence that changing the representation adds something.
Strong time decay has poor prior evidence here.

For users without fitting events, infer preferences from their supplied
history without a memorized user vector. For empty histories, compare a
small item-popularity/context fallback or a regularized mixture whose gate
uses only available inputs. Fit all priors and gates out of time. The
baseline already contains popularity and activity information; a fallback
is not automatically new signal. Evaluate global ranking and interactions
with other slices, not only the specialist's within-slice score.

### P6 — Align fitting with the first-event evaluation population

All training events overweight active users relative to one-query-per-user
evaluation. Evaluation also has substantially older histories and different
time-bucket frequencies. Compare user-balanced event weights or repeated
pseudo-cutoff training views that select the first subsequent event per user.
Vary the pseudo-cutoff to avoid throwing most supervised examples away.

Use label-free support/history covariates to diagnose distribution mismatch.
If testing clipped importance weights from a train-versus-evaluation
discriminator, fit and validate the procedure within chronological local
folds and report sensitivity; it depends on covariate-shift assumptions and
can increase variance. Do not treat evaluation membership or timestamp as
a target proxy. AUC on a weighted validation population is a diagnostic,
not a substitute for reporting the standard unweighted competition metric.

### P7 — Event propensity and conditional item choice — lower priority

A principled alternative is
`P(click item i) = P(any click in event) × P(i | event gets a click)`.
Retain no-click events when learning the first factor. The second factor
can model competition among the ten candidates, while the first preserves
cross-event ranking. Test only if error analysis finds event-propensity
information the current tree is missing.

This overlaps with already unsuccessful reference choice objectives and
our normalization test. A genuinely separate, regularized propensity model
is a new hypothesis; simply forcing candidate probabilities to sum to one
or optimizing event ranking alone is not. Judge the final global row AUC,
not only conditional choice accuracy.

### P8 — Distillation, alternative sequence models, or external data — exploratory

If the above plateaus, a diverse out-of-time teacher ensemble could provide
soft targets for a smaller or differently structured student. Keep real
and pseudo-labelled targets separate and weighted; a teacher's confident
prediction is not a new observed label. Do not validate against labels
already used to fit the teacher for those rows. Test whether there is any
new independent ranking information before spending substantial compute.

SASRec/attention or a reworked DIN/DeepFM/DCNv2 could be retried with the
current historical-click pretraining and data, but only after residual
analysis justifies extra capacity. Their names alone do not supply a reason
for improvement. Sequence modeling has weak supporting evidence here.

Do not join unrelated public recommendation datasets through opaque item
IDs or invent semantic metadata with an LLM. External data is useful only
if there is a legitimate, allowed mapping to these exact synthetic entities
or an explicitly justified pretraining transfer experiment. No such mapping
or metadata source is currently known.

## 9. Recommended next sequence and acceptance criteria

1. **Preserve and verify the official baseline; implement T1.** Reproduce
   saved predictions without retraining, add run directories/configuration,
   and establish data/feature contracts. Confirm the official record still
   references the original SHA, not a newly exported variant.
2. **Run T2 diagnostics and the bounded T3 comparison.** Establish whether
   shorter fitting/smaller initialization help the full pipeline across
   existing chronological windows under averaging. Keep one change per
   ablation before combining them.
3. **Audit and test P1.** If there are useful additional real historical
   clicks, this directly addresses a known implementation difference.
   Record the information-access policy as part of the experiment identity.
4. **Test P2/P5 for sparse histories and T5 for a cheap blend.** Use measured
   residuals and error correlation to choose the order. Small tree-only
   changes are cheaper than rebuilding six MF ensembles.
5. **Try P4 if rare-item analysis supports it, then P3/P6.** Reserve P7/P8
   for clear remaining hypotheses. Do not launch all architectures at once.

For promotion, require a repeatable gain on the final tree output, matched
validation rows, no time/label leakage, and an acceptable compute cost.
A practical submission target is at least roughly +0.001 local AUC with
consistent fold/seed behavior; this is a budget heuristic, not a significance
test or a guarantee of official improvement. A gain of +0.0002 on one reused
fold is insufficient evidence by itself. Check slice regressions, but do not
expect every slice to improve when pooled AUC does.

Freeze the finalist, refit using all seven training days, reload checkpoints,
validate the complete CSV, and give the user the file to upload. Do not claim
an official score until the user supplies the competition result. Record the
new score against the exact submitted checksum. Keep the original 0.816323
entry even if a later model is worse. Do not fit a straight-line learning
curve from public submissions or promise a 0.83 outcome.

Suggested experiment record:

```yaml
experiment_id: descriptive_unique_id
parent_run: history_upgrade_baseline
hypothesis: one sentence
status: proposed | running | completed | rejected
code_revision_or_source_sha256: ...
data_and_query_sha256: ...
historical_input_policy: fitting_only | explicitly_defined_batch_policy
cutoffs_and_query_selection: ...
mf: {seed_ids: [...], steps: ..., init_std: ..., dim: ..., aux_weight: ...}
sampling_and_context_policy: ...
feature_schema_hash: ...
tree_parameters_and_selection_window: ...
local_metrics_by_fold_and_seed: ...
paired_delta_and_uncertainty: ...
slice_metrics_with_support: ...
runtime_and_memory: ...
submission_file_and_sha256: ...
official_score_source_date_partition_and_submission_id: null
decision_and_reason: ...
```

## 10. Files, environment, and reproduction

### Source and reports to transfer

| Files | Purpose |
|---|---|
| `INSTRUCTIONS.md`, `README.md`, this file | Task contract and continuation context |
| `analysis/official_scores.json` | Durable official result with CSV identity |
| `history_upgrade.py`, `history_upgrade_trees.py`, `requirements-history-upgrade.txt` | Current best pipeline |
| `strategy1.py`, `strategy2.py`, their requirements files | Shared helpers and current feature-cache dependency |
| `test_strategy1.py`, `test_strategy2.py`, `test_history_upgrade.py` | Twenty existing unit checks |
| `analysis/profile_data.py`, `analysis/study.py`, `analysis/spectral_probe.py` | Parsing, dataset study, spectral exploration |
| `analysis/*.json`, `analysis/STRATEGY.md`, `analysis/REFERENCE_SOLUTION_STUDY.md` | Profiles, audits, original plan, reference study |
| `HISTORY_UPGRADE_RESULTS.md`, `STRATEGY1_RESULTS.md`, `STRATEGY2_RESULTS.md`, `CTR_BENCHMARK_RESULTS.md` | Detailed experiment provenance |
| `benchmark_ctr.py`, `requirements-ctr-benchmark.txt` | Actual upstream model benchmark |

Preserve existing local work, including `submission_strategy1.csv.zip`.
Git ignores raw datasets, `.venv`, analysis caches, most artifacts, and the
external clone. New `.py`/report files being untracked is separate from large
artifacts being intentionally ignored. Do not assume `git push` would transfer
the datasets or models. Check credentials and local-only files before any
future commit; do not commit secrets.

Raw CSVs can be obtained from the project's
[dataset-v1 release](https://github.com/HM30082005/MCMF_Huawei-TechArena/releases/tag/dataset-v1).
Download all three `.csv.gz` assets into the project root and decompress.
The release includes original/compressed checksums. The original CSVs total
approximately 7.7 GB. Preserve row-ID alignment exactly.

### Useful local artifacts

```text
analysis/.cache/                         encoded mmap arrays and ID maps
artifacts/strategy1/                     earlier MF checkpoints/predictions
artifacts/strategy2/features/            required 66-feature base caches
artifacts/strategy2/                     earlier tree and confirmation baseline
artifacts/ctr_benchmark/                 upstream code, benchmark models/CSVs
artifacts/history_upgrade/
  recipe.json                           actual MF recipe
  seed_selection.json                   development seed-prefix scores
  selection_locked.json                 settings before confirmation
  results.json                          full local results and export metadata
  verification.json                     execution/checkpoint/CSV verification
  models/day2.pt ... day7.pt             six fitting cutoffs, batched seeds
  parts/*.npy + *.json                   per-seed MF components and provenance
  features/*.npy + *.json                numeric features, names, cutoffs
  trees/final_LightGBM_31.txt            full-data official-scored model
  trees/final_CatBoost_6.cbm             full-data development-selected model
  confirmation_LightGBM.npz              indices, labels, raw scores
  confirmation_CatBoost.npz              indices, labels, raw scores
  confirmation_predictions.npz           selected CatBoost alias
  evaluation_LightGBM_logits.npy         full-evaluation raw scores
  evaluation_CatBoost_logits.npy         full-evaluation raw scores
  evaluation_logits.npy                  selected CatBoost alias
  submission_LightGBM.csv                0.816323 official, user reported
  submission_history_upgrade.csv         CatBoost, official unknown
artifacts/history_upgrade_run.log
artifacts/history_upgrade_tests.log
```

Raw-score versus probability conventions differ across older experiments:
the FuxiCTR confirmation NPZ files contain probabilities; the revised tree
files contain logits/raw scores. Inspect keys and generating code before
blending. Never apply sigmoid twice. AUC is invariant to sigmoid on a single
score vector, but averaging/calibration is not invariant to this choice.

The encoded arrays have event-major shapes: items/labels `(N, 10)`, supplied
history `(N, 20)`, and aligned timestamps. Existing encoded IDs are zero-based;
history padding is -1. The upgraded MF adds one to item IDs and uses zero
padding. User IDs remain zero-based. Preserve these conventions, the mapping,
and stable event/row order. The parser creates a zero-filled evaluation label
array for structural convenience; loaders exclude it. Those are **not real
evaluation labels** and must never be scored or fitted as ground truth.

Current disk sizes: encoded cache about 220 MB, upgrade artifacts about
6.1 GB, strategy-2 feature caches about 3.7 GB, reference clone about 419 MB.
Reserve room for run copies. Transfer reusable artifacts when possible;
recreate the environment instead of copying a machine-specific `.venv`.

### Environment and commands

Original hardware is Apple M5 Pro; GPU training uses PyTorch MPS, trees CPU.
Recorded environment includes Python 3.12.15, PyTorch 2.15.0, CatBoost 1.2.10,
LightGBM 4.7.0, NumPy 2.5.4, SciPy 1.18.1, and scikit-learn 1.9.1. Use
the requirements files as the starting point and record the actual recreated
environment. The installed LightGBM relies on `libomp.dylib` bundled in the
PyTorch wheel; the main script sets up its child process. Launching the tree
worker directly without that environment may fail on macOS.

```sh
cd '/Users/mihailharetonenko/Hackathons/Huwawei Tech Arena'
# If no environment exists, create a Python 3.12 environment first:
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-history-upgrade.txt

# Run only if the validated encoded data cache is absent:
.venv/bin/python analysis/profile_data.py

# Current dependency: needed only if strategy-2 feature caches are absent.
# This also prepares old MF outputs; T1 should remove that unnecessary cost.
OPENBLAS_NUM_THREADS=2 .venv/bin/python strategy2.py prepare --device auto

.venv/bin/python -m unittest test_strategy1 test_strategy2 test_history_upgrade -v

# Existing baseline recipe. Preserve its directory first; do not use this
# fixed-directory command for modified experiments until T1 is implemented.
OPENBLAS_NUM_THREADS=2 .venv/bin/python history_upgrade.py --device auto
```

Skip environment creation when `.venv` already exists. Full baseline training
took approximately 25 minutes after implementation on this machine with
caches available; a 5,000-step ten-seed daily MF fit took about two minutes.
Data preparation, other hardware, more seeds, and new experiment plumbing can
change that substantially. No training is currently left running or waiting.

Twenty unit checks previously passed, including time exclusions, repeat
deduplication, own-history cold users, exact auxiliary target removal,
permutation behavior, independent-seed values/gradients, and bitwise CPU
interruption/resume. CPU reload of a saved MF matched MPS components within
9.54e-7 on 512 varied events. All six daily models passed event/click cutoff
checks. Both tree learners passed Torch-free fitting/reload checks, and
full-data checkpoints reproduced saved evaluation predictions. The CSV
writer checked all 529,390 unique IDs and raw user/item/time alignment, then
reread outputs. These historical checks do not automatically validate a
modified implementation; rerun checks relevant to changes.

### Operating preferences

Read applicable `AGENTS.md` instructions and preserve unrelated working
changes. This is a numerical modeling project and requires no LLM endpoint
for training. If a genuinely useful LLM-assisted component is introduced,
first assess the user's FreeLLMAPI at `http://127.0.0.1:31415/v1` before
proposing direct providers. Keep provider/API keys out of source, logs,
screenshots, and commits. Do not spend money or upload competition files
without the appropriate user instruction. The current handoff task authorizes
documentation and recording a result, not new experiment execution.

## 11. Suggested prompt for the next development session

> Continue this TechArena project using CLAUDE_HANDOFF.md as the current state.
> Our user-reported official baseline is 0.816323 AUC for the exact LightGBM
> CSV with SHA-256 6137599051cfac43a71ec5ed62a96c27deaba4085e0ecaf23252bc46430cb184.
> First inspect the working tree and available artifacts, preserve that run,
> and implement versioned experiment outputs and cache/configuration contracts
> without changing baseline predictions. Then propose a bounded execution
> budget for the shorter-fit/smaller-initialization comparison and audit the
> extra pre-cutoff clicks available from supplied query histories. Keep
> technical improvements separate from principal data/model experiments,
> validate out of time, and report measured results without claiming new
> official scores. Avoid repeating already failed ideas without a new
> hypothesis. Tell me which files or data are missing before assuming a fresh
> GitHub clone contains the current implementation.
