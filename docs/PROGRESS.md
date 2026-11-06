# Progress report

> Historical snapshot. The later [research summary](experiments/research-summary.md)
> records the 0.824787 local blend and corrects earlier experiment interpretations.

Every direction tried so far on the TechArena click-prediction challenge, with
scores and improvements. Last updated 4 November 2026.

| | ROC AUC | Notes |
|---|---:|---|
| **Best official score** | **0.816323** | The only score uploaded and recorded so far |
| Same model on the local Jan 26 check | 0.822508 | Reference point for everything below |
| **Best model since, local Jan 26 check** | **0.824669** | +0.0022 locally; official score unknown until uploaded |

## How to read the scores

The metric is global ROC AUC over all candidate rows. Official uploads are limited,
so models are compared locally on rows from the training week that they did not
train on. Each scored row is a user's first event after a cutoff, as in the real
evaluation set.

- **Official:** the competition's score, known only after an upload.
- **Development (Jan 24–25):** the search window. Every idea is compared here.
- **Jan 26 check:** reserved for ideas that win on development, at most two looks a day.

**Don't compare these three with each other.** They are different rows, so each has
its own scale: the same model scored 0.7984 on development, 0.8225 on Jan 26 and
0.8163 officially. Only gains on identical rows are meaningful. Intervals are 95%
paired bootstrap intervals against the baseline.

For scale, two runs of an identical recipe differ by about 0.0001 on development,
so anything that small is noise. An idea needs at least +0.001 to count.

## Part 1: from the starter to the first official score

Local Jan 26 AUC. Gain is against the best model before it.

| Approach | Jan 26 AUC | Gain | Official |
|---|---:|---:|---:|
| Starter solution | 0.6141 | – | – |
| Factorisation with learned user vectors | 0.7062 | +0.0921 | – |
| FuxiCTR DIN (deep attention model) | 0.6727 | below best | – |
| FuxiCTR DeepFM | 0.7120 | +0.0058 | – |
| FuxiCTR DCNv2 | 0.7204 | +0.0084 | – |
| CatBoost on 66 behavioural features | 0.7268 | +0.0064 | – |
| Shared-item history factorisation, ten seeds, alone | 0.8026 | +0.0757 | – |
| ...plus CatBoost | 0.8223 | +0.0197 | – |
| **...plus LightGBM (the baseline)** | **0.8225** | +0.0003 | **0.816323** |

The large jump came from changing the representation and supervision. It dropped
learned user vectors, inferred a user's taste by averaging their clicked items' vectors
over a longer history, added historical clicks as training signal, and averaged
independently trained models. Most of these ideas came from studying the
[ATU reference solution](https://github.com/jancastachova/techarena2026-atu).

## Part 2: experiments since the baseline

Each row changes one thing from its parent. The parent is the row above unless noted.

### Training recipe for the factorisation model (MF)

| Experiment | Development AUC | Gain vs parent | Gain vs baseline [95% CI] | Jan 26 AUC | Verdict |
|---|---:|---:|---|---:|---|
| Baseline: 5,000 updates, init std 0.05, 10 seeds | 0.798361 | – | – | 0.822508 | Reference |
| 3,000 updates | 0.800252 | +0.0019 | +0.0019 [+0.0014, +0.0023] | – | **Kept** |
| + init std 0.005 | 0.800903 | +0.0007 | +0.0025 [+0.0019, +0.0031] | – | **Kept** |
| + 20 seeds | 0.801910 | +0.0010 | +0.0035 [+0.0029, +0.0041] | 0.824160 | **Kept** |
| + 30 seeds | 0.802381 | +0.0005 | +0.0040 [+0.0034, +0.0046] | 0.824512 | **Kept** |
| + 50 seeds | 0.802354 | −0.0000 | +0.0040 [+0.0035, +0.0045] | – | Rejected: curve has flattened |
| Updates scaled to each fit's size (from baseline) | 0.798377 | +0.0000 | +0.0000 [−0.0007, +0.0007] | – | Rejected: lost the 3,000-update gain |
| Varied updates per seed, 2,500–4,500 (parent: 10-seed row with std 0.005) | 0.800770 | −0.0001 | +0.0024 [+0.0018, +0.0030] | – | Rejected: longer-trained seeds are worse |

MF alone on development improves steadily with averaging, then flattens:
1 seed 0.7729, 5 seeds 0.7822, 10 seeds 0.7839, 20 seeds 0.7847, 30 seeds 0.7850, 50 seeds 0.7852.

### How the trees are trained

| Experiment | Development AUC | Gain vs parent | Gain vs baseline [95% CI] | Jan 26 AUC | Verdict |
|---|---:|---:|---|---:|---|
| **Tree count from the last fitting day, scaled and refit** (parent: 30 seeds) | 0.802224 | −0.0002 | +0.0039 [+0.0033, +0.0045] | **0.824669** | **Kept: current best** |
| Trees on each user's first event per day (parent: 3,000 updates) | 0.799998 | −0.0003 | +0.0016 [+0.0012, +0.0021] | – | Rejected |
| Tree rows weighted towards the scored history-age mix (parent: 3,000 updates) | 0.799796 | −0.0005 | +0.0014 [+0.0010, +0.0019] | – | Rejected |

The refit policy gains only +0.0002 over its parent on Jan 26. We kept it for a
structural reason: the tree count no longer depends on scored rows. Identical recipes
otherwise picked anywhere from 167 to 380 trees by chance. The parent's final model has
167 trees; the refit model has 584.

### More data and features (parent: 30 seeds)

| Experiment | Development AUC | Gain vs parent | Gain vs baseline [95% CI] | Verdict |
|---|---:|---:|---|---|
| Extra clicks from scored queries' own histories | 0.802460 | +0.0001 | +0.0041 [+0.0035, +0.0047] | Rejected: noise level, and only 0.3% more clicks on the final fit |
| Drop extended-history features (67 left) | 0.802460 | +0.0001 | +0.0041 [+0.0035, +0.0047] | No effect |
| Drop spectral features (61 left) | 0.802360 | −0.0000 | +0.0040 [+0.0034, +0.0046] | No effect |
| Compact set: also drop co-click and extended-history (48 left) | 0.802281 | −0.0001 | +0.0039 [+0.0033, +0.0045] | No effect |

### Blending (quick analysis on development)

Fixed 50/50 logit blends of different MF recipes: 3,000 updates + std 0.005 reaches
0.801412, and 20 seeds + 3,000 updates reaches 0.802172, just +0.0003 over 20 seeds
alone. Not worth a second full pipeline so far.

## Runs that don't count

An autonomous research agent (Hermes) ran part of this search. A review on 4 November
found that these runs did not test what they were meant to, so they are excluded above,
and each was redone correctly where it was still worth testing:

| Run | Problem |
|---|---|
| `p1-query-clicks-30` | A bug meant it loaded 0 extra clicks, so it just repeated its parent |
| `t4-no-spectral`, `t4-reference-features` | Brought back 12 old features they were meant to exclude |
| `t4-refit-fixed` | The refit setting had no effect at that stage, and it used an older recipe |
| `t4-refit-durham*`, `t4-refit-policy`, `t3-asymmetric-steps` | Crashed before producing a result |
| CatBoost + LightGBM blend | Measured on the reserved Jan 26 labels, against the rules; still to redo properly |

None of these affected an uploaded submission. The official baseline was verified intact
afterwards: its predictions and file checksum reproduce exactly from the saved models.

## What we've learned

1. **The MF representation drives almost everything.** Removing 22 of 70 tree features
   costs only about 0.0001.
2. **Shorter training and averaging are the reliable gains so far.** Longer MF training
   memorises, and more seeds helped until about 30.
3. **The current recipe has plateaued.** Development stays at 0.8022–0.8025 whatever we
   vary around it. The next gains need new signal, not more tuning.
4. **Stale histories are the clearest weak spot.** The longer since a user's last click,
   the worse the prediction, and the evaluation set is far staler than training:

   | Time since last click | Baseline AUC (development) | Tree training rows | Development rows | Evaluation rows |
   |---|---:|---:|---:|---:|
   | Under 6 hours | 0.8243 | 42% | 6% | 5% |
   | 6–24 hours | 0.8160 | 34% | 33% | 28% |
   | 1–3 days | 0.7969 | 18% | 39% | 34% |
   | 3–7 days | 0.7743 | 5% | 16% | 22% |
   | Over 7 days | 0.7358 | 1% | 5% | 10% |

   This likely explains much of the gap between local and official scores. Rebalancing
   the trees towards stale users didn't help, so the fix has to come from the
   representation, if it can be fixed at all: stale users may simply carry less information.

## What's next

- Train the MF on deliberately stale or shortened histories, so it learns to predict
  from the kind of history the evaluation set has.
- Measure errors on rarely seen items, then decide whether a sparse item-item signal
  (EASE, ItemKNN) is worth building. Existing co-click features add almost nothing,
  so it would need to be genuinely different.
- Try better negative sampling, such as real non-clicked impressions.
- Upload the current best model to see whether the local gain holds officially.
