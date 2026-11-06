# Experiment priorities

Updated November 4, 2026 (Europe/Dublin). This document consolidates the
priorities agreed in the project chat. All entries are planned experiments,
not implemented changes or demonstrated improvements. Later explicit priority
assignments take precedence over earlier suggestions in `HANDOFF.md`.

## Decisions and baseline to preserve

- Keep **LightGBM** as the final learner for now. Revisit model selection later.
- Include no-click events in main training and evaluation.
- Baseline preference model: shared **32-dimensional item vectors**, mean
  pooling, dot-product matching with user/item biases, and **ten independent
  seeds** whose scalar prediction components are averaged before LightGBM.
- Main supervision uses actual displayed candidates and their click labels.
  Auxiliary history supervision uses one clicked target versus **nine sampled
  alternatives**, weighted **0.1**. Alternatives are not factual unclicked
  impressions. The current auxiliary context can contain later-than-target
  clicks, provided they precede the overall fitting cutoff.
- Current tree input: **70 numeric signals**. This is a baseline feature count,
  not a target number to preserve. Some summaries use supplied histories of
  at most 20 clicks; reconstructed histories can contain up to 100 clicks.
- Recorded baseline: local January 26 AUC **0.8225077649**; user-reported
  official AUC **0.816323**, tied to the CSV checksum in
  `results/official_scores.json`. The chat's **0.824783** needs its run,
  prediction file, and validation protocol identified before comparison.

## How to interpret cost estimates

**Implementation effort** below is an unmeasured planning range in focused
engineering hours for one contributor familiar with the code. It includes
coding, relevant correctness checks, and inspecting a first result. It excludes
unattended training, broad tuning, dependency setup, and missing-data recovery.
These estimates are not promises or elapsed-time measurements.

**Experiment cost** describes what must be rerun:

| Code | Work required | How to estimate elapsed time |
|---|---|---|
| F | Build/change features and refit LightGBM; existing embeddings reusable | Measure feature generation and tree fitting separately; current notes do not provide a reliable standalone time. |
| E | Retrain embeddings, regenerate out-of-time features, and refit LightGBM | Historical reference: approximately **25 minutes for the baseline full pipeline**, once prerequisite caches existed, on the original Apple M5 Pro. |
| H | E plus a materially different or more expensive architecture/backward pass | Benchmark a pilot; no measured multiplier is available. Memory pressure can change runtime substantially. |
| B | Blend saved validation predictions/components | Cheap if compatible outputs exist; generating missing members costs F, E, or H per recipe. |

The handoff also records about **two minutes per 5,000-step ten-seed daily
embedding fit**. This is historical hardware-specific evidence, not a timing
for every new architecture. A full local comparison, extra folds, repetitions,
and a final submission refit are separate costs. Do not count the historical
25-minute run once per experiment and assume all those checks are included.

Before scheduling, check available datasets/artifacts, benchmark one unchanged
fold, and measure each stage. Record actual elapsed time, peak memory, disk
usage, hardware, seed count, and number of updates. Estimate a batch as:

`one-time preparation + sum(new stage timings across configurations/folds/repeats)`

Reuse identical stages by complete configuration/data hashes. Shared groundwork
below is paid once; implementation ranges in different rows overlap and should
not simply be added into a project deadline.

## Shared groundwork before comparisons

This is infrastructure required for trustworthy experiments, not a fourth
priority category or a new modeling hypothesis.

| Task | Deliverable | Estimated effort |
|---|---|---|
| Preserve/reproduce baseline and audit available artifacts | Immutable baseline identity and predictions; identify missing data/checkpoints and reconcile scores | 2–6 h if artifacts exist; regeneration adds runtime |
| Isolated configurations and caches | Run IDs/output directories; fingerprints cover source, feature order, all inputs, cutoff, availability policy, dimensions, weights, and seeds | 6–12 h |
| Parameterize dimensions and auxiliary weights | The embedding constructor currently defaults to 32 dimensions; expose actual dimensions and weight coefficients through all stages and identities | 2–4 h shared by several tests |
| Common comparison report | Identical validation rows; AUC deltas, slices, runtime, provenance, and saved predictions | 4–8 h |

Keep refactoring separate from model changes and verify prediction equivalence.
Retain the Mac's Torch-free LightGBM worker arrangement where required.

## Priority 1 — first-priority experiments

Within this tier, execute prerequisites before dependent tests; membership does
not require running every combination of first-priority ideas.

| ID | Change and initial comparison | Implementation effort | Experiment cost |
|---|---|---|---|
| P1-01 | **Earlier-clicks-only historical extension.** Make each known historical click an auxiliary target, with only that user's strictly earlier clicks as context. Example: Sunday A predicts Monday B; Tuesday C cannot help predict B. Compare with current other-click context at weight **0.1**, holding the click pool fixed. | 3–6 h | E; time-indexed preparation also needs measurement |
| P1-02 | **History masking/dropout.** Hide a declared fraction of earlier history clicks while retaining the target. This tests robustness to incomplete histories. Start separately from cropping; declare empty-context handling and augmentation weights. | 2–4 h | E |
| P1-03 | **History cropping.** Retain shorter contiguous portions of the valid earlier history while preserving order/timestamps and the target. Test a small declared set of lengths; compare with unmodified history. | 2–4 h | E per selected setting |
| P1-04 | **64-dimensional vectors.** Increase from 32 to 64 learned numbers per item. First compare fixed settings, then give promising dimensions equal small regularization/duration budgets. Watch for improved training fit but worse later-date AUC. | 1–3 h after shared dimension plumbing | E; larger tables/dot products, total runtime must be measured |
| P1-05 | **Candidate-specific attention.** Weight history differently for each candidate, so relevant past clicks contribute more to its match. Use a small regularized attention network; compare against mean pooling. | 8–16 h | H; attention must process candidate–history relationships |
| P1-06 | **Sequence-aware self-attention.** Encode order and optionally time information to learn changing interests/next-click patterns. Preserve chronological masks and prevent target/future access. This is a larger architecture experiment. | 16–32 h | H; benchmark memory and sequence-length scaling |
| P1-07 | **Learned weighted averaging.** Learn candidate-independent importance weights for prior clicks rather than assigning equal weight. Distinct from candidate-specific attention. | 4–8 h | E/H depending on the weighting network |
| P1-08 | **Recent versus long-term interest.** Add separate candidate-match scores for the last 24 hours, last seven days, and all available permitted history, plus counts. Let LightGBM combine them. Define any cap and empty windows. | 3–6 h | F if embeddings and timestamped click logs are available |
| P1-09 | **Interest change.** Measure how different recent and older history vectors are. Declare the windows and avoid confusing missing history with genuine interest change. | 2–4 h after P1-08 | F |
| P1-10 | **Activity change.** Compare recent click frequency with the user's earlier observed baseline. Include coverage/support because supplied histories are incomplete. | 3–6 h | F; requires a timestamped click aggregation pass |
| P1-11 | **Item momentum.** Add changes in recent versus longer-term click rates or exposure volume, with smoothing/support. Recent rates already exist: test whether explicit trends add value. | 3–6 h | F |
| P1-12 | **Ordered item transitions.** Estimate whether item B tends to follow A and score candidates from the user's last known click. Use earlier fitting sequences, sparse counts, and smoothing; do not treat a truncated history as a complete sequence. | 4–8 h | F plus transition-table construction |
| P1-13 | **Shared-embedding cosine similarity.** Add cosine between a candidate and the pooled history alongside dot product. Calculate within each seed, then average scalar similarities; never average coordinates across seeds. Define zero-vector behavior. Existing spectral cosine features are not this test. | 2–4 h | F if checkpoints are reusable |
| P1-14 | **Longer-history behavioral summaries.** Extend missing recency, gap, and related summaries from supplied history to reconstructed history. Extended length, distinct-item count, and repeat count already exist; avoid duplicate features. | 3–6 h | F |
| P1-15 | **Remove unhelpful feature groups.** Drop one group at a time and refit LightGBM, keeping embeddings fixed. Test whether a smaller feature set improves AUC; evaluate removals on the same basis as additions. | 2–4 h for reusable removal tooling | F per group-removal run |
| P1-16 | **Additional supplied historical clicks — conditional priority test.** Compare the current fitting-only click pool with one also containing permissible pre-cutoff clicks from the exact scoring queries' supplied histories. Audit incremental unique clicks first; initially add them to auxiliary training only. | 3–6 h audit plus 2–4 h integration | Audit pass, then E only if useful extra data exists |

P1-16 has a rules qualification: the supplied `INSTRUCTIONS.md` does not
explicitly prohibit or authorize fitting on evaluation-input histories.
Its compatibility is an interpretation, not organizer confirmation. Official
submission eligibility remains unverified against any additional rules. Mirror
the same batch-input availability policy in validation; exclude held-out target
labels, post-cutoff clicks, and histories from later discarded events. Past
timestamps alone do not prove online availability. This is separate from
P1-01, which first uses the existing fitting-only click pool.

“Add useful signals” is an umbrella for the individually listed feature tests,
not a separate run or permission to promote all future ideas to priority 1.

## Priority 2 — second-priority experiments

| ID | Change and initial comparison | Implementation effort | Experiment cost |
|---|---|---|---|
| P2-01 | **Auxiliary weight 0.3.** Give earlier-only historical training more influence than the initial 0.1 setting; keep other settings fixed. | 0.5–1 h after shared weight plumbing | E |
| P2-02 | **Auxiliary weight 1.0.** Give its average loss equal coefficient to the main loss. This does not imply equal gradient influence or equal share of learning. Compare with 0.1 and 0.3. | 0.5–1 h after shared weight plumbing | E |
| P2-03 | **Gradient-similarity weighting.** Measure whether main and auxiliary task gradients on shared item embeddings point in compatible directions; reduce/suppress conflicting auxiliary updates under an explicitly defined rule. Same priority as the fixed-weight tests. | 8–16 h | H; separate gradient computation adds work |
| P2-04 | **Contrastive learning.** Create two augmented views of the same valid earlier history and train their representations to agree, alongside click prediction. Specify temperature, loss weight, and handling of other users with similar tastes. Compare with masking/cropping without this extra objective. | 8–16 h | H; extra views/contrastive comparisons |
| P2-05 | **16-dimensional vectors.** Reduce from 32 to 16 to test whether simpler preferences generalize better. Give promising sizes the same small tuning budget as P1-04. | 1–2 h after shared dimension plumbing | E; smaller table/work, total speedup unmeasured |
| P2-06 | **Recent activity bursts.** Identify closely spaced clicks versus returning after a long gap. Treat this as partial click-based session evidence, not a complete browsing log. | 2–4 h | F |
| P2-07 | **Previous exposures without clicks.** Count/time earlier real displays of this candidate to this user that went unclicked. Use labelled displays only; mark unavailable coverage and keep all current-event labels out. | 4–8 h | F plus user–item exposure-counter construction |
| P2-08 | **Similarity among competing candidates.** Add pairwise candidate-vector similarity summaries to describe competition between similar options. Relative match-score features already exist. | 2–4 h | F |
| P2-09 | **Sparse-data handling.** Test limited changes to rate smoothing and support/coverage features for rare items and sparse users. Existing rates already have smoothing and counts. | 3–6 h | F per bounded smoothing setting |
| P2-10 | **Missing-information indicators.** Audit existing indicators, then distinguish unavailable evidence from observed zero/no match where gaps remain. Specify padding and empty-history behavior consistently. | 2–4 h | F |

The initial P1-01 comparison uses **0.1** before P2-01/P2-02. For fair
comparison, retain batch ratios, loss averaging, nine sampled alternatives,
data policy, and updates. Current code scales auxiliary regularization with
the auxiliary weight: explicitly preserve that coupled recipe or hold its
coefficient separately fixed. Do not silently change both and attribute the
result only to supervision strength. For adaptive weighting, keep regularization
outside the task-gradient comparison and log actual weights and gradient sizes.

## Priority 3 — third-priority experiments

| ID | Change and initial comparison | Implementation effort | Experiment cost |
|---|---|---|---|
| P3-01 | **GradNorm.** Adapt task weights based on gradient sizes and relative learning rates. Specify how the main objective remains the selection target; compare with fixed weights and gradient similarity. | 10–18 h; shared adaptive-gradient tooling can reduce this | H |
| P3-02 | **Recency-weighted averaging.** Give older clicks progressively less weight, initially with a small declared decay search. Distinct from P1-08, which keeps separate time-window scores. | 2–4 h | E for a retrained pooling replacement |
| P3-03 | **Max pooling.** Take the largest coordinate across history vectors rather than their average. Retrain consistently; define empty histories and check sensitivity to unusual clicks. | 2–4 h | E |
| P3-04 | **Multiple interest vectors/groups.** Represent different clusters of a user's history separately, and score how each candidate matches them. Specify group count and sparse-history handling. This supersedes the earlier priority-2 “multiple interest groups” assignment. | 8–16 h | F for a fixed-embedding grouping feature; H for a jointly learned replacement, tested separately |
| P3-05 | **Learned matching function.** Replace/supplement dot product with a small neural interaction function. Separate this change from pooling changes and control extra capacity. | 6–12 h | H |
| P3-06 | **Fewer/more preference models.** Compare saved seed prefixes such as 1/3/5/10; consider 15/20 only if justified. Change real seed count, not duplicate outputs. This supersedes earlier higher-priority seed-count suggestions. | 2–4 h | F/B for saved prefixes; E for extra seeds not yet trained |
| P3-07 | **Unequal ensemble weights.** Compare equal averaging with a small development-selected weighting scheme; avoid an unconstrained weight search. | 2–4 h | F/B; tree refit needed if its input mixture changes |
| P3-08 | **Diverse-recipe ensembles.** Combine independently successful dimensions/pooling/objective recipes, checking complementary errors. Preserve shared-user validation alignment and consistent raw-score/probability conventions. | 3–6 h after member outputs exist | B plus F/E/H for missing members |
| P3-09 | **Change the averaging stage.** Compare averaged preference components feeding one LightGBM with separate full pipelines whose final probabilities are averaged. Compare raw-score/probability blending only on appropriate scales. | 4–8 h | Multiple F fits; E/H if member embeddings are missing, then B |
| P3-10 | **Explicit feature combinations.** Add a small number of justified relationships such as preference match by history support. LightGBM already learns interactions, so require incremental benefit. | 2–4 h | F |

Model-family selection is **deferred**, not silently assigned priority 3.
LightGBM remains fixed during these experiments. Other previously discussed
but unassigned ideas (such as weights 0.01/0.03, hard-negative sampling, or
larger dimension searches) are not added to the committed priority queue.

## Execution order, comparison rules, and result record

1. Complete shared groundwork and preserve the original anchor. Establish
   comparable chronological development windows; January 26 has already been
   reused and is not an untouched final test.
2. Work through priority 1, using dependencies and measured effort to order
   work within the tier. Feature-only tests can reuse embeddings; pooling,
   dimensions, and auxiliary changes need new fits. Record any pilot using
   fewer seeds; confirm finalists with the intended ensemble.
3. Test each change separately against an explicitly identified baseline.
   Historical clicks are deduplicated by `(user, item, timestamp)`; separate
   repeats remain distinct. Augmented views are not new independent evidence.
   Preserve all candidates in each validation event and score globally.
4. **Keep only supported positive AUC improvements.** Record negative and
   inconclusive results; do not keep them by default. Check dates, optimization
   seeds, and paired user/event resampling for finalists. Overlapping windows
   are correlated; a tiny increase on one reused window is weak evidence.
5. Combine successful changes in small groups and retest. Check incremental
   contribution by removing each retained group from the combination. Individually
   helpful changes may not help together. Allow equal bounded architecture-specific
   tuning where required, and report fixed-setting and tuned comparisons separately.
6. Move to priorities 2 and 3 after the required dependencies, preserving the
   original anchor alongside the evolving best configuration. Tiers express
   user preference, not permission for an exhaustive combinatorial search.
7. Freeze finalists before official evaluation. Refit on permitted data,
   verify every submission row, and associate reported scores with exact CSV
   hashes. Public leaderboard feedback is also tuning feedback.

For every experiment, save this minimum record:

```yaml
experiment_id: P1-01
status: planned # running / complete / rejected / inconclusive / retained
baseline_run: ...
hypothesis_and_single_change: ...
code_revision_and_uncommitted_source_hash: ...
data_hashes_and_available_click_policy: ...
cutoffs_and_scoring_row_hashes: ...
configuration_and_seed_ids: ...
implementation_hours_estimated_and_actual: ...
stage_runtime_estimated_and_actual: ...
hardware_peak_memory_and_disk: ...
global_auc_by_window_and_delta: ...
seed_checks_and_paired_uncertainty: ...
diagnostics_logloss_and_history_user_item_slices: ...
artifact_paths: ...
decision_and_reason: ...
```

## Sources and existing project evidence

- [Project handoff](HANDOFF.md): current pipeline, prior negative results,
  temporal validation, runtime evidence, and decisions made in this chat.
- [History-upgrade report](experiments/history-upgrade.md) and
  [Strategy 2 report](experiments/strategy2.md): existing features and measured results.
- [Competition instructions](../INSTRUCTIONS.md) and
  [official score record](../results/official_scores.json).
- [DIN](https://arxiv.org/abs/1706.06978),
  [NAIS](https://arxiv.org/abs/1809.07053), and
  [SASRec](https://arxiv.org/abs/1808.09781): attention-based alternatives.
- [CL4SRec](https://arxiv.org/abs/2010.14395) and
  [DuoRec](https://arxiv.org/abs/2110.05730): augmented/contrastive history learning.
- [Gradient similarity](https://arxiv.org/abs/1812.02224) and
  [GradNorm](https://arxiv.org/abs/1711.02257): adaptive auxiliary weighting.

Research motivates experiments; it does not establish that they improve this
dataset. The priority levels are the user's decisions, while engineering-hour
estimates and the within-tier execution suggestions are planning judgments.
