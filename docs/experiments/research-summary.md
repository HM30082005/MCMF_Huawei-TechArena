# Research results and engineering lessons

This report summarizes the local research checkout through **November 4, 2026**, prepared for the public repository on November 6. Scores are taken from the [archived evaluator output](research-leaderboard.md), rather than reconstructed from notebook claims. The source snapshot came from research commit `fa6c5df`; the main repository's earlier pipelines remain available in Git history.

## Results and evidence

| Run | Development AUC | Confirmation AUC | Confirmation gain vs baseline |
|---|---:|---:|---:|
| `baseline-0.816323` | 0.798361 | 0.822508 | — |
| `claude-s3000-init005-seeds20` | 0.801910 | 0.824160 | +0.001653 |
| `t3-budget-seeds30` | 0.802381 | 0.824512 | +0.002004 |
| `t4-tree-refit` | 0.802224 | 0.824669 | +0.002161 |
| `blend-refit-seeds-stale` | **0.802802** | **0.824787** | **+0.002280** |

The blend's paired development gain over the baseline was **+0.004442**, with a 95% event-bootstrap interval of **[+0.003868, +0.004978]**. Development gains were positive on both January 24 and January 25. The interval is conditional on fitted models; it does not account for repeated model selection.

The only recorded official result is **0.816323** for the original ten-seed LightGBM submission. Its [score ledger](../../results/official_scores.json) includes the CSV checksum and explicitly records that the competition receipt was not independently verified. Neither the later blend nor the local confirmation result establishes a newer official score or leaderboard rank.

The supplied leaderboard in the project conversation lists MCMF at 0.825749, but does not identify a submission checksum or associate that score with these local runs. It is therefore not substituted into the official score ledger or claimed as the result of this pipeline.

## What changed

The later experiments retain the shared 32-dimensional item table, historical-click auxiliary loss, and boosted-tree scorer. They investigate fitting duration, initialization, independent seed count, the timing of auxiliary context, and tree-selection policy.

- **Shorter fitting and smaller initialization:** 3,000 updates and initial standard deviation 0.005, compared with 5,000 and 0.05 in the original baseline.
- **Independent seed ensembles:** 20–30 seeds improve the recorded results. Larger 50–75 seed trials show diminishing returns and do not justify unlimited compute.
- **Fitting-period tree selection:** choose the tree count on the final fitting day, scale it for the refit, and train on all permitted fitting rows. This avoids using the scored window to choose that stage's tree count.
- **Prediction blending:** the best local blend averages logits from the 30-seed recipe, its tree-refit variant, the 20-seed recipe, and a 50-seed auxiliary-context variant. Components were chosen using development evidence; weights are equal.

The [saved manifests](../../configs/experiments/) preserve the component recipes and source run metadata. One recorded inconsistency is retained: `t4-tree-refit/run.json` names the original baseline as parent, while its hypothesis and recipe describe a tree-only comparison to the 30-seed run. The narrative comparison follows the recipe; the archived metadata has not been silently rewritten.

## Negative results and corrected experiments

- **Query-history augmentation:** the corrected run added pre-cutoff clicks to the auxiliary objective, but its tree gain over the 30-seed parent was only about +0.00008 on development. The initial zero-addition result came from a lookup bug and was not a valid negative experiment.
- **Feature ablations:** earlier attempts accidentally restored old learned-MF columns. Corrected reference/no-spectral/no-extended trials were close to their parent, so they do not establish a strong benefit from those feature groups.
- **Mixed fitting budgets:** varying the number of updates across seeds did not outperform the uniform-budget parent.
- **Auxiliary context gaps:** a wider 48-hour gap lost to the 24-hour alternative. Combining the 24-hour gap with the tree-refit recipe also lost to its 30-seed parent; improvements did not simply add together.
- **Incomplete experiments:** a stale-history training experiment was still described as running in the final notebook. It is not presented as a measured success.

The [historical notebook](../../research/lab/NOTES.md) preserves the original entries and a correction section. Some entries contain outdated or inconsistent interpretations; use the evaluator snapshot for recorded scores and this summary for the corrected conclusions. “Eligible for confirmation” in the archived leaderboard is a baseline-relative evaluator status, not proof that the run beat the strongest current candidate.

## Evaluation design

Development uses the first event per user during Jan 24–25, with models fitted before Jan 24. Confirmation uses the first event per user on Jan 26, with models fitted before that day. Events remain whole and the evaluator requires exact agreement with the frozen row indices.

The harness separates raw predictions from scoring, checks finite values and window identity, reports performance slices, and uses paired bootstrap deltas. Its promotion rule requires at least +0.001 development AUC over baseline, a positive lower confidence bound, and positive gains on both development days. It limits confirmation checks to two per UTC day.

These safeguards reduce accidental leakage and uncontrolled testing; they do not make the reused confirmation set an untouched final test. Local windows also differ from the evaluation population. Sparse and stale histories remain a weakness. The original history-upgrade report measured AUC 0.776727 for supplied histories of 1–9 clicks, versus 0.833390 for 10–20 clicks, for its development-selected CatBoost model; those slice scores should not be relabelled as the later blend's results.

## Reproducibility

The public repository includes training code, synthetic tests, aggregate reports, pinned dependencies, run manifests, and the official submission checksum. It excludes raw data, model checkpoints, cached features, per-row predictions, and submission CSVs.

A reviewer can run the unit tests without competition data. Rebuilding a pipeline requires the original dataset, preprocessing, and training. Exact replay of the recorded blend additionally requires all four component fits and their aligned evaluation logits. JSON manifests document those experiments but are not executable config files. Source commit identifiers refer to the original local research history and may not all resolve on GitHub.

The current launcher adds safeguards introduced after some historical experiments, including a 30-seed cap. Replaying the historical 50-seed component through that launcher is intentionally refused. The pipeline CLI can express the archived recipe directly, but this is an archival replay, not the current recommended search policy. The original automation setup is macOS/Hermes-specific and assumes existing baseline artifacts; it is not needed for tests or direct pipeline training.

The legacy baseline replay tool checks existing saved models and their exact CSV hash; it cannot reconstruct absent checkpoints from a fresh clone. No full training or independent score reproduction was performed while preparing this portfolio update.

## Validation of this repository update

The integrated source snapshot passed **60 unit tests** locally on November 6, 2026. Tests cover model and feature invariants, checkpoint resume, experiment configuration, tree refit behaviour, and the evaluator using synthetic examples. This checks implementation behaviour; it does not reproduce the competition AUC.
