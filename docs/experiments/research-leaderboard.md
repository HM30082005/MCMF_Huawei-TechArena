# Internal leaderboard

Generated 2026-11-04 22:17 UTC by `research/evaluate.py`. Do not edit by hand. Development is the search window (first event per user, Jan 24–25); confirmation (first event per user, Jan 26) is reserved for development winners. Deltas are against the 0.816323 baseline on identical rows, with 95% paired bootstrap intervals.

| # | Run | Status | Confirmation AUC | Δ conf | Development AUC | Δ dev [95% CI] | Δ Jan 24 / Jan 25 | Hypothesis |
|---:|---|---|---:|---:|---:|---|---|---|
| 1 | `blend-refit-seeds-stale` | SUBMIT CANDIDATE | 0.824787 | +0.002280 | 0.802802 | +0.004442 [+0.003868, +0.004978] | +0.004233 / +0.005309 | Equal blend of different MF fits beats any single run; components chosen on development rows only (0.802802 there) |
| 2 | `t4-tree-refit` | SUBMIT CANDIDATE | 0.824669 | +0.002161 | 0.802224 | +0.003863 [+0.003252, +0.004452] | +0.003731 / +0.004381 | Pick tree count by early stopping on the last fitting day, scale by days and refit (reference exp 13) instead of fixed d |
| 3 | `t3-budget-seeds30` | SUBMIT CANDIDATE | 0.824512 | +0.002004 | 0.802381 | +0.004020 [+0.003427, +0.004583] | +0.003840 / +0.004806 | 3,000 MF updates + init std 0.005 + 30 seeds (extend seed curve 20 |
| 4 | `claude-s3000-init005-seeds20` | SUBMIT CANDIDATE | 0.824160 | +0.001653 | 0.801910 | +0.003549 [+0.002874, +0.004142] | +0.003308 / +0.004541 | 20 MF seeds instead of 10 (averaging curve still rising 5->10); parent claude-s3000-init005; code claude/mf-epoch-schedu |
| 5 | `baseline-0.816323` | baseline (official 0.816323) | 0.822508 | – | 0.798361 | – | – | Baseline: ten-seed shared-item history MF + LightGBM (31 leaves); official AUC 0.816323 |
| 6 | `p1-query-clicks-v2` | eligible for confirmation | – | – | 0.802460 | +0.004099 [+0.003516, +0.004707] | +0.003839 / +0.005219 | Pre-cutoff clicks from scored queries' histories added to the auxiliary loss only (first real test of P1; the earlier ru |
| 7 | `t4-features-no-extended` | eligible for confirmation | – | – | 0.802460 | +0.004099 [+0.003535, +0.004653] | +0.003888 / +0.004988 | Feature ablation 'no-extended' with the old learned-MF features correctly excluded (earlier runs restored them); parent  |
| 8 | `p1-query-clicks-30` | eligible for confirmation | – | – | 0.802445 | +0.004085 [+0.003498, +0.004680] | +0.003958 / +0.004577 | P1: Add query history clicks to T3 recipe (3000 steps, init std 0.005, 30 seeds) |
| 9 | `t4-features-no-spectral` | eligible for confirmation | – | – | 0.802360 | +0.003999 [+0.003449, +0.004570] | +0.003834 / +0.004671 | Feature ablation 'no-spectral' with the old learned-MF features correctly excluded (earlier runs restored them); parent  |
| 10 | `t3-budget-seeds50` | eligible for confirmation | – | – | 0.802354 | +0.003993 [+0.003452, +0.004535] | +0.003734 / +0.005128 | Extend seeds curve to 50 under T3 recipe (3000 steps, init std 0.005, 50 seeds) |
| 11 | `t4-features-reference` | eligible for confirmation | – | – | 0.802281 | +0.003920 [+0.003320, +0.004542] | +0.003774 / +0.004452 | Feature ablation 'reference' with the old learned-MF features correctly excluded (earlier runs restored them); parent t3 |
| 12 | `t4-no-spectral` | eligible for confirmation | – | – | 0.802214 | +0.003853 [+0.003252, +0.004458] | +0.003608 / +0.004825 | Tree-only with no-spectral features (keep co-click, remove spectral only) |
| 13 | `t4-reference-features` | eligible for confirmation | – | – | 0.802140 | +0.003780 [+0.003168, +0.004409] | +0.003639 / +0.004292 | Tree-only with reference 39 features |
| 14 | `s15-seeds75-stale24h` | eligible for confirmation | – | – | 0.802085 | +0.003725 [+0.003100, +0.004295] | +0.003456 / +0.004858 | Extend seed curve to 75 seeds with 24h stale-context |
| 15 | `s14-seeds50-stale24h` | eligible for confirmation | – | – | 0.801918 | +0.003557 [+0.002940, +0.004161] | +0.003347 / +0.004425 | Extend seed curve to 50 seeds with 24h stale-context to beat t3-budget-seeds30's dev 0.802381 |
| 16 | `s13-seeds30-stale24h` | eligible for confirmation | – | – | 0.801670 | +0.003309 [+0.002677, +0.003897] | +0.003064 / +0.004361 | Extend seed curve to 30 seeds with 24h stale-context to beat t3-budget-seeds30's dev 0.802381 |
| 17 | `t4-refit-stale24h` | eligible for confirmation | – | – | 0.801561 | +0.003200 [+0.002547, +0.003838] | +0.002958 / +0.004198 | Combine proven tree-refit-policy with 24h stale-context |
| 18 | `s12-seeds15-stale24h` | eligible for confirmation | – | – | 0.801156 | +0.002795 [+0.002178, +0.003406] | +0.002552 / +0.003830 | Extend seed curve to 15 seeds with 24h stale-context to beat t4-tree-refit's dev 0.8022 |
| 19 | `claude-s3000-init005` | eligible for confirmation | – | – | 0.800903 | +0.002542 [+0.001933, +0.003137] | +0.002409 / +0.003199 | Init std 0.005 on top of 3,000 steps (reference exp 13: smaller start keeps its gain under averaging); parent t3-steps30 |
| 20 | `t3-asymmetric-steps-v2` | eligible for confirmation | – | – | 0.800770 | +0.002409 [+0.001817, +0.003007] | +0.002292 / +0.002884 | Varying MF updates per seed (2500-4500) makes the average more diverse than uniform 3,000; parent claude-s3000-init005 |
| 21 | `t4-refit-fixed` | eligible for confirmation | – | – | 0.800519 | +0.002158 [+0.001507, +0.002704] | +0.002166 / +0.002086 | Fixed label format (2D vs 1D) for confirmation phase comparison; 30 seeds, 5000 steps |
| 22 | `s10-stale-context-24h` | eligible for confirmation | – | – | 0.800398 | +0.002037 [+0.001391, +0.002730] | +0.001840 / +0.002820 | Auxiliary context that omits clicks within 24 h of the target helps stale users; first real test of this idea |
| 23 | `t3-steps3000` | eligible for confirmation | – | – | 0.800252 | +0.001891 [+0.001446, +0.002304] | +0.001693 / +0.002783 | 3,000 MF updates generalise better than 5,000 (reference experiment 13; backlog item 1) |
| 24 | `claude-firstday` | eligible for confirmation | – | – | 0.799998 | +0.001637 [+0.001177, +0.002110] | +0.001464 / +0.002409 | Trees trained on each user's first event per day (scored-population alignment); parent t3-steps3000, tree-only; code cla |
| 25 | `claude-ageweight` | eligible for confirmation | – | – | 0.799796 | +0.001435 [+0.000961, +0.001897] | +0.001275 / +0.002197 | Tree rows weighted by history-age bin towards the scored rows' age distribution (covariate-shift alignment); parent t3-s |
| 26 | `s11-stale-context-48h` | eligible for confirmation | – | – | 0.799395 | +0.001034 [+0.000312, +0.001711] | +0.000887 / +0.001621 | Wider stale-context (48h) with same 10 seeds to beat t4-tree-refit's dev 0.8022 |
| 27 | `claude-epochs4.2` | development only | – | – | 0.798377 | +0.000017 [-0.000687, +0.000660] | -0.000289 / +0.001476 | Constant 4.2 passes per MF fit (= 3,000 steps on 6 days) beats fixed 3,000 steps; parent t3-steps3000; code claude/mf-ep |

## Submission candidates

- `blend-refit-seeds-stale`: confirmation 0.824787 (+0.002280); `submission_blend.csv` sha256 `42592f972707ce95b0cdaca38aadba410b553fa74d88f662a86dab1e7179b685`
- `t4-tree-refit`: confirmation 0.824669 (+0.002161); full-data CSV not built yet
- `t3-budget-seeds30`: confirmation 0.824512 (+0.002004); full-data CSV not built yet
- `claude-s3000-init005-seeds20`: confirmation 0.824160 (+0.001653); full-data CSV not built yet
