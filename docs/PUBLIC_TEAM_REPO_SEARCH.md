# Public team/repository search — 4 November 2026

Searched all 75 supplied team names (with shortened/normalized punctuation for some names) against GitHub repository names/descriptions/READMEs using `"<name>" techarena in:readme`, then broadened discovery using CTR, Huawei, Ireland, and the exact dataset columns. Also searched the web for leading/distinctive names. This is a public-source search, not an exhaustive search of all code, private repositories, or unindexed pages.

## Main finding

Useful public implementations exist. No verified repository was found for Latent Space, Cybots, NextToken, Access [-1], or The Dark Triad, and no inspected repository substantiates their ~0.834 scores. Similar names alone do not establish team identity. Scores below are authors' reports, not reproduced results.

| Leaderboard team | Public repository | Attribution evidence | Useful clue |
|---|---|---|---|
| Huawei TUDers (#69, 0.810459) | [Ife-Ojo/Huawei-CTR-Shared-Baseline](https://github.com/Ife-Ojo/Huawei-CTR-Shared-Baseline) and [Huawei-TUDers](https://github.com/Ife-Ojo/Huawei-TUDers) | Strong: matching team name on related repository and exact score | Separate event-click probability from conditional item choice; SVD geometry and tree ensemble |
| SmartRec (#52, 0.823422) | [YASASHWI70/techarena_ctr](https://github.com/YASASHWI70/techarena_ctr) | Likely: final report gives exact score 0.823422; team name not explicitly verified | ~12-dimensional taste model; implicit ALS with explicit skipped-item negatives; softmax over ten candidates plus no click; validation cohort weighting |
| Overfitters (#54, 0.822518) | [Abhinav0909/Recommendation-System](https://github.com/Abhinav0909/Recommendation-System) | Likely: README and named best submission give exact score 0.822518; team name not explicitly verified | Label-free exposure SVD, 8/16 dimensions, learned item groups, residual listwise models |
| No verified team mapping | [jancastachova/techarena2026-atu](https://github.com/jancastachova/techarena2026-atu) | Reported public 0.825754, close to MCMF's 0.825749 but a different score; do not infer ownership. This workspace identifies MCMF's own repository as HM30082005/MCMF_Huawei-TechArena | Existing reference has substantially advanced since our October 30 source study |
| Unidentified team | [turab45/techarena-ctr-prediction](https://github.com/turab45/techarena-ctr-prediction) | Exact challenge, no team mapping | Slate transformer, item2vec, 355-feature GBDT; transformer reports 0.81292 local, not official |
| Unidentified team | [Engr-Riaz-Ahmad-Butt/techarena-ctr-prediction](https://github.com/Engr-Riaz-Ahmad-Butt/techarena-ctr-prediction) | Exact challenge, no team mapping | Exposure ALS interactions and activity-adjusted validation; reported official 0.816623 |
| Unidentified team | [H1di/Huawei-Hackaton-2026](https://github.com/H1di/Huawei-Hackaton-2026) | Matching schema and competition | Point-in-time statistics and chronological stacking infrastructure; inspected README has no competitive score |

## Most useful hypotheses for our project

### 1. Improve validation coverage before chasing small gains

The likely SmartRec report measures 34% of evaluation users with fewer than ten known clicks, versus 22% in its one-day holdout. Its activity-weighted local estimate 0.82346 tracked its reported public 0.823422. These percentages and agreement are author-reported, not independently checked here.

The likely Overfitters report finds only 66.62% of evaluation first events fall within 24 hours of cutoff. Their revised checks use the first event per user across all remaining days, with feature models frozen before cutoff and meta-training labels filtered by actual timestamp.

Our README currently uses each user's first January 26 event, so this is directly relevant. Add multi-day first-event checks and performance by activity/history coverage; consider activity-weighted AUC as a diagnostic alongside ordinary AUC. Weighting is a proxy, not a guarantee of leaderboard performance.

Sources: [SmartRec final report](https://github.com/YASASHWI70/techarena_ctr/blob/40a4cab2dd95bdbe3a160abfb330246a2eb357d3/reports/final_report.md), [weighting implementation](https://github.com/YASASHWI70/techarena_ctr/blob/40a4cab2dd95bdbe3a160abfb330246a2eb357d3/pipeline/eval_weight.py), [Overfitters report](https://github.com/Abhinav0909/Recommendation-System/blob/88ddec5c0ef84b53c22578c029f6cb0beab2e9ff/code_patterns/README.md).

### 2. Compare small taste dimensions and learn from skipped impressions

The likely SmartRec README argues for a static, roughly 12-dimensional taste space. Its iALS code defaults to dimension 12 and treats clicked pairs, shown-but-never-clicked pairs, and unseen pairs differently, with regularization. Its taste model pools clicked vectors and a learned multiple of skipped-item vectors, then models ten candidates plus no click.

This suggests a controlled comparison around 8/12/16 dimensions against our 32-dimensional shared-item model, and explicit exposure/skip signals. It does not prove the official simulator has exactly 12 dimensions.

Sources: [method](https://github.com/YASASHWI70/techarena_ctr/blob/40a4cab2dd95bdbe3a160abfb330246a2eb357d3/README.md), [iALS source](https://github.com/YASASHWI70/techarena_ctr/blob/40a4cab2dd95bdbe3a160abfb330246a2eb357d3/pipeline/ials.py).

### 3. Add an independent exposure representation

The likely Overfitters solution uses normalized user–item impression matrices, truncated SVD, removes the leading popularity direction, and adds 8/16-dimensional candidate/history/exposure features plus learned groups. Representations must be fitted before each cutoff. Its later rounds explicitly report selection leakage and failed gains: use the retained baseline and audited code as references, not the best-looking local number.

Sources: [repository results](https://github.com/Abhinav0909/Recommendation-System/blob/88ddec5c0ef84b53c22578c029f6cb0beab2e9ff/README.md), [method notes](https://github.com/Abhinav0909/Recommendation-System/blob/88ddec5c0ef84b53c22578c029f6cb0beab2e9ff/code_patterns/README.md).

### 4. Refresh the already-known ATU reference

Our existing study inspected commit 1ed9fd2. The current main tree was c916d319e798166d9da2963d9c29e6d49e278801 during this search. Its README is stale (0.82033), while the detailed upload log reaches 0.825754 (rounded to 0.82575 in the experiment table).

Later work includes audience-minus-exposure representations; multiple graph hops; cosine similarities projected onto eight strong principal directions; a longer-trained MF view; and binary LightGBM/XGBoost plus a neural choice model. This is a concrete source of additional hypotheses beyond the original tied-embedding/seed improvements we already studied.

Source: [detailed results](https://github.com/jancastachova/techarena2026-atu/blob/c916d319e798166d9da2963d9c29e6d49e278801/experiments/RESULTS.md).

## Limits and search coverage

No model was trained, no external code was executed, and no submission was downloaded or uploaded. Code and documentation inspection do not independently establish correctness or absence of leakage. Reported negative experiments apply to their setups; they do not rule out our separately specified recent/long-term history experiment.

The following table records the initial GitHub README search for every supplied team. A nonzero hit count includes unrelated keyword matches and is not a verified team match. Empty results mean no hit for this query, not proof that no public implementation exists.

| Rank | Search name | README-search hits |
|---|---|---|
| 1 | Latent Space | 1 |
| 2 | Cybots | 0 |
| 3 | NextToken | 0 |
| 4 | Access | 20 |
| 5 | The Dark Triad | 0 |
| 6 | CSDT | 0 |
| 7 | Him-An | 0 |
| 8 | SSS | 0 |
| 9 | Slopify | 0 |
| 10 | prjkt-avengers | 0 |
| 11 | Technological University Dublin | 3 |
| 12 | What?! | 20 |
| 13 | free lunch | 0 |
| 14 | Team Signal | 0 |
| 15 | Rererererecommendation | 0 |
| 16 | PROMETHEUS | 1 |
| 17 | Learn2RecSmartly | 0 |
| 18 | P3 Labs | 0 |
| 19 | YLZ | 0 |
| 20 | Nova | 2 |
| 21 | Epsilon | 0 |
| 22 | Vitality | 1 |
| 23 | MUnemployed | 0 |
| 24 | Recommended For You | 0 |
| 25 | the recommenders | 1 |
| 26 | Beyond Baseline | 0 |
| 27 | Triple T | 0 |
| 28 | THUNDERSTORM | 0 |
| 29 | Success | 9 |
| 30 | Rustaceans | 0 |
| 31 | Yeepee | 0 |
| 32 | MYS | 0 |
| 33 | Tensor Titans | 0 |
| 34 | SAFA | 0 |
| 35 | M-IRE | 0 |
| 36 | Conversion | 5 |
| 37 | 404 Found | 0 |
| 38 | The Clanker Program | 0 |
| 39 | HalluciNation | 2 |
| 40 | ClickBait | 0 |
| 41 | Elevated | 0 |
| 42 | MCMF | 0 |
| 43 | VNAI | 0 |
| 44 | HackBots | 0 |
| 45 | SignalRank | 0 |
| 46 | Orbit | 2 |
| 47 | DCU_IT | 0 |
| 48 | BotProof Rec | 0 |
| 49 | Git happens | 0 |
| 50 | Ctrl + Predict | 0 |
| 51 | Warriors | 0 |
| 52 | SmartRec | 0 |
| 53 | NextClick AI | 0 |
| 54 | Overfitters | 0 |
| 55 | Tourist | 0 |
| 56 | Six Six Six | 0 |
| 57 | LocalPulse | 0 |
| 58 | this is our awesome project | 0 |
| 59 | SPB | 0 |
| 60 | ChaiScript | 0 |
| 61 | UCD Hack Club | 0 |
| 62 | TCD Top-K | 0 |
| 63 | UOG | 0 |
| 64 | Algorithmic Frontier | 0 |
| 65 | giv us | 0 |
| 66 | EVA-1 | 0 |
| 67 | limerick_legends | 0 |
| 68 | UnevenGradient | 0 |
| 69 | Huawei TUDers | 2 |
| 70 | ANAtropy | 0 |
| 71 | GCDstud | 0 |
| 72 | Ping Panthers | 0 |
| 73 | Messi | 0 |
| 74 | EirHack | 0 |
| 75 | Learn2Rec | 0 |

Additional discovery queries included `techarena 2026`, `techarena Ireland in:readme`, `"techarena" CTR in:readme`, `"history_item_ids" "clicked" in:readme`, and `"predicted_probability" "Huawei" in:readme`. Name-only SignalRank hits inspected concerned a different recruitment-ranking hackathon and were excluded.

