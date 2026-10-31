# Research backlog

Ranked by expected real-AUC gain × probability it holds up ÷ compute. Seeded on
2026-10-31 from `docs/HANDOFF.md` (sections 5–9). The rationale and evidence
below come from that handoff; read the cited section before starting an item.
Keep this file current: update status and evidence after every run, re-rank
freely, and add your own ideas with a rationale. Statuses: `todo`, `running`,
`done (+gain)`, `rejected (why)`, `blocked (why)`.

## Ground truth to reason from

- **The model.** Ten independent shared-item MF seeds pool a user's
  history-item vectors and dot them with the candidate, with scalar user and
  item biases. It is trained with impression BCE plus an auxiliary
  historical-click loss (1 click vs 9 popularity-sampled alternatives). The
  MF scores and 54 behavioural features feed LightGBM (70 features).
  Development AUC is 0.798361, confirmation 0.822508, official 0.816323.
- **Where the big gains came from.** Changing representation and supervision
  took confirmation AUC from 0.7268 to 0.8225, and swapping CatBoost for
  LightGBM barely mattered. The reference repo's sequence of public scores
  shows the same thing: pooled history 0.767 → aux clicks 0.777 → 5 seeds 0.796
  → 10 seeds 0.798 → longer histories 0.805 → other clicks as context 0.813 →
  shared item table 0.816. **Representation and data changes are what move
  this metric.**
- **The evaluation population differs from training.** Evaluation scores each
  user's first event in the following week. Histories are older (median last
  click 42.6 h vs 8.5 h) and shorter (median 15 vs 20), with more empty ones
  (1.26% vs 0.33%) and 2.7% users unseen in training. Random splits hide this.
- **Weak slices (CatBoost, confirmation).** Empty history 0.545 (198 events),
  history 1–9: 0.777, users without fitting events 0.744, history 10+: 0.833.
  These are rare in absolute terms, so a specialist alone cannot deliver +0.014.
- **The global row-level AUC rewards cross-event ranking.** Event propensity
  matters, and forcing per-event probabilities to sum to 1 did not help.

## Already tried: don't repeat without a new hypothesis (HANDOFF §6)

| Idea | Evidence | Consequence |
|---|---|---|
| Learned user preference vectors | Our old MF overfit; removing them gave the reference a large gain; regularised residual vectors also failed | Keep scalar biases; infer taste from history |
| More updates/epochs by default | Our MF overfit; reference degraded at 8,000+ updates | Study step curves under averaging instead |
| 64 dimensions | Reference: often worse than 32 | Small check only |
| Strong recency, order, learned position weights | Mostly failed in the reference; overfit | Pooled taste is favoured; needs residual evidence first |
| Attention / Transformers / DIN | No clear gain; our DIN benchmark scored 0.673 | Only with a specific hypothesis and the new supervision |
| Choice-only loss, event-group ranking, LambdaRank | Reference worse on final trees/public | Keep the global row-level objective |
| Per-event normalisation | Our sweep chose zero adjustment | Don't force sums to one |
| 49 auxiliary negatives instead of 9 | Reference worse | Changing the negative *distribution* is a different hypothesis |
| Regularisation picked from one seed | Gains vanished at 3 seeds | Always judge under the averaged ensemble |
| Shared spectral initialisation | One-seed gain vanished under averaging; reduces diversity | A separate graph *score* is a different test |
| Candidate position, shown-without-click | Flat or worse | Low priority |
| More tree tuning, tiny blends | About +0.0002–0.0003 | Need repeatable gains before spending effort |

## Ranked backlog

### 1. T3: shorter fit, smaller initialisation, more seeds (HANDOFF §5 item 1, §7 T3) — todo
**Hypothesis:** the reference's experiment-13 recipe (3,000 updates, init std
0.005, 15 seeds) generalises better than ours (5,000, 0.05, 10).
**Rationale:** it is the main recipe difference left from the reference, whose
step sweep on a six-day fit favoured 3,000 over 5,000 updates. It reports about
0.818 public versus our 0.816. Longer fits are known to overfit here, and
averaging rewards diverse, less-overfit seeds.
**Test:** one change per run against the baseline, development only:
`--steps 3000`; `--init-std 0.005`; both; both with `--seeds 15` (45k
seed-updates vs our 50k). Seed-prefix selection inside the pipeline picks the
averaging count on development; keep it.
**Cost:** about 15 min per run.

### 2. P1 audit: extra real clicks from query histories (HANDOFF §5 item 2, §8 P1) — todo
**Hypothesis:** adding the pre-cutoff clicks in the *supplied histories of the
scored queries* to the MF fitting click log improves the representation. Those
are real historical positives; the reference does this and we don't.
**Rationale:** the largest unclosed implementation difference from the
reference. The value is unmeasured; first count what it adds.
**Test:** first an analysis script (no training): for each cutoff, count new
unique (user, item, time) clicks, new users and items, and support gains for
rare items. If the gain is material, use them **auxiliary-only** at first,
keeping the impression denominators unchanged. Filter *every* click and context
to before that model's cutoff (the daily cutoff for daily features, not Jan 27).
Never use outcomes of held-out queries. Record the information-access policy in
`run.json` (`historical_inputs`). If few clicks are added, reject on that
evidence.

### 3. Error analysis to direct the rest (HANDOFF §7 T2, §4) — todo
**Why:** the weak-slice numbers above are for CatBoost, not our LightGBM anchor,
and rare-item support was never measured. Measure LightGBM development errors by
history length (supplied vs reconstructed), history age, user and item
fitting support, candidate repeats, time bucket, and events with any click. Do
it on development rows only. The result decides between items 4, 5 and 6.

### 4. P2/P5: robustness to short and old histories — todo
**Hypothesis:** training on several *views* of the same known clicks (random
context subsets, context dropout, caps of 5/10/20/50) makes the pooled
representation robust to the shorter, older evaluation histories. Shrinking
short-history vectors towards a learned prior helps cold and sparse users.
**Rationale:** evaluation histories are shorter and older than training ones,
and the 1–9 and empty slices are weak. This changes views of existing
positives without inventing labels.
**Also test:** frequency vs unique-item pooling, bounded repeat weights, and
user-balanced vs event-uniform auxiliary sampling (with explicit weights so
active users don't dominate). Avoid strong time decay, which has poor prior
evidence.

### 5. T5: CatBoost + LightGBM blend, and diverse MF recipes — todo
**Hypothesis:** a fixed blend of near-tied learners, or of MF ensembles with
different recipes, adds a small but real gain.
**Rationale:** CatBoost and LightGBM are near-tied (confirmation 0.82225 vs
0.82251). The gain is limited unless errors are complementary, so measure error
correlation first. Recipe diversity from items 1, 2 and 4 may be worth more
than more tree seeds. Choose blend weights on development only; compare
probability and logit averaging; never stack on in-sample predictions.

### 6. P4: rare-item collaborative signal — todo (if item 3 shows rare-item errors)
**Hypothesis:** a second collaborative score (top-k co-click neighbours, an
EASE-style sparse linear item model, or a small LightGCN) helps items with
little support, if added as tree features with shrinkage.
**Rationale:** item IDs are individual items with no metadata, so rare items
must borrow strength from co-click structure. Shared spectral *initialisation*
failed, but a separate prediction signal is a different test. Budget memory:
dense 50,000² float32 is 10 GB, so stay sparse. Use the same temporal and
query-history policy as the MF.

### 7. P3: better negatives — todo
Mix popularity-sampled alternatives with uniform or smoothed-popularity ones,
or use hard negatives from real earlier candidate sets (actual non-clicked
impressions). Mask positive collisions and examine false negatives among other
known liked items. Keep K=9 at first, because K=49 failed. Retest the
auxiliary weight under the new normalisation.

### 8. P6: align fitting with the first-event population — todo
**Hypothesis:** active users dominate training, while evaluation has one query
per user. User-balanced event weights, or pseudo-cutoff views that select the
first subsequent event per user, align training with the scored population.
Clipped importance weights from a train-vs-evaluation discriminator on
label-free covariates are possible too; fit and validate them chronologically
and report sensitivity. Always report the standard unweighted AUC.

### 9. T4: tree fitting policy and feature ablations — todo (cheap, small expected gain)
Compare the fixed iteration counts with last-day early-stop-and-refit (declare
any scaling). Ablate feature groups: a compact reference-like set (they use 39
features), no spectral/co-click, no extended-history features. Then a bounded
search over leaves, min-leaf and L2. These are tree-only, so reuse the
baseline's MF parts via APFS clones.

### 10. P7: event propensity × conditional choice — todo (low)
`P(click i) = P(any click) × P(i | click)`, keeping no-click events for the
first factor. Only test this if item 3 shows event-propensity information the
trees miss; similar objectives failed before.

### 11. P8: distillation and sequence models — todo (exploratory, last)
Only after the above plateaus, and only with a specific residual hypothesis.
Keep real and pseudo labels separate, and don't validate on rows used to fit
the teacher.

## Engineering items (do when they unblock research)

- Cache identity: `strategy1.cache_fingerprint()` does not hash the evaluation
  arrays (HANDOFF §7 T1). Fix it before any change that touches evaluation
  inputs.
- Extract a behaviour-only feature builder from `strategy2.py`. New base
  features currently require the expensive old-MF preparation path.
- Log training and auxiliary losses separately, plus step-wise development
  curves for prefix-ensemble studies (HANDOFF §7 T6).
