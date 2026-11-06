# Research backlog: status

Your working list. The reasoning for each item is in `research/lab/RATIONALE.md`, which is read-only.
Update statuses here and add new ideas with a one-line rationale.
Statuses: `todo`, `running`, `done (+gain)`, `rejected (why)`, `blocked (why)`.

**Current status:** `t4-refit-stale24h` underperformed (-0.0008 vs t3-budget-seeds30). `blend-refit-seeds-stale` leads with 0.802802 dev. `t4-stale-history-12h` running (stale-history-hours MF view).

**Best recipe:** `--steps 3000 --init-std 0.005 --seeds 30 --tree-refit-policy` (`t4-tree-refit`, Jan 26 0.824669). **Fast 10-seed parent for MF ideas:** `claude-s3000-init005` (dev 0.800903). **MF parent for tree-only ideas:** `t3-budget-seeds30`.

||| # | Item | Status |
|||---|------|--------|
||| 1 | T3: updates, initial std, seeds | done (+0.0020 on Jan 26) |
||| 2 | P1: clicks from query histories | rejected (noise level, 0.3% more clicks) |
||| 3 | Error analysis by slice | todo: staleness measured, rare-item support still open |
||| 4 | P2/P5: robustness to short and old histories | todo |
||| 5 | T5: blends on development rows only | todo |
||| 6 | P4: rare-item collaborative signal | todo, only if item 3 shows rare-item errors |
||| 7 | P3: better negatives | todo |
||| 8 | T6: aligning the MF training population | todo; tree-side alignment rejected |
||| 9 | T4: tree policy and feature ablations | done (refit adopted, features no effect) |
||| 10 | P7: event propensity | todo, low priority |
||| 11 | P8: distillation, sequence models | todo, last |
||| 13 | Stale-history MF views (`--stale-context-hours`, `--stale-history-hours`) | running (t4-stale-history-12h) |
||| 14 | S10: stale-context 24h test | done (+0.002, below t4-tree-refit's 0.8022) |
||| 15 | S11: stale-context 48h | rejected (hurt performance, -0.001) |
||| 16 | S12: 15 seeds with 24h stale-context | done (+0.0007) |
||| 17 | S13: 30 seeds with 24h stale-context | done (+0.0005, still below threshold) |
||| 18 | S14: 50 seeds with 24h stale-context | done (+0.00025, diminishing returns) |
||| 19 | S15: 75 seeds with 24h stale-context | done (+0.00017, seed curve flattened) |
||| 20 | T4-refit + Stale: tree-refit-policy with 24h stale-context | rejected (underperformed parent -0.0008) |

## New ideas

### Stale-history-hours MF view (new)
Hypothesis: Training MF histories stop 12h before each event, mimicking the gap between training (8.5h median last click) and evaluation (42.6h median). Based on HANDOFF.md §5 item 9 - staleness is the steepest weak slice, tree-side alignment failed, so the change belongs in the representation. Test with 10 seeds for speed against claude-s3000-init005 parent.