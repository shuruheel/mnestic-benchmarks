# The temporal-belief workload

**What did we believe, and why — and what happens when a source is retracted?**

The hybrid-recall suite measures *retrieval*. This workload measures the other half of an
agentic-memory substrate: **temporally-correct belief with provenance**. mnestic shipped the
pieces across the 0.10.x line — bitemporality (`TxTime`, `:as_of`), top-k proof-carrying
aggregation (`min_cost_k`), skyline/contested-set reasoning (`pareto_min`), declarative
belief revision (`:reconcile`). This workload is the artifact that proves they **compose
end-to-end**, deterministically and replayably.

## The five phases

| phase | what happens | what's asserted |
|---|---|---|
| 1 ingest | sources + bitemporal claims; valid-time split (half the subjects' claims valid from `VT_A`, half from `VT_B`) | — |
| 2 proofs | top-k proof chains per reading via `min_cost_k` (cost = −ln trust − ln conf) | proof costs, row-exact |
| 3 contradict | a new max-trust source contests every 10th subject; the contested readings surface as a `pareto_min` skyline over `[best proof cost, best source rank]`; the derived `belief` relation is `:reconcile`d from the skyline | the contested set — **multi-answer**: a contested subject returns several attributed readings instead of silently merging them |
| 4 retract | the contradictor's claims are valid-time-retracted; skyline re-derived; `belief` re-`:reconcile`d | the settled set |
| 5 assert | valid-time queries (`@ VT`) and transaction-time queries (`:as_of` on the **derived** belief, at instants captured between phases) | both axes, row-exact |

Every phase is timed; every assertion is against a **pure-Python oracle** (`oracle.py`) that
shares nothing with the engine but the seeded scenario.

## Run it

```bash
pip install -e ".[mnestic]"
python -m temporal_belief.cli --engine sqlite --subjects 1000
python -m pytest tests/test_temporal_belief.py     # the same workload as a test suite
```

Exit code 0 iff every engine answer matched the oracle. Representative single-machine run
(sqlite, 1000 subjects / ~1.8k claims, 2026-07-27, M-series): total ≈ 106 ms, of which
`:reconcile` ≈ 32%.

## Honesty notes (the workload *enforces* these, it doesn't just state them)

- **Proof identity is a user convention.** The `min_cost_k` pack carries `[src, subj, val]`
  because *we put it there* — the engine gives programmable proof-carrying aggregation, not
  automatic lineage. The oracle asserts costs, and the payload convention is documented, not
  magicked.
- **`:reconcile` is whole-relation recompute.** The reported `:reconcile` share of runtime
  is the standing measurement for the ROADMAP's scoped-`:reconcile` trigger: if it dominates
  at a scale you care about, that is the evidence that item has been waiting for. File an
  issue with your run.
- **Valid time is history, not state.** After the retraction, `@ VT_LATE` (an instant
  *before* the retraction on the valid axis) still shows the retracted source's claims —
  that is the axis doing its job. (The first draft of this workload's own oracle got this
  wrong and the engine corrected it; the fixed expectation is commented at the assertion.)
- **Not included (yet):** the companion axis — running the same oracle-verified workload
  against a server-based knowledge engine to isolate the per-query network floor an
  in-process engine never pays. That comparison follows the same correctness-first
  discipline as the hybrid-recall suite when it lands; this workload is its prerequisite.
