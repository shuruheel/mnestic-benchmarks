"""temporal-belief-bench: the reproducible temporal-belief workload.

Proves that mnestic's bitemporal + provenance features COMPOSE end-to-end:
ingest sources -> derive top-k proofs (min_cost_k) -> introduce a
contradiction and surface the non-dominated readings (pareto_min skyline) ->
retract a source -> :reconcile the derived belief -> assert both the
valid-time and the transaction-time answers against a pure-Python oracle.

Deterministic and replayable: seeded LCG scenario, exact oracle, no
wall-clock dependence except the engine's own commit stamps (captured, not
assumed). See docs/TEMPORAL-BELIEF.md for methodology and the honesty notes
this workload enforces.
"""

__version__ = "0.1.0"
