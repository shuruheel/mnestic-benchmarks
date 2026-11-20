# Harness correctness evidence

These are tiny smoke runs, not release performance measurements. They include
unsuccessful harness iterations and completed candidate checks. `e2-v3` still
used numeric generation-order FTS ties; `e2-v4` corrects them to primary-key order.
The candidate runs have no query errors, ineligible keys, duplicate keys or
exact-vector/FTS oracle mismatches. Approximate vector recall is an observation,
not asserted equal to exact recall.

The earlier smoke manifests recorded the planned small-campaign counts (20 E2
queries and 100 E3 pairs); actual tiny runs use two. The scale and raw records
identify that distinction. The final harness emits the actual per-scale counts
and distinguishes direct uncapped calls from supervised runs. No performance claim
is derived from any smoke timing, including runs concurrent with compilation.
