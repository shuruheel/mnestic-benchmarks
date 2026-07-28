"""Pure-Python oracle for the temporal-belief workload.

Computes, with no database anywhere near it, the exact expected answers for
every phase: current claims, top-k proof costs, the Pareto-non-dominated
contested readings, and the post-retraction state. The engine must match
these row-for-row; sharing nothing with the engine but the scenario is what
makes the assertion worth something.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Deterministic scenario
# ---------------------------------------------------------------------------


class Lcg:
    """The same LCG on both sides of every assertion (and none of Python's
    ambient `random` state)."""

    def __init__(self, seed: int = 0x2545F4914F6CDD1D):
        self.state = seed

    def next(self, bound: int) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (self.state >> 33) % bound


@dataclass
class Claim:
    subj: str
    src: str
    val: int
    conf: float  # in (0, 1]


@dataclass
class Scenario:
    sources: Dict[str, float]  # src -> trust in (0, 1]
    claims: List[Claim]  # phase-1 ingest
    contradictor: str  # the phase-3 source
    contradictions: List[Claim]  # phase-3 conflicting claims
    subjects: List[str] = field(default_factory=list)


def build_scenario(n_subjects: int = 200, n_sources: int = 8, seed: int | None = None) -> Scenario:
    rng = Lcg() if seed is None else Lcg(seed)
    sources = {f"s{i}": 0.5 + 0.05 * (rng.next(10)) for i in range(n_sources)}
    subjects = [f"subj{i:04}" for i in range(n_subjects)]
    claims = []
    for subj in subjects:
        # 1-3 sources claim a value for each subject; values mostly agree.
        n = 1 + rng.next(3)
        base_val = rng.next(100)
        picked: Set[str] = set()
        for _ in range(n):
            src = f"s{rng.next(n_sources)}"
            if src in picked:
                continue
            picked.add(src)
            conf = 0.5 + rng.next(50) / 100.0
            claims.append(Claim(subj, src, base_val, round(conf, 2)))
    # The contradictor: a NEW source (max trust — it must be able to contest)
    # asserting a DIFFERENT value for every 10th subject.
    contradictor = "s_contra"
    sources[contradictor] = 0.95
    contradictions = [
        Claim(subj, contradictor, 1000 + i, 0.9)
        for i, subj in enumerate(subjects)
        if i % 10 == 0
    ]
    return Scenario(sources, claims, contradictor, contradictions, subjects)


# ---------------------------------------------------------------------------
# Expected answers
# ---------------------------------------------------------------------------


def proof_cost(trust: float, conf: float) -> float:
    return -math.log(trust) - math.log(conf)


def top_k_proofs(
    scenario: Scenario, live_claims: List[Claim], k: int = 3
) -> Dict[Tuple[str, int], List[float]]:
    """(subj, val) -> the k cheapest proof costs, ascending. Proof identity is
    the [src, subj, val] pack — a USER CONVENTION carried in the payload, which
    is the honesty note: min_cost_k gives programmable proof-carrying
    aggregation, not automatic lineage."""
    by_reading: Dict[Tuple[str, int], List[float]] = {}
    for c in live_claims:
        cost = proof_cost(scenario.sources[c.src], c.conf)
        by_reading.setdefault((c.subj, c.val), []).append(cost)
    return {rv: sorted(cs)[:k] for rv, cs in by_reading.items()}


def source_rank(scenario: Scenario) -> Dict[str, int]:
    """Trust rank (0 = most trusted); ties broken by name for determinism —
    the same tie-break the engine-side query encodes."""
    return {
        src: i
        for i, (src, _) in enumerate(
            sorted(scenario.sources.items(), key=lambda kv: (-kv[1], kv[0]))
        )
    }


def contested_readings(
    scenario: Scenario, live_claims: List[Claim]
) -> Dict[str, Set[int]]:
    """Per subject: the Pareto-non-dominated readings over the vector
    [best_proof_cost, best_src_rank] — componentwise minima over the reading's
    claims, exactly what the engine-side `min(c), min(r)` head aggregates
    compute. A reading survives unless another reading is <= on every axis and
    != (strictly better somewhere). Ties survive together — a contested
    subject returns SEVERAL readings instead of silently merging them."""
    ranks = source_rank(scenario)
    cand: Dict[str, Dict[int, List[float]]] = {}
    for c in live_claims:
        cost = proof_cost(scenario.sources[c.src], c.conf)
        rank = float(ranks[c.src])
        cur = cand.setdefault(c.subj, {}).setdefault(c.val, [math.inf, math.inf])
        cur[0] = min(cur[0], cost)
        cur[1] = min(cur[1], rank)
    out: Dict[str, Set[int]] = {}
    for subj, readings in cand.items():
        surv = set()
        for val, vec in readings.items():
            dominated = any(
                all(o[i] <= vec[i] for i in range(2)) and o != vec
                for v2, o in readings.items()
                if v2 != val
            )
            if not dominated:
                surv.add(val)
        out[subj] = surv
    return out


def live_claims(scenario: Scenario) -> List[Claim]:
    """Phase-1/2 live set: the base claims, before the contradictor exists."""
    return scenario.claims


def live_after_retraction(scenario: Scenario) -> List[Claim]:
    return scenario.claims  # the contradictor's claims are gone


def live_after_contradiction(scenario: Scenario) -> List[Claim]:
    return scenario.claims + scenario.contradictions
