"""The engine-side temporal-belief workload: five phases, each timed, each
asserted against `oracle.py`.

Phases (docs/TEMPORAL-BELIEF.md):
  1. ingest      — sources + bitemporal claims (valid-time split VT_A/VT_B)
  2. proofs      — top-k proof chains per reading via `min_cost_k`
  3. contradict  — a new max-trust source contests every 10th subject;
                   the contested readings surface via a `pareto_min` skyline;
                   the derived `belief` relation is `:reconcile`d from it
  4. retract     — the contradictor's claims are valid-time retracted;
                   skyline re-derived, `belief` re-`:reconcile`d
  5. assert      — valid-time answers (@ VT instants), transaction-time
                   answers (`:as_of` captured instants on the DERIVED belief),
                   proofs, and both skylines — all row-exact vs the oracle

Determinism: seeded scenario; the only wall-clock inputs are the engine's own
commit stamps, and those are *captured between phases* and passed back as
`:as_of` parameters, never assumed.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from temporal_belief import oracle

VT_A = 1_000_000  # even subjects' claims valid from here (logical clock)
VT_B = 2_000_000  # odd subjects' claims valid from here
VT_MID = 1_500_000  # between the two: sees only the VT_A half
VT_LATE = 3_000_000  # after both
VT_RETRACT = 4_000_000  # the contradictor's claims retracted from here on

PROOF_K = 3

PROOFS_SCRIPT = """
proof[subj, val, min_cost_k(pack, 3)] :=
    *claim{subj, src, val, conf @ 'NOW'},
    *source{src, trust},
    pack = [[src, subj, val], -ln(trust) - ln(conf)]
?[subj, val, pack] := proof[subj, val, pack]
"""

SKYLINE_SCRIPT = """
cand[subj, val, min(c), min(r)] :=
    *claim{subj, src, val, conf @ 'NOW'},
    *source{src, trust},
    *rank{src, r},
    c = -ln(trust) - ln(conf)
sky[subj, pareto_min(v)] := cand[subj, val, c, r], v = [c, to_float(r)]
?[subj, val] := sky[subj, v], cand[subj, val, c, r], v == [c, to_float(r)]
"""


class Workload:
    def __init__(self, db, scenario: oracle.Scenario | None = None):
        self.db = db
        self.scenario = scenario or oracle.build_scenario()
        self.timings: Dict[str, float] = {}
        self.instants: Dict[str, int] = {}
        self.failures: List[str] = []

    # -- engine plumbing ----------------------------------------------------

    def _run(self, script: str, params: dict | None = None) -> List[list]:
        out = self.db.run_script(script, params or {}, False)
        return out["rows"]

    def _mark(self, name: str) -> None:
        """Capture a tt instant strictly after everything committed so far."""
        time.sleep(0.005)
        self.instants[name] = time.time_ns() // 1_000
        time.sleep(0.005)

    def _timed(self, name: str, fn) -> Any:
        t0 = time.perf_counter()
        out = fn()
        self.timings[name] = time.perf_counter() - t0
        return out

    def _check(self, what: str, got, want) -> None:
        if got != want:
            self.failures.append(
                f"{what}: engine and oracle disagree\n  engine: {got!r}\n  oracle: {want!r}"
            )

    # -- phases -------------------------------------------------------------

    def phase_ingest(self) -> None:
        s = self.scenario
        self._run(
            ":create source {src: String, tt: TxTime => trust: Float}"
        )
        self._run(
            ":create claim {subj: String, src: String, vt: Validity, tt: TxTime "
            "=> val: Int, conf: Float}"
        )
        self._run(":create rank {src: String => r: Int}")
        self._run(":create belief {subj: String, val: Int, tt: TxTime}")

        def go():
            rows = [[src, trust] for src, trust in sorted(s.sources.items()) if src != s.contradictor]
            self._run(
                "?[src, trust] <- $rows :put source {src => trust}", {"rows": rows}
            )
            # rank covers every source incl. the contradictor (data, not belief)
            ranks = oracle.source_rank(s)
            self._run(
                "?[src, r] <- $rows :put rank {src => r}",
                {"rows": [[src, r] for src, r in sorted(ranks.items())]},
            )
            claim_rows = []
            for c in s.claims:
                vt = VT_A if int(c.subj[4:]) % 2 == 0 else VT_B
                claim_rows.append([c.subj, c.src, [vt, True], c.val, c.conf])
            for i in range(0, len(claim_rows), 4000):
                self._run(
                    "?[subj, src, vt, val, conf] <- $rows "
                    ":put claim {subj, src, vt => val, conf}",
                    {"rows": claim_rows[i : i + 4000]},
                )

        self._timed("ingest", go)
        self._mark("t1_ingested")

    def phase_proofs(self) -> Dict:
        rows = self._timed("proofs", lambda: self._run(PROOFS_SCRIPT))
        # min_cost_k is a multi-row aggregate: one row per surviving pack
        # ([payload, cost]), so proofs of one reading arrive as sibling rows.
        got: Dict[tuple, List[float]] = {}
        for r in rows:
            got.setdefault((r[0], r[1]), []).append(round(r[2][1], 9))
        got = {rv: sorted(cs) for rv, cs in got.items()}
        want_raw = oracle.top_k_proofs(
            self.scenario, oracle.live_claims(self.scenario), k=PROOF_K
        )
        want = {rv: [round(c, 9) for c in cs] for rv, cs in want_raw.items()}
        self._check("phase-2 top-k proof costs", got, want)
        return got

    def _skyline(self) -> Dict[str, set]:
        rows = self._run(SKYLINE_SCRIPT)
        out: Dict[str, set] = {}
        for subj, val in rows:
            out.setdefault(subj, set()).add(val)
        return out

    def _reconcile_belief(self) -> None:
        self._run(
            SKYLINE_SCRIPT.replace(
                "?[subj, val] := sky[subj, v], cand[subj, val, c, r], v == [c, to_float(r)]",
                "?[subj, val] := sky[subj, v], cand[subj, val, c, r], v == [c, to_float(r)]\n"
                ":reconcile belief {subj, val}",
            )
        )

    def phase_contradict(self) -> None:
        s = self.scenario

        def go():
            self._run(
                "?[src, trust] <- $rows :put source {src => trust}",
                {"rows": [[s.contradictor, s.sources[s.contradictor]]]},
            )
            self._run(
                "?[subj, src, vt, val, conf] <- $rows "
                ":put claim {subj, src, vt => val, conf}",
                {
                    "rows": [
                        [c.subj, c.src, [VT_A, True], c.val, c.conf]
                        for c in s.contradictions
                    ]
                },
            )

        self._timed("contradict_ingest", go)
        sky = self._timed("skyline_contested", self._skyline)
        want = oracle.contested_readings(s, oracle.live_after_contradiction(s))
        self._check("phase-3 contested skyline", sky, want)
        self._timed("reconcile_contested", self._reconcile_belief)
        self._mark("t2_contested")

    def phase_retract(self) -> None:
        s = self.scenario

        def go():
            # Valid-time retraction: assert-false rows from VT_RETRACT on.
            self._run(
                "?[subj, src, vt, val, conf] <- $rows "
                ":put claim {subj, src, vt => val, conf}",
                {
                    "rows": [
                        [c.subj, c.src, [VT_RETRACT, False], c.val, c.conf]
                        for c in s.contradictions
                    ]
                },
            )

        self._timed("retract", go)
        sky = self._timed("skyline_settled", self._skyline)
        want = oracle.contested_readings(s, oracle.live_after_retraction(s))
        self._check("phase-4 post-retraction skyline", sky, want)
        self._timed("reconcile_settled", self._reconcile_belief)
        self._mark("t3_settled")

    def phase_assert_axes(self) -> None:
        s = self.scenario
        # ---- valid time: @ VT_MID sees only the VT_A half of the base claims
        rows = self._run(
            "?[count(subj)] := *claim{subj, src @ $vt}", {"vt": VT_MID}
        )
        # The VT_A half of the base claims — PLUS the contradictor's claims,
        # which also assert from VT_A and whose later retraction sits at
        # VT_RETRACT on the valid axis: at VT_MID they WERE valid, and the
        # current belief about that period still says so. (Valid time is
        # history, not state; the first draft of this oracle forgot that and
        # the engine corrected it.)
        want_mid = len(
            {(c.subj, c.src) for c in s.claims if int(c.subj[4:]) % 2 == 0}
            | {(c.subj, c.src) for c in s.contradictions}
        )
        self._check("VT @ mid (VT_A half + not-yet-retracted claims)", rows[0][0], want_mid)
        # ---- valid time is history, not state: at VT_LATE (pre-retraction
        # instant on the valid axis) the contradictor's claims are STILL true,
        # even though the current belief has moved on.
        rows = self._run(
            "?[count(subj)] := *claim{subj, src @ $vt}, src == $contra",
            {"vt": VT_LATE, "contra": s.contradictor},
        )
        self._check(
            "VT @ late still shows the retracted source's claims",
            rows[0][0],
            len(s.contradictions),
        )
        rows = self._run(
            "?[count(subj)] := *claim{subj, src @ $vt}, src == $contra",
            {"vt": VT_RETRACT + 1, "contra": s.contradictor},
        )
        self._check("VT after the retraction instant shows none", rows[0][0], 0)

        # ---- transaction time, on the DERIVED belief: what did we believe?
        def belief_at(instant_us: int) -> Dict[str, set]:
            rows = self._run(
                "?[subj, val] := *belief{subj, val}\n:as_of $t",
                {"t": instant_us},
            )
            out: Dict[str, set] = {}
            for subj, val in rows:
                out.setdefault(subj, set()).add(val)
            return out

        want_contested = oracle.contested_readings(
            s, oracle.live_after_contradiction(s)
        )
        want_settled = oracle.contested_readings(s, oracle.live_after_retraction(s))
        self._check(
            "TT :as_of t2 — the belief DURING the contest",
            belief_at(self.instants["t2_contested"]),
            want_contested,
        )
        self._check(
            "TT :as_of t3 — the settled belief after retract + reconcile",
            belief_at(self.instants["t3_settled"]),
            want_settled,
        )

    # -- driver -------------------------------------------------------------

    def run(self) -> Dict:
        self.phase_ingest()
        self.phase_proofs()
        self.phase_contradict()
        self.phase_retract()
        self._timed("assert_axes", self.phase_assert_axes)
        return {
            "ok": not self.failures,
            "failures": self.failures,
            "timings_s": {k: round(v, 4) for k, v in self.timings.items()},
            "scale": {
                "subjects": len(self.scenario.subjects),
                "claims": len(self.scenario.claims),
                "contradictions": len(self.scenario.contradictions),
                "sources": len(self.scenario.sources),
            },
        }
