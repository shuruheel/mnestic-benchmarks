"""Adapter contract.

Each engine implements up to three *native* component retrievals over the chunk corpus:

    search_vector(spec)  -> ranked chunk ids by vector similarity
    search_fts(spec)     -> ranked chunk ids by full-text / BM25
    search_graph(spec)   -> ranked chunk ids by k-hop graph proximity to the seed entity

The base class fuses whatever the engine supports with the **canonical RRF**
(hybrid_recall.fusion.rrf_fuse) — identical math for every engine — so recall@k reflects
each engine's *retrieval fidelity*, not differences in fusion. An engine that cannot do a
signal (e.g. LanceDB has no graph) simply contributes no ranking for it; that gap shows up
honestly as lost recall and is recorded via `capabilities`.

`native_hybrid(spec)` is an OPTIONAL fast path some engines provide (mnestic's
`hybrid_search`, LanceDB's RRF reranker) that fuses vector+FTS in a single engine call. It
is measured and reported separately as a latency/ergonomics highlight — it is NOT used for
the scored recall comparison, to keep that comparison fair.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..fusion import rrf_fuse
from ..workload.schema import Workload, WorkloadMeta


@dataclass
class Capabilities:
    vector: bool
    fts: bool
    graph: bool
    native_fusion: bool          # can it fuse vector+FTS in a single engine call?
    fusion_locus: str            # "native" | "app-side"
    engines_needed: int          # distinct systems a real deployment needs for all 3 signals
    embedded: bool               # in-process, no server?
    transactional: bool = False  # writes (chunk + edges + indexes) commit atomically
    time_travel: bool = False    # query memory as-of a past point in time
    incremental_index: bool = False  # claims all indexes update on write (no rebuild)
    notes: str = ""


@dataclass
class IngestStats:
    rows: int
    seconds: float

    @property
    def rows_per_sec(self) -> float:
        return self.rows / self.seconds if self.seconds > 0 else 0.0


@dataclass
class BuildStats:
    seconds: float
    disk_bytes: int


@dataclass
class QuerySpec:
    text: str
    vector: np.ndarray
    seed_entity: str
    k: int
    pool_n: int
    hops: int
    rrf_k: float


@dataclass
class QueryResult:
    ids: list[str]
    round_trips: int = 0
    components: dict[str, list[str]] = field(default_factory=dict)


class Adapter(abc.ABC):
    name: str = "base"
    capabilities: Capabilities

    def __init__(self) -> None:
        self.version: str = "unknown"
        self.meta: WorkloadMeta | None = None

    # --- lifecycle (override) -------------------------------------------
    @abc.abstractmethod
    def setup(self, meta: WorkloadMeta, workdir: Path) -> None: ...

    @abc.abstractmethod
    def ingest(self, wl: Workload) -> IngestStats: ...

    @abc.abstractmethod
    def build_indexes(self) -> BuildStats: ...

    def teardown(self) -> None:  # noqa: B027  (optional)
        pass

    def upsert_memory(self, cid: str, text: str, vector, entity_ids: list[str]) -> None:
        """Insert a single new memory (chunk + its embedding + links to existing entities)
        the way an agent writes one at runtime. Used by the read-your-writes / freshness
        axis. Override per engine; default signals "not supported"."""
        raise NotImplementedError

    # --- native component retrievals (override what the engine supports) -
    def search_vector(self, spec: QuerySpec) -> list[str]:
        raise NotImplementedError

    def search_fts(self, spec: QuerySpec) -> list[str]:
        raise NotImplementedError

    def search_graph(self, spec: QuerySpec) -> list[str]:
        raise NotImplementedError

    def native_hybrid(self, spec: QuerySpec) -> list[str] | None:
        """Optional single-call vector+FTS fusion. Return None if unsupported."""
        return None

    # --- fused query (shared; do not override) --------------------------
    def query(self, spec: QuerySpec) -> QueryResult:
        rankings: list[list[str]] = []
        components: dict[str, list[str]] = {}
        round_trips = 0
        if self.capabilities.vector:
            v = self.search_vector(spec)
            components["vector"] = v
            rankings.append(v)
            round_trips += 1
        if self.capabilities.fts:
            f = self.search_fts(spec)
            components["fts"] = f
            rankings.append(f)
            round_trips += 1
        if self.capabilities.graph:
            g = self.search_graph(spec)
            components["graph"] = g
            rankings.append(g)
            round_trips += 1
        fused = rrf_fuse(rankings, k=spec.k, rrf_k=spec.rrf_k)
        return QueryResult(ids=fused, round_trips=round_trips, components=components)


def dir_size_bytes(path: Path) -> int:
    """Recursive on-disk size in bytes (file or directory)."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total
