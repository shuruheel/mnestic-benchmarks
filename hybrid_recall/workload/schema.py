"""Data model for the agentic-memory hybrid-recall workload.

A workload is a synthetic knowledge graph plus a corpus of "memory chunks":

  - Entity : a typed graph node (Person/Concept/Event/Org) with a label, a short
             text summary, and an embedding.
  - Edge   : a typed directed relation between two entities (the graph structure).
  - Chunk  : a "memory" document (the recall corpus) with text, an embedding, and
             links to >=1 entity (provenance). Recall is measured over chunks.
  - Query  : a recall request = (text, embedding, seed_entity). The task is to
             return the top-k chunks under a 3-signal RRF fusion of:
               1. vector kNN(embedding)  2. BM25(text)  3. graph proximity to seed.

Embeddings live in contiguous float32 matrices (entity_emb, chunk_emb, query_emb);
Entity/Chunk/Query carry an integer row index into the relevant matrix. This keeps
large scales memory- and IO-efficient (vectors as .npy, everything else as jsonl).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ENTITY_TYPES = ("Person", "Concept", "Event", "Org")
EDGE_TYPES = ("relates_to", "part_of", "caused_by", "mentions", "succeeds")


@dataclass(slots=True)
class Entity:
    id: str
    etype: str
    label: str
    summary: str
    cluster: int
    vec_idx: int


@dataclass(slots=True)
class Edge:
    src: str
    dst: str
    rel: str


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    entity_ids: list[str]
    cluster: int
    vec_idx: int


@dataclass(slots=True)
class Query:
    id: str
    text: str
    seed_entity: str
    cluster: int
    vec_idx: int


@dataclass(slots=True)
class WorkloadMeta:
    name: str            # scale name, e.g. "small"
    seed: int
    dim: int
    k: int               # top-k cutoff for recall@k and result sets
    pool_n: int          # per-signal candidate depth fetched before fusion
    graph_hops: int      # k for k-hop graph proximity
    rrf_k: float         # RRF constant (rank smoothing)
    clusters: int
    n_entities: int
    n_edges: int
    n_chunks: int
    n_queries: int
    n_incremental: int = 0   # held-out memories for the read-your-writes / freshness axis
    real_embeddings: bool = False


@dataclass(slots=True)
class Workload:
    meta: WorkloadMeta
    entities: list[Entity]
    edges: list[Edge]
    chunks: list[Chunk]
    queries: list[Query]
    entity_emb: np.ndarray   # (n_entities, dim) float32, row = Entity.vec_idx
    chunk_emb: np.ndarray    # (n_chunks, dim)   float32, row = Chunk.vec_idx
    query_emb: np.ndarray    # (n_queries, dim)  float32, row = Query.vec_idx

    # Held-out incremental memories for the read-your-writes / freshness axis. Each
    # inc_query is a probe that should return its inc_chunk once the chunk is upserted
    # and synchronously indexed. inc_chunks link to existing entities (so the graph
    # signal applies). Not part of the base corpus or the recall@k ground truth.
    inc_chunks: list[Chunk] = field(default_factory=list)
    inc_queries: list[Query] = field(default_factory=list)
    inc_chunk_emb: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    inc_query_emb: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))

    # Derived lookups (built lazily by ensure_index()).
    _entity_by_id: dict[str, Entity] = field(default_factory=dict, repr=False)
    _chunk_by_id: dict[str, Chunk] = field(default_factory=dict, repr=False)
    _adj: dict[str, list[str]] = field(default_factory=dict, repr=False)
    _chunks_by_entity: dict[str, list[str]] = field(default_factory=dict, repr=False)

    def ensure_index(self) -> None:
        if self._entity_by_id:
            return
        self._entity_by_id = {e.id: e for e in self.entities}
        self._chunk_by_id = {c.id: c for c in self.chunks}
        adj: dict[str, list[str]] = {e.id: [] for e in self.entities}
        for e in self.edges:
            adj.setdefault(e.src, []).append(e.dst)
            adj.setdefault(e.dst, []).append(e.src)  # undirected proximity
        self._adj = adj
        cbe: dict[str, list[str]] = {}
        for c in self.chunks:
            for eid in c.entity_ids:
                cbe.setdefault(eid, []).append(c.id)
        self._chunks_by_entity = cbe

    @property
    def adjacency(self) -> dict[str, list[str]]:
        self.ensure_index()
        return self._adj

    @property
    def chunks_by_entity(self) -> dict[str, list[str]]:
        self.ensure_index()
        return self._chunks_by_entity

    # --- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(asdict(self.meta), indent=2))
        np.save(d / "entity_emb.npy", self.entity_emb)
        np.save(d / "chunk_emb.npy", self.chunk_emb)
        np.save(d / "query_emb.npy", self.query_emb)
        _dump_jsonl(d / "entities.jsonl", (asdict(e) for e in self.entities))
        _dump_jsonl(d / "edges.jsonl", (asdict(e) for e in self.edges))
        _dump_jsonl(d / "chunks.jsonl", (asdict(c) for c in self.chunks))
        _dump_jsonl(d / "queries.jsonl", (asdict(q) for q in self.queries))
        if self.inc_chunks:
            _dump_jsonl(d / "inc_chunks.jsonl", (asdict(c) for c in self.inc_chunks))
            _dump_jsonl(d / "inc_queries.jsonl", (asdict(q) for q in self.inc_queries))
            np.save(d / "inc_chunk_emb.npy", self.inc_chunk_emb)
            np.save(d / "inc_query_emb.npy", self.inc_query_emb)

    @classmethod
    def load(cls, path: str | Path) -> "Workload":
        d = Path(path)
        meta = WorkloadMeta(**json.loads((d / "meta.json").read_text()))
        wl = cls(
            meta=meta,
            entities=[Entity(**r) for r in _load_jsonl(d / "entities.jsonl")],
            edges=[Edge(**r) for r in _load_jsonl(d / "edges.jsonl")],
            chunks=[Chunk(**r) for r in _load_jsonl(d / "chunks.jsonl")],
            queries=[Query(**r) for r in _load_jsonl(d / "queries.jsonl")],
            entity_emb=np.load(d / "entity_emb.npy"),
            chunk_emb=np.load(d / "chunk_emb.npy"),
            query_emb=np.load(d / "query_emb.npy"),
        )
        if (d / "inc_chunks.jsonl").exists():
            wl.inc_chunks = [Chunk(**r) for r in _load_jsonl(d / "inc_chunks.jsonl")]
            wl.inc_queries = [Query(**r) for r in _load_jsonl(d / "inc_queries.jsonl")]
            wl.inc_chunk_emb = np.load(d / "inc_chunk_emb.npy")
            wl.inc_query_emb = np.load(d / "inc_query_emb.npy")
        return wl


def _dump_jsonl(path: Path, rows) -> None:
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")))
            fh.write("\n")


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]
