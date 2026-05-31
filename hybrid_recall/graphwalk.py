"""Shared k-hop BFS that reproduces the ground-truth graph signal exactly.

Engines that store the graph relationally (mnestic, SQLite, DuckDB) drive this with two
tiny per-engine primitives:

    neighbors(frontier)  -> entity ids adjacent (either direction) to any id in frontier
    chunks_of(entities)  -> chunk ids linked to any of those entities

The walk assigns each entity its minimum hop distance from the seed (0..hops) and ranks
chunks by (min linking-entity hop, chunk id) — identical to
workload.groundtruth.graph_proximity_topn — so a correct relational engine matches the
oracle's graph component, and the only recall loss comes from the *other* signals' index
approximations. Graph-native engines (Kuzu) may instead express this as one Cypher query.
"""

from __future__ import annotations

from collections.abc import Callable


def bfs_chunk_ranking(
    neighbors: Callable[[list[str]], list[str]],
    chunks_of: Callable[[list[str]], list[str]],
    seed: str,
    hops: int,
    pool_n: int,
) -> list[str]:
    dist: dict[str, int] = {seed: 0}
    frontier: list[str] = [seed]
    for d in range(1, hops + 1):
        nxt = neighbors(frontier)
        new = [n for n in nxt if n not in dist]
        if not new:
            break
        for n in new:
            dist[n] = d
        frontier = new

    chunk_hop: dict[str, int] = {}
    for d in range(0, hops + 1):
        ents_d = [e for e, dd in dist.items() if dd == d]
        if not ents_d:
            continue
        for cid in chunks_of(ents_d):
            if cid not in chunk_hop:
                chunk_hop[cid] = d
    ranked = sorted(chunk_hop.items(), key=lambda kv: (kv[1], kv[0]))
    return [cid for cid, _ in ranked[:pool_n]]
