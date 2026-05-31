"""Per-engine adapters. Import lazily so a missing optional dependency only fails the
engine that needs it, never the whole harness."""

from __future__ import annotations

# name -> "module:ClassName"
REGISTRY: dict[str, str] = {
    "mnestic": "hybrid_recall.adapters.mnestic_adapter:MnesticAdapter",
    "sqlite": "hybrid_recall.adapters.sqlite_adapter:SQLiteAdapter",
    "duckdb": "hybrid_recall.adapters.duckdb_adapter:DuckDBAdapter",
    "kuzu": "hybrid_recall.adapters.kuzu_adapter:KuzuAdapter",
    "lancedb": "hybrid_recall.adapters.lancedb_adapter:LanceDBAdapter",
    # phase 2 (server engines, gated behind the [server] extra + docker)
    "neo4j": "hybrid_recall.adapters.neo4j_adapter:Neo4jAdapter",
    "qdrant": "hybrid_recall.adapters.qdrant_adapter:QdrantAdapter",
    "arcadedb": "hybrid_recall.adapters.arcadedb_adapter:ArcadeDBAdapter",
}

EMBEDDED = ("mnestic", "sqlite", "duckdb", "kuzu", "lancedb")
SERVER = ("neo4j", "qdrant", "arcadedb")


def load_adapter(name: str):
    """Import and return the adapter class for `name` (raises ImportError if its
    optional dependency is not installed)."""
    import importlib

    if name not in REGISTRY:
        raise KeyError(f"unknown engine '{name}'. known: {sorted(REGISTRY)}")
    module_path, cls_name = REGISTRY[name].split(":")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)
