"""Workload, ground-truth, and fusion correctness — no engine dependencies."""

from __future__ import annotations

import tempfile
from pathlib import Path

from hybrid_recall.fusion import rrf_fuse
from hybrid_recall.metrics import recall_at_k
from hybrid_recall.workload import generate as g
from hybrid_recall.workload.groundtruth import compute_ground_truth
from hybrid_recall.workload.schema import Workload

TINY = dict(
    entities=300, edges=1500, chunks=1200, dim=48, queries=60, graph_hops=2, k=10, clusters=12
)


def _gen(seed=7):
    return g.generate("tiny", seed=seed, **TINY)


def test_generation_deterministic():
    wl1, wl2 = _gen(), _gen()
    gt1, gt2 = compute_ground_truth(wl1), compute_ground_truth(wl2)
    assert gt1 == gt2
    assert all(len(v) == TINY["k"] for v in gt1.values())


def test_save_load_roundtrip():
    wl = _gen()
    gt = compute_ground_truth(wl)
    with tempfile.TemporaryDirectory() as d:
        wl.save(d)
        wl2 = Workload.load(Path(d))
    assert compute_ground_truth(wl2) == gt


def test_signals_correlated_but_distinct():
    """Vector~FTS should agree more than either agrees with graph (the regime fusion needs)."""
    from hybrid_recall.workload.groundtruth import (
        _BM25Index,
        exact_vector_topn,
        graph_proximity_topn,
    )

    wl = _gen()
    wl.ensure_index()
    bm = _BM25Index([c.text for c in wl.chunks])
    cid = [c.id for c in wl.chunks]

    def jac(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if a | b else 1.0

    vb = vg = 0.0
    n = 40
    for q in wl.queries[:n]:
        v = [cid[i] for i in exact_vector_topn(wl.query_emb[q.vec_idx], wl.chunk_emb, 20)]
        b = [cid[i] for i in bm.top(q.text.split(), 20)]
        gr = graph_proximity_topn(wl, q.seed_entity, 2, 20)
        vb += jac(v, b)
        vg += jac(v, gr)
    assert vb / n > vg / n  # content signals reinforce; graph is distinct


def test_rrf_fuse_basic():
    # an item ranked highly in two lists should beat one ranked once
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=3)
    assert fused[0] in {"a", "b"}
    assert set(fused) <= {"a", "b", "c", "d"}


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a", "b", "x"], 3) == 2 / 3
    assert recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    assert recall_at_k(["x"], ["a"], 1) == 0.0


# --- vocabulary space (regression: `medium` and `large` could never be generated) -----------
#
# `_make_vocab` rejection-samples DISTINCT pseudo-words, so it can only terminate if the
# requested vocabulary fits inside the syllable alphabet's word space. With 36 syllables and
# words of 2-3 syllables that space is 36**2 + 36**3 = 47,952 -- but `medium` asks for
# clusters(256) * topic_size(200) = 51,200 and `large` for 204,800. Both spun forever, which
# is why no medium/large result had ever been produced. These guard the invariant.


def _vocab_size_for(clusters: int, topic_size: int = 200) -> int:
    """Mirror of the sizing rule in generate.generate()."""
    return max(2000, clusters * topic_size)


def test_vocab_space_covers_every_configured_scale():
    """Every scale in configs/scales.yaml must be *generatable*: the word space the sampler can
    draw from has to be at least as large as the vocabulary it is asked to produce."""
    import yaml

    scales = yaml.safe_load((Path(__file__).parent.parent / "configs" / "scales.yaml").read_text())
    syllables = len(g._SYLL)
    for name, params in scales.items():
        need = _vocab_size_for(params["clusters"])
        max_syll = g._max_syllables(need)
        space = sum(syllables**i for i in range(2, max_syll + 1))
        assert space >= need, (
            f"scale {name!r}: vocabulary of {need:,} exceeds the {space:,}-word space "
            f"(<= {max_syll} syllables) -- _make_vocab could never terminate"
        )


def test_make_vocab_terminates_and_is_unique_at_medium_scale():
    """The production artifact: `medium` needs 51,200 distinct words. Before the fix this call
    never returned."""
    import numpy as np

    need = _vocab_size_for(256)  # medium: 256 clusters x 200 topic words
    vocab = g._make_vocab(np.random.default_rng(1234), need)
    assert len(vocab) == need
    assert len(set(vocab)) == need


def test_small_and_tiny_vocab_unchanged():
    """Widening the word space must not perturb the scales that already worked -- the published
    reference results must stay reproducible. These digests pin the exact draw sequence."""
    import hashlib

    import numpy as np

    expected = {
        3200: "7ab0930b605112d5",   # tiny  (16 clusters -> max(2000, 3200))
        12800: "b0784cfefacf5c60",  # small (64 clusters -> 12,800)
    }
    for size, digest in expected.items():
        vocab = g._make_vocab(np.random.default_rng(1234), size)
        got = hashlib.sha256("|".join(vocab).encode()).hexdigest()[:16]
        assert got == digest, f"vocab draw sequence changed at size {size}: {got} != {digest}"
