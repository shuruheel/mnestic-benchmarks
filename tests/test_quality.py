"""Labelled-quality axis: metric definitions and query sanitization (no download)."""

from __future__ import annotations

import math

from hybrid_recall import quality


def test_ndcg_perfect_and_partial():
    rel = {"a": 1, "b": 1}
    assert quality.ndcg_at_k(["a", "b", "x"], rel) == 1.0
    # one relevant doc at rank 2 out of two relevant: (1/log2(3)) / (1 + 1/log2(3))
    got = quality.ndcg_at_k(["x", "a"], rel)
    want = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert abs(got - want) < 1e-12
    assert quality.ndcg_at_k(["x", "y"], rel) == 0.0


def test_graded_relevance_uses_gain():
    rel = {"a": 2, "b": 1}
    assert quality.ndcg_at_k(["a", "b"], rel) == 1.0
    assert quality.ndcg_at_k(["b", "a"], rel) < 1.0


def test_recall_at_n():
    assert quality.recall_at(["a", "x", "b"], {"a": 1, "b": 1, "c": 1}, n=2) == 1 / 3


def test_terms_strip_operators_and_punctuation():
    terms = quality._terms("Does NOT p53 (TP53) regulate AND/OR repair? p53!")
    assert terms == ["does", "p53", "tp53", "regulate", "repair"]


def test_dataset_is_pinned():
    spec = quality.DATASETS["scifact"]
    assert spec["url"].startswith("https://") and len(spec["sha256"]) == 64
