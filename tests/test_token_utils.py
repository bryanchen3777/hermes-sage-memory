"""Phase 7：Token 優化測試"""
from __future__ import annotations

import time
import pytest

from sage_memory.token_utils import TokenBudget, SummaryCompressor, PrefetchCache
from sage_memory.models import Fact, ContextResult


# ── TokenBudget ───────────────────────────────────────────────

def test_budget_consume_within_limit():
    b = TokenBudget(100)
    assert b.consume("hello world") is True
    assert b.remaining < 100


def test_budget_consume_exceeds_limit():
    b = TokenBudget(5)
    long_text = "a" * 200  # ~50 tokens
    assert b.consume(long_text) is False


def test_budget_reset():
    b = TokenBudget(100)
    b.consume("some text here")
    b.reset()
    assert b.remaining == 100


def test_budget_remaining_never_negative():
    b = TokenBudget(10)
    b.consume("a" * 1000)  # 超出後 remaining 應為 0
    assert b.remaining >= 0


# ── SummaryCompressor ─────────────────────────────────────────

@pytest.fixture
def sample_facts():
    return [
        Fact(subject="Alice", predicate="likes",    object="coffee",   weight=0.9),
        Fact(subject="Alice", predicate="lives_in", object="NYC",      weight=0.85),
        Fact(subject="Alice", predicate="works_at", object="TechCorp", weight=0.8),
        Fact(subject="Bob",   predicate="knows",    object="Alice",    weight=0.7),
        Fact(subject="Bob",   predicate="likes",    object="tea",      weight=0.6),
    ]


@pytest.fixture
def sample_result(sample_facts):
    chain = [
        Fact(subject="Bob",   predicate="knows",    object="Alice"),
        Fact(subject="Alice", predicate="works_at", object="TechCorp"),
    ]
    return ContextResult(
        facts=sample_facts,
        chains=[chain],
        summary="",
        token_estimate=0,
    )


def test_compressor_groups_by_subject(sample_result):
    """相同 subject 的 facts 應合併為一行"""
    c = SummaryCompressor()
    b = TokenBudget(800)
    summary = c.compress(sample_result, b)
    lines = summary.split("\n")
    # Alice 作為 subject 的事實應在同一行（不包含 chain 中作為 object 的 Alice）
    alice_subj_lines = [l for l in lines if l.startswith("- Alice:")]
    assert len(alice_subj_lines) == 1
    assert "likes" in alice_subj_lines[0]
    assert "lives_in" in alice_subj_lines[0]


def test_compressor_strongest_chain_only(sample_result):
    """chains 只輸出一條"""
    c = SummaryCompressor()
    b = TokenBudget(800)
    summary = c.compress(sample_result, b)
    chain_count = summary.count("Key Chain")
    assert chain_count <= 1


def test_compressor_respects_token_budget(sample_result):
    """嚴格 token 預算下輸出不超出"""
    c = SummaryCompressor()
    b = TokenBudget(30)   # 非常小的預算
    summary = c.compress(sample_result, b)
    actual_tokens = len(summary) // 4
    assert actual_tokens <= 35   # 給 5 token buffer


def test_compressor_empty_result():
    """空 result 應回傳空字串"""
    c = SummaryCompressor()
    b = TokenBudget(800)
    result = ContextResult(facts=[], chains=[], summary="", token_estimate=0)
    assert c.compress(result, b) == ""


def test_estimate_tokens_reasonable(sample_result):
    """estimate_tokens 應在合理範圍內"""
    c = SummaryCompressor()
    est = c.estimate_tokens(sample_result)
    assert 5 < est < 500


# ── PrefetchCache ─────────────────────────────────────────────

def test_cache_hit():
    cache = PrefetchCache(ttl_seconds=60)
    cache.set("what does Alice like", "Alice likes coffee")
    result = cache.get("what does Alice like")
    assert result == "Alice likes coffee"


def test_cache_miss():
    cache = PrefetchCache(ttl_seconds=60)
    assert cache.get("unknown query") is None


def test_cache_ttl_expiry():
    cache = PrefetchCache(ttl_seconds=0.05)  # 50ms TTL
    cache.set("query", "summary")
    time.sleep(0.1)
    assert cache.get("query") is None  # 已過期


def test_cache_invalidate():
    cache = PrefetchCache(ttl_seconds=60)
    cache.set("q1", "s1")
    cache.set("q2", "s2")
    cache.invalidate()
    assert cache.size == 0


def test_cache_max_size_eviction():
    cache = PrefetchCache(ttl_seconds=60, max_size=3)
    cache.set("q1", "s1")
    cache.set("q2", "s2")
    cache.set("q3", "s3")
    cache.set("q4", "s4")   # 應觸發 eviction
    assert cache.size == 3


def test_cache_case_insensitive_key():
    """大小寫不同的相同 query 應命中同一快取"""
    cache = PrefetchCache(ttl_seconds=60)
    cache.set("What does Alice like", "result")
    assert cache.get("what does alice like") == "result"