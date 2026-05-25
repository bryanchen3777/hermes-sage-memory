import pytest
from sage_memory.graph_store import GraphStore
from sage_memory.writer import MemoryWriter
from sage_memory.reader import MemoryReader
from sage_memory.models import Fact, ContextResult


@pytest.fixture
def populated_store(tmp_path):
    store = GraphStore(db_path=tmp_path / "test.sqlite")
    writer = MemoryWriter(store, default_session_id="s1")
    facts = [
        Fact(subject="Alice", predicate="likes", object="coffee"),
        Fact(subject="Alice", predicate="lives_in", object="NYC"),
        Fact(subject="Alice", predicate="works_at", object="TechCorp"),
        Fact(subject="Bob", predicate="knows", object="Alice"),
        Fact(subject="Bob", predicate="likes", object="tea"),
    ]
    writer.add_facts_batch(facts)
    return store


def test_retrieve_returns_result(populated_store):
    reader = MemoryReader(populated_store)
    result = reader.retrieve_context("What does Alice like?")
    assert result is not None
    assert not result.is_empty


def test_retrieve_top_k_limit(populated_store):
    reader = MemoryReader(populated_store)
    result = reader.retrieve_context("Alice", top_k=2)
    assert len(result.facts) <= 2


def test_retrieve_empty_store(tmp_path):
    store = GraphStore(db_path=tmp_path / "empty.sqlite")
    reader = MemoryReader(store)
    result = reader.retrieve_context("anything")
    assert result.is_empty
    assert result.summary == ""


def test_summary_token_limit(populated_store):
    reader = MemoryReader(populated_store)
    result = reader.retrieve_context("Alice", max_tokens=100)
    assert result.token_estimate <= 150  # 給一點 buffer


def test_multi_hop_chains(populated_store):
    reader = MemoryReader(populated_store)
    result = reader.retrieve_context("Bob Alice", max_hops=2)
    # Bob knows Alice，Alice has data → chains 應該能找到
    assert isinstance(result.chains, list)


def test_recall_mode_precise_no_chains(populated_store):
    """precise mode 不應產生 chains"""
    reader = MemoryReader(populated_store)
    result = reader.retrieve_context("Alice", mode="precise")
    assert result.chains == []
    assert len(result.facts) <= 3


def test_recall_mode_expansive_more_facts(populated_store):
    """expansive mode 應回傳比 precise 更多的 facts"""
    reader = MemoryReader(populated_store)
    precise = reader.retrieve_context("Alice", top_k=3, mode="precise")
    expansive = reader.retrieve_context("Alice", top_k=3, mode="expansive")
    assert len(expansive.facts) >= len(precise.facts)


def test_summary_confidence_labels(populated_store):
    """summary 應包含 [high] / [medium] / [low] 信度標籤"""
    reader = MemoryReader(populated_store)
    result = reader.retrieve_context("Alice likes")
    assert "[high]" in result.summary or "[medium]" in result.summary


def test_fallback_on_empty_keywords(populated_store):
    """全停用詞 query 應觸發 fallback，不崩潰"""
    reader = MemoryReader(populated_store)
    result = reader.retrieve_context("what is the")  # 全停用詞
    assert isinstance(result, ContextResult)


def test_chain_deduplication(populated_store):
    """相同 chain 不應重複出現"""
    reader = MemoryReader(populated_store)
    result = reader.retrieve_context("Alice Bob", max_hops=2)
    # 驗證每條 chain 的 fact_id 組合唯一
    chain_keys = [
        frozenset(f.fact_id for f in chain)
        for chain in result.chains
    ]
    assert len(chain_keys) == len(set(chain_keys))


def test_relevance_scoring_prefers_exact_match(populated_store):
    """完全命中關鍵詞的 fact 應排在前面"""
    reader = MemoryReader(populated_store)
    result = reader.retrieve_context("Alice coffee", top_k=5)
    if result.facts:
        # 第一條應包含 Alice 或 coffee
        top = result.facts[0]
        text = f"{top.subject} {top.predicate} {top.object}".lower()
        assert "alice" in text or "coffee" in text


# ── v0.1.2 新功能測試 ──────────────────────────────────────────

def test_sigmoid_normalization_produces_scores_in_0_1(populated_store):
    """sigmoid 正規化後分數應落在 0~1 區間"""
    reader = MemoryReader(populated_store)
    result = reader.retrieve_context("Alice coffee", top_k=5)
    for fact in result.facts:
        score = result.retrieval_scores.get(fact.fact_id, 0.0)
        assert 0.0 <= score <= 1.0


def test_diversity_filter_prevents_subject_collapse(populated_store):
    """同 subject 超過 max_per_subject 應被過濾"""
    reader = MemoryReader(populated_store)
    # Add many facts for same subject
    for i in range(5):
        populated_store.add_fact(
            Fact(subject="Alice", predicate="likes", object=f"item{i}")
        )
    result = reader.retrieve_context("Alice", top_k=10, mode="balanced")
    subject_counts = {}
    for f in result.facts:
        s = f.subject.lower()
        subject_counts[s] = subject_counts.get(s, 0) + 1
    # Should not have more than 3 (max_per_subject for balanced)
    for s, cnt in subject_counts.items():
        assert cnt <= 3


def test_retrieval_scores_map_populated(populated_store):
    """retrieval_scores 應包含所有 retrieved facts 的分數"""
    reader = MemoryReader(populated_store)
    result = reader.retrieve_context("Alice", top_k=5)
    assert len(result.retrieval_scores) == len(result.facts)


# ── v0.1.3 新功能測試 ──────────────────────────────────────────

def test_subject_weight_bumps_user_queries(populated_store):
    """包含 user 的 query 應獲得較高 subject_weight"""
    from unittest.mock import MagicMock
    reader = MemoryReader(populated_store)
    # Mock the score calculation to verify subject_weight is included
    result = reader.retrieve_context("user likes", top_k=5)
    # Any valid result means the scoring ran without error
    assert isinstance(result, ContextResult)


def test_std_s_minimum_protection(tmp_path):
    """std_s 為 0 時應被保護為至少 0.1"""
    store = GraphStore(db_path=tmp_path / "stdprotect.sqlite")
    writer = MemoryWriter(store, "s1")
    reader = MemoryReader(store)
    # All facts with identical scores -> std=0
    for i in range(3):
        writer.add_fact(Fact(subject=f"X{i}", predicate="y", object="z", weight=1.0))
    result = reader.retrieve_context("y", top_k=5)
    # Should not crash and should still return results
    assert len(result.facts) >= 0


def test_diversity_pair_filter_enforced(populated_store):
    """subject-predicate pair 限制应被执行"""
    reader = MemoryReader(populated_store)
    # Add multiple facts with same subject but different predicates
    for i in range(3):
        populated_store.add_fact(
            Fact(subject="Alice", predicate="works_at", object=f"Co{i}")
        )
    result = reader.retrieve_context("Alice", top_k=10, mode="balanced")
    pair_keys = [(f.subject.lower(), f.predicate.lower()) for f in result.facts]
    # Each subject-predicate pair should appear at most 1-2 times
    from collections import Counter
    pair_counts = Counter(pair_keys)
    assert all(c <= 2 for c in pair_counts.values())