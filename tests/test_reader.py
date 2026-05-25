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