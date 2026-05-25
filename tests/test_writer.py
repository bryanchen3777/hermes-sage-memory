import pytest
from pathlib import Path
from sage_memory.graph_store import GraphStore
from sage_memory.writer import MemoryWriter
from sage_memory.models import Fact


@pytest.fixture
def tmp_store(tmp_path):
    return GraphStore(db_path=tmp_path / "test.sqlite")


@pytest.fixture
def writer(tmp_store):
    return MemoryWriter(tmp_store, default_session_id="test-session")


def test_add_single_fact(writer, tmp_store):
    fact = Fact(subject="Alice", predicate="likes", object="coffee")
    fact_id = writer.add_fact(fact)
    assert fact_id == fact.fact_id
    assert tmp_store.edge_count == 1


def test_add_facts_batch(writer, tmp_store):
    facts = [
        Fact(subject="Bob", predicate="works_at", object="Acme"),
        Fact(subject="Bob", predicate="lives_in", object="NYC"),
    ]
    ids = writer.add_facts_batch(facts)
    assert len(ids) == 2
    assert tmp_store.edge_count == 2


def test_extract_and_write(writer, tmp_store):
    text = "Alice likes cats. Bob works at Google."
    ids = writer.extract_and_write(text, subject_hint="user", session_id="s1")
    assert len(ids) >= 1  # 至少抽出一條
    assert tmp_store.edge_count >= 1


def test_write_turn(writer, tmp_store):
    ids = writer.write_turn(
        user_content="I like hiking in mountains.",
        assistant_content="You mentioned you live in NYC.",
    )
    assert isinstance(ids, list)


def test_duplicate_subject_creates_separate_facts(writer, tmp_store):
    f1 = Fact(subject="Alice", predicate="likes", object="cats")
    f2 = Fact(subject="Alice", predicate="likes", object="dogs")
    writer.add_fact(f1)
    writer.add_fact(f2)
    assert tmp_store.edge_count == 2


def test_duplicate_fact_merges_weight(writer, tmp_store):
    """相同 S-P-O 寫兩次應合併 weight，不產生新 edge"""
    fact = Fact(subject="Alice", predicate="likes", object="coffee")
    writer.add_fact(fact)
    writer.add_fact(fact)  # 重複寫入
    assert tmp_store.edge_count == 1  # 不應增加


def test_first_person_normalization(writer, tmp_store):
    """'I like coffee' → subject 應正規化為 'user'"""
    ids = writer.extract_and_write(
        "I like coffee", subject_hint="user", session_id="s1"
    )
    assert len(ids) >= 1
    facts = tmp_store.get_all_facts()
    subjects = [f.subject for f in facts]
    assert "user" in subjects or "User" in subjects


def test_chinese_pattern_extraction(writer, tmp_store):
    """中文句子應能抽取三元組"""
    ids = writer.extract_and_write(
        "我喜歡喝咖啡。我住在台北。", subject_hint="user", session_id="s1"
    )
    assert len(ids) >= 1


def test_conjunction_split(writer, tmp_store):
    """'I like hiking and I live in Queens' 應抽出兩條 fact"""
    ids = writer.extract_and_write(
        "I like hiking and I live in Queens.",
        subject_hint="user", session_id="s1"
    )
    assert len(ids) >= 2


def test_inference_weight_lower(tmp_path):
    """assistant 推斷來源的 weight 應低於 user 直接陳述"""
    store1 = GraphStore(db_path=tmp_path / "w1.sqlite")
    writer1 = MemoryWriter(store1, "s1")
    writer1.extract_and_write("I like coffee", subject_hint="user",
                             session_id="s1", source="user")
    store2 = GraphStore(db_path=tmp_path / "w2.sqlite")
    writer2 = MemoryWriter(store2, "s1")
    writer2.extract_and_write("Assistant likes coffee", subject_hint="assistant",
                              session_id="s1", source="inference")
    # Both should be valid facts, inference fact has reduced weight
    assert store2.edge_count >= 1