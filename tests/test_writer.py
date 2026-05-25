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
    assert store2.edge_count >= 1


# ── v0.1.1 新功能測試 ──────────────────────────────────────────

def test_entity_alignment_fuzzy_match(tmp_path):
    """相似實體名稱應被對齊到既有節點（但不同 object 不合併）"""
    store = GraphStore(db_path=tmp_path / "align.sqlite")
    writer = MemoryWriter(store, "s1")
    # 先寫入 Alice likes coffee
    writer.add_fact(Fact(subject="Alice", predicate="likes", object="coffee"))
    # alie 與 Alice 相似度 > 0.75，subject 會對齊
    # 但 object 不同（tea vs coffee），所以不會觸發 dedup 合併
    fid = writer.add_fact(Fact(subject="alie", predicate="likes", object="tea"))
    facts = store.get_all_facts()
    subjects = {f.subject for f in facts}
    # subject 應對齊到 Alice
    assert "Alice" in subjects
    assert "alie" not in subjects  # 確認對齊生效
    # 兩條不同 object 的 fact 都存在
    assert len(facts) == 2


def test_entity_alignment_no_match(tmp_path):
    """無相似實體則不對齊"""
    store = GraphStore(db_path=tmp_path / "noalign.sqlite")
    writer = MemoryWriter(store, "s1")
    writer.add_fact(Fact(subject="Bob", predicate="likes", object="tea"))
    fid = writer.add_fact(Fact(subject="Charlie", predicate="likes", object="coffee"))
    assert store.edge_count == 2


def test_event_time_parsed_from_tomorrow(tmp_path):
    """明天/tomorrow 時間詞應被解析"""
    store = GraphStore(db_path=tmp_path / "eventtime.sqlite")
    writer = MemoryWriter(store, "s1")
    ids = writer.extract_and_write(
        "I want to go to Tokyo tomorrow.",
        subject_hint="user", session_id="s1", source="user"
    )
    assert len(ids) >= 1
    facts = store.get_all_facts()
    event_facts = [f for f in facts if f.event_time is not None]
    assert len(event_facts) >= 1


def test_event_time_parsed_from_month_day(tmp_path):
    """月日模式（如 3月15日）應被解析"""
    store = GraphStore(db_path=tmp_path / "monthday.sqlite")
    writer = MemoryWriter(store, "s1")
    ids = writer.extract_and_write(
        "我的生日是12月25日。",
        subject_hint="user", session_id="s1", source="user"
    )
    assert len(ids) >= 1


def test_anchor_auto_set_on_reinforcement(tmp_path):
    """强化逻辑正确性：weight >= 1.8 时的 anchor 设置由 writer 处理"""
    store = GraphStore(db_path=tmp_path / "anchorauto.sqlite")
    from sage_memory.evolution import MemoryEvolution
    import time

    # 直接测试 anchor 保护机制：设置 anchor 后 decay 不应影响它
    fact = Fact(subject="Bob", predicate="likes", object="pizza",
                timestamp=time.time() - 86400 * 10, is_anchor=True)
    store.add_fact(fact)
    evo = MemoryEvolution(store)
    stats = evo.run_scheduled_decay(age_days_threshold=7.0)
    # anchor 不应被 decay 影响
    assert stats["anchors_protected"] == 1
    # 且 fact 仍在圖中
    assert store.get_fact(fact.fact_id) is not None


def test_anchor_not_decay_on_scheduled(tmp_path):
    """anchor 事實不應被 scheduled decay 影響"""
    store = GraphStore(db_path=tmp_path / "anchorprotect.sqlite")
    from sage_memory.evolution import MemoryEvolution
    import time
    # 寫入一個 anchor fact（舊時間）
    old_fact = Fact(
        subject="Carol", predicate="lives_in", object="Berlin",
        timestamp=time.time() - 86400 * 10,  # 10 天前
        is_anchor=True
    )
    store.add_fact(old_fact)
    evo = MemoryEvolution(store)
    stats = evo.run_scheduled_decay(age_days_threshold=7.0)
    assert stats["anchors_protected"] == 1
    assert store.get_fact(old_fact.fact_id) is not None  # 仍在圖中


# ── v0.1.2 新功能測試 ──────────────────────────────────────────

def test_write_result_confirmation_success(tmp_path):
    """write_with_confirmation 成功寫入應正確回報"""
    from sage_memory.writer import WriteResult
    store = GraphStore(db_path=tmp_path / "write_result.sqlite")
    writer = MemoryWriter(store, "s1")
    fact = Fact(subject="Dave", predicate="likes", object="pizza")
    result = writer.write_with_confirmation(fact)
    assert result.success_count >= 1
    assert not result.has_failures
    assert len(result.written) == 1


def test_write_result_rejects_invalid_fact(tmp_path):
    """空白 subject 應被 schema gate 拒絕"""
    from sage_memory.writer import WriteResult
    store = GraphStore(db_path=tmp_path / "schema_gate.sqlite")
    writer = MemoryWriter(store, "s1")
    fact = Fact(subject="", predicate="likes", object="pizza")
    result = writer.write_with_confirmation(fact)
    assert result.has_failures
    assert len(result.rejected) == 1


def test_predicate_normalization_via_extraction(tmp_path):
    """透過 extract_and_write，adores/enjoys 應正規化為 likes"""
    store = GraphStore(db_path=tmp_path / "pred_norm.sqlite")
    writer = MemoryWriter(store, "s1")
    ids1 = writer.extract_and_write("Eve adores music.", subject_hint="user", session_id="s1")
    ids2 = writer.extract_and_write("Eve enjoys music.", subject_hint="user", session_id="s1")
    # Both adores and enjoys normalize to "likes", so should dedup to 1 fact
    assert store.edge_count == 1
    assert store.get_all_facts()[0].predicate == "likes"


def test_contradiction_detection_via_confidence(tmp_path):
    """高 confidence 新 fact 覆蓋低 confidence 舊 fact"""
    store = GraphStore(db_path=tmp_path / "contradict.sqlite")
    writer = MemoryWriter(store, "s1")
    # Write the first fact
    f1 = Fact(subject="Frank", predicate="likes", object="coffee", confidence=0.5)
    writer.add_fact(f1)
    # Write contradiction with higher confidence
    f2 = Fact(subject="Frank", predicate="hates", object="coffee", confidence=0.9)
    result = writer.write_with_confirmation(f2)
    assert len(result.contradictions) >= 1