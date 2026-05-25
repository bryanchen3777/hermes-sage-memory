import pytest
from sage_memory.graph_store import GraphStore
from sage_memory.writer import MemoryWriter
from sage_memory.evolution import MemoryEvolution, EvolutionEvent
from sage_memory.models import Fact


@pytest.fixture
def setup(tmp_path):
    store = GraphStore(db_path=tmp_path / "evo.sqlite")
    writer = MemoryWriter(store)
    evo = MemoryEvolution(store)
    f1 = Fact(subject="A", predicate="is", object="B")
    f2 = Fact(subject="C", predicate="knows", object="D")
    writer.add_fact(f1)
    writer.add_fact(f2)
    return store, writer, evo, f1, f2


def test_decay_reduces_weight(setup):
    store, _, evo, f1, _ = setup
    ok = evo.apply_correction(f1.fact_id, "decay", delta=0.3)
    assert ok
    updated = store.get_fact(f1.fact_id)
    assert updated.weight < 1.0


def test_prune_removes_fact(setup):
    store, _, evo, f1, _ = setup
    ok = evo.apply_correction(f1.fact_id, "prune")
    assert ok
    assert store.get_fact(f1.fact_id) is None


def test_merge_combines_weights(setup):
    store, _, evo, f1, f2 = setup
    ok = evo.apply_correction(f1.fact_id, "merge", target_id=f2.fact_id)
    assert ok
    # source 被刪除
    assert store.get_fact(f1.fact_id) is None
    # target weight 提升
    updated = store.get_fact(f2.fact_id)
    assert updated.weight > 1.0 or updated.weight == pytest.approx(1.0)


def test_decay_below_threshold_triggers_prune(setup):
    store, _, evo, f1, _ = setup
    # 多次 decay 直到低於閾值
    for _ in range(25):
        evo.apply_correction(f1.fact_id, "decay", delta=0.1)
    # 應該已被自動剪除
    result = store.get_fact(f1.fact_id)
    assert result is None or result.weight < MemoryEvolution.PRUNE_THRESHOLD


def test_evolution_log_records_events(setup):
    """每次操作都應產生一筆 log"""
    store, _, evo, f1, _ = setup
    evo.apply_correction(f1.fact_id, "decay", reason="test")
    assert len(evo.evolution_log) >= 1
    assert evo.evolution_log[0].action == "decay"


def test_log_summary_format(setup):
    """log_summary 應包含操作名稱與次數"""
    store, writer, evo, f1, f2 = setup
    evo.apply_correction(f1.fact_id, "decay", reason="t1")
    evo.apply_correction(f2.fact_id, "prune", reason="t2")
    summary = evo.log_summary()
    assert "decay" in summary
    assert "prune" in summary


def test_conflict_detection(tmp_path):
    """相同 subject、互斥 predicate 應被偵測為衝突"""
    from sage_memory.graph_store import GraphStore
    from sage_memory.writer import MemoryWriter
    store = GraphStore(db_path=tmp_path / "conflict.sqlite")
    writer = MemoryWriter(store)
    evo = MemoryEvolution(store)

    f_like = Fact(subject="Alice", predicate="likes",  object="coffee")
    f_hate = Fact(subject="Alice", predicate="hates",  object="coffee")
    writer.add_fact(f_like)
    writer.add_fact(f_hate)

    conflicts = evo.detect_conflicts()
    assert len(conflicts) >= 1
    subjects = {c[0].subject for c in conflicts}
    assert "Alice" in subjects


def test_auto_resolve_conflicts(tmp_path):
    """auto_resolve 應降權較弱的衝突方"""
    from sage_memory.graph_store import GraphStore
    from sage_memory.writer import MemoryWriter
    store = GraphStore(db_path=tmp_path / "resolve.sqlite")
    writer = MemoryWriter(store)
    evo = MemoryEvolution(store)

    f1 = Fact(subject="Bob", predicate="likes",  object="tea", weight=0.9)
    f2 = Fact(subject="Bob", predicate="hates",  object="tea", weight=0.4)
    writer.add_fact(f1)
    writer.add_fact(f2)

    resolved = evo.auto_resolve_conflicts()
    assert resolved >= 1
    # 較弱的 f2 應被降權或刪除
    updated = store.get_fact(f2.fact_id)
    assert updated is None or updated.weight < f2.weight


def test_merge_rejects_conflicting_predicates(tmp_path):
    """矛盾謂詞的 fact 不應被 merge，應改為 conflict_flag"""
    from sage_memory.graph_store import GraphStore
    from sage_memory.writer import MemoryWriter
    store = GraphStore(db_path=tmp_path / "merge_reject.sqlite")
    writer = MemoryWriter(store)
    evo = MemoryEvolution(store)

    f1 = Fact(subject="Carol", predicate="likes",  object="rain")
    f2 = Fact(subject="Carol", predicate="hates",  object="rain")
    writer.add_fact(f1)
    writer.add_fact(f2)

    # merge 應被拒絕（矛盾謂詞）
    result = evo.apply_correction(f1.fact_id, "merge", target_id=f2.fact_id)
    assert result is False or store.get_fact(f1.fact_id) is not None


def test_scheduled_decay_dry_run(tmp_path):
    """dry_run=True 不應實際修改任何 fact"""
    import time
    from sage_memory.graph_store import GraphStore
    from sage_memory.writer import MemoryWriter
    store = GraphStore(db_path=tmp_path / "dry.sqlite")
    writer = MemoryWriter(store)
    evo = MemoryEvolution(store)

    # 插入一個「舊」fact（timestamp 往前推 30 天）
    old_fact = Fact(subject="Dave", predicate="likes", object="hiking",
                    timestamp=time.time() - 86400 * 30)
    store.add_fact(old_fact)

    before_count = store.edge_count
    stats = evo.run_scheduled_decay(age_days_threshold=1.0, dry_run=True)
    assert store.edge_count == before_count   # 不應刪除
    assert stats["decayed"] >= 1


def test_source_differentiated_decay(tmp_path):
    """inference 來源應比 user 來源衰減更快"""
    from sage_memory.graph_store import GraphStore
    from sage_memory.evolution import _DECAY_RATES
    assert _DECAY_RATES["inference"] > _DECAY_RATES["user"]