"""Phase 5 專屬：SQLite 持久化、migration、export/import、stats 測試"""
import json
import time
from pathlib import Path
import pytest

from sage_memory.graph_store import GraphStore, _SCHEMA_VERSION
from sage_memory.models import Fact


@pytest.fixture
def store(tmp_path):
    return GraphStore(db_path=tmp_path / "test.sqlite")


@pytest.fixture
def populated(tmp_path):
    s = GraphStore(db_path=tmp_path / "pop.sqlite")
    facts = [
        Fact(subject="Alice", predicate="likes",    object="coffee",   source="user"),
        Fact(subject="Alice", predicate="lives_in", object="NYC",      source="user"),
        Fact(subject="Bob",   predicate="works_at", object="TechCorp", source="inference"),
        Fact(subject="Bob",   predicate="knows",    object="Alice",    source="user"),
        Fact(subject="Carol", predicate="is",       object="engineer", source="user"),
    ]
    for f in facts:
        s.add_fact(f)
    s.flush()
    return s


# ── Schema & Migration ────────────────────────────────────────

def test_schema_version_written(store):
    """DB 應寫入 schema version"""
    conn = store._get_conn()
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key='version'"
    ).fetchone()
    assert row is not None
    assert int(row["value"]) == _SCHEMA_VERSION


def test_tags_column_exists(store):
    """v2 migration 應新增 tags 欄位"""
    conn = store._get_conn()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(facts)").fetchall()]
    assert "tags" in cols


def test_session_index_exists(store):
    """v2 migration 應建立 session_id 索引"""
    conn = store._get_conn()
    indexes = [r[1] for r in conn.execute(
        "SELECT * FROM sqlite_master WHERE type='index'"
    ).fetchall()]
    assert any("session" in idx for idx in indexes)


# ── WAL 模式 ──────────────────────────────────────────────────

def test_wal_mode_enabled(store):
    """應啟用 WAL journal mode"""
    conn = store._get_conn()
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_batch_commit(tmp_path):
    """batch_size 內的寫入不應立即 commit，flush 後才持久化"""
    s = GraphStore(db_path=tmp_path / "batch.sqlite", batch_size=5)
    for i in range(4):
        s.add_fact(Fact(subject=f"N{i}", predicate="is", object="node"))
    # 尚未 flush，但 edge_count 在 graph 中已存在
    assert s.edge_count == 4
    s.flush()
    # flush 後 DB 應有 4 筆
    conn = s._get_conn()
    count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    assert count == 4


# ── Context Manager ───────────────────────────────────────────

def test_context_manager_closes(tmp_path):
    """with 語法應自動 flush + close"""
    db = tmp_path / "ctx.sqlite"
    with GraphStore(db_path=db) as s:
        s.add_fact(Fact(subject="X", predicate="is", object="Y"))
    # close 後 _conn 應為 None
    assert s._conn is None
    # 重新開啟應能讀到資料
    s2 = GraphStore(db_path=db)
    assert s2.edge_count == 1


# ── Stats ─────────────────────────────────────────────────────

def test_stats_keys(populated):
    """stats() 應包含所有必要的 key"""
    s = populated.stats()
    required = {
        "total_facts", "active_facts", "pruned_facts",
        "avg_weight", "node_count", "edge_count",
        "source_breakdown", "oldest_fact_days", "db_path",
    }
    assert required.issubset(s.keys())


def test_stats_source_breakdown(populated):
    """source_breakdown 應包含 user 和 inference"""
    s = populated.stats()
    assert "user" in s["source_breakdown"]
    assert "inference" in s["source_breakdown"]


# ── Export / Import ───────────────────────────────────────────

def test_export_json(populated, tmp_path):
    """export_json 應產生正確筆數的 JSON Lines"""
    out = tmp_path / "export.jsonl"
    count = populated.export_json(out)
    assert count == populated.edge_count
    lines = out.read_text().strip().split("\n")
    assert len(lines) == count
    # 每行應是合法 JSON
    for line in lines:
        d = json.loads(line)
        assert "subject" in d and "predicate" in d and "object" in d


def test_import_json_roundtrip(populated, tmp_path):
    """export 後 import 到新 store，資料應完整還原"""
    out = tmp_path / "roundtrip.jsonl"
    original_count = populated.edge_count
    populated.export_json(out)

    s2 = GraphStore(db_path=tmp_path / "import.sqlite")
    result = s2.import_json(out)
    assert result["imported"] == original_count
    assert s2.edge_count == original_count


def test_import_json_skip_low_weight(populated, tmp_path):
    """import 時 min_weight 過濾應跳過低權重 facts"""
    # 先加一個低 weight fact 並 export
    populated.add_fact(Fact(
        subject="LowW", predicate="is", object="weak", weight=0.02
    ))
    populated.flush()
    out = tmp_path / "loww.jsonl"
    populated.export_json(out)

    s2 = GraphStore(db_path=tmp_path / "filtered.sqlite")
    result = s2.import_json(out, min_weight=0.05)
    assert result["skipped"] >= 1
    # LowW 不應存在
    assert not any(
        f.subject == "LowW" for f in s2.get_all_facts(min_weight=0.0)
    )