"""
Phase 6：mock Hermes 環境下的完整 Plugin 生命週期測試
不依賴真實 Hermes，用 MockContext 模擬 ctx.register_memory_provider()
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from sage_memory.adapter import SAGELiteProvider
from sage_memory.models import Fact


# ── Mock Hermes Context ───────────────────────────────────────

class MockHermesContext:
    """模擬 Hermes Plugin ctx 物件"""
    def __init__(self):
        self.registered_providers: list = []

    def register_memory_provider(self, provider) -> None:
        self.registered_providers.append(provider)


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def hermes_home(tmp_path) -> Path:
    return tmp_path / "hermes_home"


@pytest.fixture
def provider(hermes_home) -> SAGELiteProvider:
    p = SAGELiteProvider(top_k=5, max_hops=2, max_tokens=800)
    p.initialize(
        "session-test",
        hermes_home=str(hermes_home),
        agent_context={"profile": "test_profile"},
    )
    return p


# ── Plugin 註冊 ─────────────────────────────────────────────

def test_plugin_register(hermes_home):
    """register(ctx) 應成功掛載 SAGELiteProvider"""
    ctx = MockHermesContext()
    # 模擬 hermes_plugin.register()
    provider = SAGELiteProvider()
    ctx.register_memory_provider(provider)
    assert len(ctx.registered_providers) == 1
    assert ctx.registered_providers[0].name == "sage_lite"


def test_is_available(provider):
    """networkx 已安裝，is_available 應為 True"""
    assert provider.is_available() is True


def test_provider_name(provider):
    assert provider.name == "sage_lite"


# ── 初始化與 Profile 隔離 ─────────────────────────────────────

def test_profile_db_path(provider, hermes_home):
    """DB 應建立在正確的 profile 路徑下"""
    expected = (
        hermes_home / "profiles" / "test_profile"
        / "sage_memory" / "graph.sqlite"
    )
    assert provider._store.db_path == expected
    assert expected.exists()


def test_profile_isolation(hermes_home):
    """不同 profile 應使用不同 DB，資料互不影響"""
    p1 = SAGELiteProvider()
    p1.initialize("s1", hermes_home=str(hermes_home),
                  agent_context={"profile": "alice"})
    p1.sync_turn("I love jazz music.", "", session_id="s1")
    p1.shutdown()

    p2 = SAGELiteProvider()
    p2.initialize("s2", hermes_home=str(hermes_home),
                  agent_context={"profile": "bob"})
    # bob 的 profile 不應有 alice 的資料
    result = p2.prefetch("jazz", session_id="s2")
    assert "jazz" not in result.lower()
    p2.shutdown()


# ── 完整對話生命週期 ──────────────────────────────────────────

def test_full_lifecycle(provider):
    """write → prefetch → tool_call → session_end 完整流程"""
    # 1. 寫入對話
    provider.sync_turn(
        "I like hiking and I live in Queens.",
        "Got it, I'll remember that.",
        session_id="session-test",
    )
    # 2. prefetch 應有內容
    context = provider.prefetch("What do I like?", session_id="session-test")
    assert isinstance(context, str)

    # 3. sage_add_fact 工具
    r = json.loads(provider.handle_tool_call("sage_add_fact", {
        "subject": "user", "predicate": "works_at", "object": "TechCorp"
    }))
    assert r["status"] == "ok"
    assert "fact_id" in r

    # 4. sage_recall 工具
    r2 = json.loads(provider.handle_tool_call("sage_recall", {
        "query": "where does user work"
    }))
    assert r2["fact_count"] >= 1

    # 5. sage_stats 工具
    r3 = json.loads(provider.handle_tool_call("sage_stats", {}))
    assert r3["active_facts"] >= 1

    # 6. session_end
    provider.on_session_end([])


def test_system_prompt_block_format(provider):
    """system_prompt_block 應包含 active_facts 和 profile"""
    provider.sync_turn("I enjoy reading.", "", session_id="session-test")
    block = provider.system_prompt_block()
    assert "sage_lite" in block.lower() or "SAGE-lite" in block
    assert "test_profile" in block


def test_on_pre_compress(provider):
    """on_pre_compress 有資料時應回傳非空字串"""
    provider.sync_turn("I like coffee.", "", session_id="session-test")
    result = provider.on_pre_compress([])
    assert isinstance(result, str)
    assert len(result) > 0


def test_on_pre_compress_empty_store(hermes_home):
    """空 store 時 on_pre_compress 應回傳空字串"""
    p = SAGELiteProvider()
    p.initialize("s", hermes_home=str(hermes_home),
                 agent_context={"profile": "empty_profile"})
    result = p.on_pre_compress([])
    assert result == ""


# ── Session Switch ────────────────────────────────────────────

def test_session_switch_same_profile(provider):
    """同 profile 切換 session 不應重建 DB"""
    old_store = provider._store
    provider.on_session_switch("session-new")
    assert provider._store is old_store  # 同一個 store 物件
    assert provider._session_id == "session-new"


def test_session_switch_new_profile(provider, hermes_home):
    """切換到新 profile 應重建 DB"""
    old_store = provider._store
    provider.on_session_switch(
        "session-new",
        agent_context={"profile": "another_profile"},
    )
    assert provider._store is not old_store
    assert provider._profile_name == "another_profile"


# ── Tool 錯誤處理 ─────────────────────────────────────────────

def test_unknown_tool_returns_error(provider):
    """未知工具名稱應回傳 error JSON 而非拋出例外"""
    r = json.loads(provider.handle_tool_call("nonexistent_tool", {}))
    assert "error" in r


def test_sage_correct_not_found(provider):
    """對不存在的 fact_id 執行 correct 應回傳 not_found"""
    r = json.loads(provider.handle_tool_call("sage_correct", {
        "fact_id": "nonexistent-id-12345",
        "action": "prune",
    }))
    assert r["status"] == "not_found"


# ── Config ────────────────────────────────────────────────────

def test_save_config_updates_values(provider):
    provider.save_config({
        "top_k": 10,
        "max_hops": 3,
        "max_tokens": 1200,
        "recall_mode": "expansive",
    }, hermes_home="~/.hermes")
    assert provider.top_k == 10
    assert provider.max_hops == 3
    assert provider.max_tokens == 1200
    assert provider.recall_mode == "expansive"


def test_get_config_schema_keys(provider):
    schema = provider.get_config_schema()
    keys = {s["key"] for s in schema}
    assert {"top_k", "max_hops", "max_tokens", "recall_mode"}.issubset(keys)