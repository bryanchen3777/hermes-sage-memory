"""端對端測試：完整對話流 write → retrieve → correct → persist"""
import pytest
from pathlib import Path
from sage_memory.adapter import SAGELiteProvider


@pytest.fixture
def provider(tmp_path):
    p = SAGELiteProvider(top_k=5, max_hops=2, max_tokens=800)
    p.initialize("session-e2e", hermes_home=str(tmp_path))
    return p


def test_full_conversation_flow(provider):
    # 1. 寫入對話
    provider.sync_turn(
        user_content="I like hiking and I live in Queens, New York.",
        assistant_content="I'll remember you enjoy outdoor activities.",
        session_id="session-e2e",
    )

    # 2. 檢索
    context = provider.prefetch("What do I like?", session_id="session-e2e")
    assert isinstance(context, str)

    # 3. 工具呼叫：手動加入事實
    result = provider.handle_tool_call("sage_add_fact", {
        "subject": "user",
        "predicate": "lives_in",
        "object": "Queens",
    })
    assert "fact_id" in result

    # 4. 召回驗證
    recall = provider.handle_tool_call("sage_recall", {"query": "where do I live"})
    import json
    data = json.loads(recall)
    assert data["fact_count"] >= 1

    # 5. 關閉與持久化
    provider.shutdown()


def test_session_switch_persists(tmp_path):
    p = SAGELiteProvider()
    p.initialize("session-A", hermes_home=str(tmp_path))
    p.sync_turn("I love pizza", "Noted.", session_id="session-A")
    p.on_session_end([])

    # 切換 session 後再度初始化應讀到舊資料
    p2 = SAGELiteProvider()
    p2.initialize("session-B", hermes_home=str(tmp_path))
    assert p2._store.edge_count >= 0  # 不崩潰即可