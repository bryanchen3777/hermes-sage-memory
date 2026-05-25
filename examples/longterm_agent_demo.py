"""
長期對話代理示範
展示跨 session 記憶保留、decay 機制、conflict 解決
"""
import time
from sage_memory import SAGELiteProvider

def run_demo():
    home = "/tmp/sage-longterm-demo"

    # Session 1：建立記憶
    print("=== Session 1：初次對話 ===")
    p1 = SAGELiteProvider(top_k=5, recall_mode="balanced")
    p1.initialize("session-1", hermes_home=home,
                  agent_context={"profile": "agent-demo"})
    p1.sync_turn(
        "My name is Alex. I work at OpenAI and I love Python.",
        "Nice to meet you, Alex!",
        session_id="session-1",
    )
    p1.sync_turn(
        "I prefer remote work and I live in San Francisco.",
        "Got it, remote work in SF.",
        session_id="session-1",
    )
    stats = p1._store.stats()
    print(f"  寫入 {stats['active_facts']} 條事實")
    p1.shutdown()

    # Session 2：跨 session 檢索
    print("\n=== Session 2：跨 session 記憶檢索 ===")
    p2 = SAGELiteProvider(top_k=5, recall_mode="balanced")
    p2.initialize("session-2", hermes_home=home,
                  agent_context={"profile": "agent-demo"})
    context = p2.prefetch("Where does Alex live?", session_id="session-2")
    print(f"  檢索結果：\n{context}")

    # 模擬用戶糾錯
    print("\n=== 糾錯示範：Alex 換工作了 ===")
    import json
    all_facts = p2._store.get_all_facts()
    openai_fact = next(
        (f for f in all_facts if "openai" in f.object.lower()), None
    )
    if openai_fact:
        result = p2.handle_tool_call("sage_correct", {
            "fact_id": openai_fact.fact_id,
            "action": "decay",
            "delta": 0.5,
            "reason": "user_updated_job",
        })
        print(f"  糾錯結果：{result}")
        # 加入新工作
        p2.handle_tool_call("sage_add_fact", {
            "subject": "Alex", "predicate": "works_at", "object": "Anthropic"
        })
        print("  新增：Alex works_at Anthropic")

    p2.shutdown()
    print("\n[*] Longterm memory Demo complete")

if __name__ == "__main__":
    run_demo()