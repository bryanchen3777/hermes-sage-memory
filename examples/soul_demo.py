"""
SOUL 整合示範（Hermes SOUL 角色系統）
展示 SAGE-lite 如何與 SOUL.md 角色設定協同運作
"""
from sage_memory import SAGELiteProvider

def run_demo():
    home = "/tmp/sage-soul-demo"

    print("=== SOUL Demo：長期角色關係追蹤 ===\n")
    provider = SAGELiteProvider(
        top_k=8, max_hops=3, recall_mode="expansive"
    )
    provider.initialize(
        "soul-session",
        hermes_home=home,
        agent_context={"profile": "yua-character"},
    )

    # 角色建立對話
    conversations = [
        ("你好，我叫陳小華，是一名工程師。",
         "你好！很高興認識你，小華。"),
        ("我在紐約工作，但我來自台灣台北。",
         "原來你是從台北來的工程師！"),
        ("我有一隻貓叫做咪咪，牠很愛撒嬌。",
         "咪咪聽起來很可愛！"),
        ("我喜歡看動漫，最近在追《怪獸8號》。",
         "《怪獸8號》很熱門呢！"),
        ("我不喜歡早起，工作都在深夜最有效率。",
         "你是夜貓子型的工程師！"),
    ]

    for user_msg, ai_msg in conversations:
        provider.sync_turn(user_msg, ai_msg, session_id="soul-session")

    import json
    stats = json.loads(provider.handle_tool_call("sage_stats", {}))
    print(f"記憶統計：{stats['active_facts']} 條事實，{stats['node_count']} 個實體\n")

    queries = [
        "小華的職業是什麼？",
        "小華有養什麼寵物？",
        "小華的作息習慣",
    ]
    for q in queries:
        print(f"Q: {q}")
        result = provider.prefetch(q, session_id="soul-session")
        print(f"A: {result}\n")

    provider.shutdown()
    print("[*] SOUL Demo complete")

if __name__ == "__main__":
    run_demo()