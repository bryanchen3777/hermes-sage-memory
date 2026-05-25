"""
SOUL / 角色扮演場景示範
展示 SAGE-lite 如何追蹤角色關係、情感變化、事件因果
"""
from pathlib import Path
from sage_memory import SAGELiteProvider

def run_demo():
    provider = SAGELiteProvider(top_k=5, max_hops=2, recall_mode="balanced")
    provider.initialize("soul-demo", hermes_home="/tmp/sage-demo")

    # 模擬角色對話
    turns = [
        ("我叫小明，我喜歡打棒球。",          "我記住了，小明喜歡打棒球。"),
        ("我討厭下雨天，因為不能練習。",       "了解，你討厭下雨天。"),
        ("我的教練叫王老師，他很嚴格。",       "王老師是你的棒球教練。"),
        ("昨天比賽我們隊贏了！",              "恭喜你們獲勝！"),
        ("王老師說我有潛力成為主投手。",       "王老師很看好你的投球天賦。"),
    ]

    for user_msg, assistant_msg in turns:
        provider.sync_turn(user_msg, assistant_msg, session_id="soul-demo")

    print("=== 記憶圖統計 ===")
    import json
    stats = json.loads(provider.handle_tool_call("sage_stats", {}))
    print(f"  活躍事實：{stats['active_facts']} 條")
    print(f"  實體節點：{stats['node_count']} 個")

    print("\n=== 查詢：小明的興趣 ===")
    print(provider.prefetch("小明喜歡什麼", session_id="soul-demo"))

    print("\n=== 查詢：關於王老師 ===")
    print(provider.prefetch("王老師", session_id="soul-demo"))

    provider.shutdown()
    print("\n[*] Demo complete. Data persisted to /tmp/sage-demo")

if __name__ == "__main__":
    run_demo()