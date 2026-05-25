# hermes-sage-memory

> Causal graph memory plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).
> Replace flat text memory with a self-evolving temporal knowledge graph — zero external dependencies.
>
> [Hermes Agent](https://github.com/NousResearch/hermes-agent) 的因果圖譜記憶外掛。
> 將平面文字記憶替換為自我演化的时间知识图谱——无外部依赖。

[![Tests](https://img.shields.io/badge/tests-110%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 給 AI 裝上真正的長期記憶 / Equipping AI with True Long-Term Memory

**hermes-sage-memory** — 這個項目的核心目的，是解決 AI 在長久相處時最根本的痛點：**聊再久，它還是不認識你。**

### 1. 用大白話解釋它是什麼 / In Plain Terms — What Is It?

想像你雇了一個全能的秘書，他精通天文地理，卻有一個致命缺點：**記憶像金魚**。每次聊完，只要對話太長，或隔了一段時間，他就會把你的生活習慣、昨天交代的重要事項、甚至你們最私密的心事忘得一乾二淨。每次聊天都像第一次見面，你必須不厭其煩地重新介紹自己。

hermes-sage-memory 就是幫這個 AI 秘書**裝上真正的長期大腦**。它不是死板地讓 AI「背更多字」，而是讓它像人一樣，把對話提煉成有結構的記憶，隨著相處越久，它就越懂你。

### 2. 它有什麼厲害的地方？/ What Makes It Powerful?

**真正的長短期記憶：** 不會因為對話太長就忘記前面。重要的事情（例如：你喜歡喝不加糖的黑咖啡、你正在進行的重要專案、你說過的承諾）會被自動提煉、分層儲存，情境對了就立刻召回。

**記憶會自我演化：** 重複出現的事情會被強化、長期沒提到的會慢慢淡化、互相矛盾的記憶會自動解決衝突。它不是死的資料庫，而是像人類大腦一樣是「活的」。

**完全個人化、完全隔離：** 每個 AI 角色擁有自己獨立的記憶圖譜。它記得的是「你和它之間」的專屬回憶，絕對不會混入其他人的資料。

### 3. 對比其他技術，它獨特在哪裡？/ How Is It Different from Other Technologies?

市面上多數技術只是讓 AI 變成「記性好一點的資料庫」。而 hermes-sage-memory 試圖做的是：**讓 AI 真正記得你這個人**——不只是你說過的字，而是你們之間的關係、偏好與時間軌跡。

| 核心技術直觀對比 | 一般 AI 聊天 (滑動窗口) | 傳統 RAG (關鍵字搜尋) | hermes-sage-memory |
|---|---|---|---|
| **記憶結構** | 僅記得最近幾句話 | 搜尋關鍵字，撈出死板的文字片段 | 網狀關係圖譜（誰、在何時、喜歡什麼） |
| **遺忘方式** | 對話一長就徹底忘記 | 只記得個別字眼，不記得前後關係 | 自然淡化，重要與重複的記憶永遠留著 |
| **跨話題關聯** | 完全無能力 | 效果微弱 | 強大（可建立 A → B → C 的多跳因果鏈） |
| **自我修正** | 無法自己修正錯誤 | 無法解決衝突 | 衝突自動解決、發現錯誤可動態糾正 |
| **個人化程度** | 每次開新對話就重置 | 依賴你提問的字眼 | 隨相處時間與互動深淺，持續動態加深 |

> 💡 **項目現狀：** 已通過 110 個自動化測試，支援 10 個獨立角色 Profile，達到零外部依賴，可直接無縫掛載於 Hermes Agent 框架運行。

---

## Why SAGE-lite? / 為什麼用 SAGE-lite？

| | Hermes Built-in Memory<br>Hermes 內建記憶 | SAGE-lite |
|---|---|---|
| Structure 結構 | Flat text (MEMORY.md) | Causal graph (S→P→O) 因果圖譜 |
| Retrieval 檢索 | Keyword / vector snippet | Multi-hop causal traversal 多跳因果遍歷 |
| Self-correction 自我修正 | Manual rewrite 手動改寫 | Auto decay/prune/merge 自動衰減/修剪/合併 |
| Cross-session 跨 session | Fragmented 碎片化 | Profile-isolated SQLite 隔離存儲 |
| Context injection 上下文注入 | Full text dump 全文傾倒 | Top-K compressed summary 精選壓縮摘要 |
| External deps 外部依賴 | None | None (NetworkX + SQLite) |

---

## Quick Start / 快速開始

### Install / 安裝

```bash
pip install hermes-sage-memory
```

### Use as Hermes Plugin / 做為 Hermes 外掛使用

```bash
# Copy plugin entry to Hermes plugins directory
# 將外掛複製到 Hermes plugins 目錄
cp -r plugins/memory/sage_lite /path/to/hermes-agent/plugins/memory/

# Launch Hermes with SAGE-lite memory
# 以 SAGE-lite 記憶啟動 Hermes
hermes --memory-provider sage_lite
```

### Standalone Usage / 獨立使用

```python
from sage_memory import SAGELiteProvider

provider = SAGELiteProvider(top_k=5, max_hops=2, max_tokens=800)
provider.initialize("my-session", hermes_home="~/.hermes")

# Write a conversation turn / 寫入對話輪次
provider.sync_turn(
    user_content="I love hiking and I live in Queens, New York.",
    assistant_content="Got it, I'll remember that.",
    session_id="my-session",
)

# Retrieve relevant context / 檢索相關上下文
context = provider.prefetch("What do I enjoy?", session_id="my-session")
print(context)
# Memory
# - User: likes hiking(1.0), lives_in Queens(1.0)
```

---

## Architecture / 架構

![SAGE Lite Architecture](architecture.png)

```
Hermes Agent
│
▼
┌─────────────────────────────┐
│ SAGELiteProvider │ ← Hermes MemoryProvider ABC
│ (sage_memory/adapter.py)    │
└──────────┬──────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
  Writer        Reader
  add_fact()   retrieve_context()
  write_turn()  multi-hop chains
    │             │
    └──────┬──────┘
           ▼
  Temporal Knowledge Graph
  NetworkX MultiDiGraph
  (S, P, O, timestamp, weight)
           │
           ▼
  Self-Evolution Loop
  decay / prune / merge
  conflict detection
           │
           ▼
  SQLite Persistence
  profiles/<name>/sage_memory/graph.sqlite
```

---

## Memory Lifecycle / 記憶生命週期

1. **Write 寫入** — `sync_turn()` extracts triples via pattern matching 透過模式匹配提取三元組
2. **Retrieve 檢索** — `prefetch()` scores by `weight × recency × relevance`，returns compressed summary 回傳壓縮摘要
3. **Evolve 演化** — scheduled decay ages old facts; `apply_correction()` handles user feedback 排程衰減舊事實；處理用戶反饋
4. **Persist 持久化** — WAL-mode SQLite with batch commits; JSON Lines export/import WAL 模式 SQLite 批量提交；JSON Lines 匯出入

---

## Recall Modes / 檢索模式

| Mode 模式 | Facts 事實 | Chains 鏈 | Token Use | Best For 適用場景 |
|---|---|---|---|---|
| `precise` | top 3 | none | minimal 最小 | Quick factual queries 快速事實查詢 |
| `balanced` | top 5 | top 2 | moderate 適中 | Default general use 預設通用場景 |
| `expansive` | top 10 | top 3 | higher 較高 | Deep reasoning tasks 深度推理任務 |

---

## Tools Exposed to Hermes / 暴露給 Hermes 的工具

| Tool 工具 | Description 說明 |
|---|---|
| `sage_add_fact` | Manually add a structured fact 手動新增結構化事實 |
| `sage_correct` | Decay / prune / merge a fact 衰減/修剪/合併事實 |
| `sage_recall` | Trigger manual recall with mode selection 手動觸發檢索 |
| `sage_stats` | Return graph health statistics 回傳圖譜健康統計 |

---

## Project Structure / 專案結構

```
hermes-sage-memory/
├── sage_memory/
│   ├── models.py          # Fact, ContextResult dataclasses
│   ├── graph_store.py     # NetworkX + SQLite (WAL, migration, export)
│   ├── writer.py          # Triple extraction + dedup + normalization
│   ├── reader.py          # Multi-hop retrieval + 3-mode recall
│   ├── evolution.py       # Decay / prune / merge + conflict detection
│   ├── token_utils.py     # Budget tracking + compression + cache
│   └── adapter.py         # Hermes MemoryProvider ABC implementation
├── integrations/
│   └── hermes_plugin.py   # register(ctx) entry point
├── plugins/
│   └── memory/sage_lite/
│       └── __init__.py
├── tests/                  # 110 tests, 0 external dependencies 無外部依賴
├── docs/
├── examples/
├── SAGE Lite架構圖.png    # Architecture diagram 架構圖
├── README.md
└── pyproject.toml
```

---

## Configuration / 設定

```python
provider = SAGELiteProvider(
    top_k=5,           # Facts retrieved per query 每查詢檢索的事實數
    max_hops=2,        # Graph traversal depth 圖譜遍歷深度
    max_tokens=800,    # Context injection budget 上下文注入預算
    recall_mode="balanced",  # precise / balanced / expansive
)
```

Or via Hermes config UI — SAGE-lite exposes `get_config_schema()` for interactive setup.  
或透過 Hermes 設定 UI——SAGE-lite 暴露 `get_config_schema()` 供互動式設定。

---

## Running Tests / 執行測試

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Roadmap / 發展藍圖

- [ ] Async prefetch (background graph traversal) 異步預取（背景圖譜遍歷）
- [ ] LLM-assisted triple extraction (optional, pluggable) LLM 輔助三元組抽取（可選插拔）
- [ ] Neo4j backend adapter Neo4j 後端適配器
- [ ] Distributed graph store for multi-agent setups 多代理分散式圖譜存儲
- [ ] Web UI for memory graph visualization 記憶圖譜視覺化 Web UI

---

## License

MIT © 2026