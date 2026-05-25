# hermes-sage-memory

> A lightweight causal graph memory plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).  
> Replace flat text memory with a self-evolving temporal knowledge graph — zero external dependencies.

[![Tests](https://img.shields.io/badge/tests-75%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Why SAGE-lite?

| | Hermes Built-in Memory | SAGE-lite |
|---|---|---|
| Structure | Flat text (MEMORY.md) | Causal graph (S→P→O) |
| Retrieval | Keyword / vector snippet | Multi-hop causal traversal |
| Self-correction | Manual rewrite | auto decay / prune / merge |
| Cross-session search | Fragmented | Profile-isolated SQLite |
| Context injection | Full text dump | Top-K compressed summary |
| External deps | None | None (NetworkX + SQLite) |

---

## Quick Start

### Install

```bash
pip install hermes-sage-memory
```

### Use as Hermes Plugin

```bash
# Copy plugin entry to Hermes plugins directory
cp -r plugins/memory/sage_lite /path/to/hermes-agent/plugins/memory/

# Launch Hermes with SAGE-lite memory
hermes --memory-provider sage_lite
```

### Standalone Usage

```python
from sage_memory import SAGELiteProvider

provider = SAGELiteProvider(top_k=5, max_hops=2, max_tokens=800)
provider.initialize("my-session", hermes_home="~/.hermes")

# Write a conversation turn
provider.sync_turn(
    user_content="I love hiking and I live in Queens, New York.",
    assistant_content="Got it, I'll remember that.",
    session_id="my-session",
)

# Retrieve relevant context
context = provider.prefetch("What do I enjoy?", session_id="my-session")
print(context)
# Memory
# - User: likes hiking(1.0), lives_in Queens(1.0)
```

---

## Architecture

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

## Memory Lifecycle

1. **Write** — `sync_turn()` extracts triples from each conversation turn via pattern matching
2. **Retrieve** — `prefetch()` scores facts by `weight × recency × relevance`, returns compressed summary
3. **Evolve** — scheduled decay ages old facts; `apply_correction()` handles user feedback
4. **Persist** — WAL-mode SQLite with batch commits; export/import via JSON Lines

---

## Recall Modes

| Mode | Facts | Chains | Token Use | Best For |
|---|---|---|---|---|
| `precise` | top 3 | none | minimal | Quick factual queries |
| `balanced` | top 5 | top 2 | moderate | Default general use |
| `expansive` | top 10 | top 3 | higher | Deep reasoning tasks |

---

## Tools Exposed to Hermes

| Tool | Description |
|---|---|
| `sage_add_fact` | Manually add a structured fact |
| `sage_correct` | Decay / prune / merge a fact |
| `sage_recall` | Trigger manual recall with mode selection |
| `sage_stats` | Return graph health statistics |

---

## Project Structure

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
├── tests/                  # 75 tests, 0 external dependencies
├── examples/
├── docs/
├── README.md
└── pyproject.toml
```

---

## Configuration

```python
provider = SAGELiteProvider(
    top_k=5,           # Facts retrieved per query
    max_hops=2,        # Graph traversal depth
    max_tokens=800,    # Context injection budget
    recall_mode="balanced",  # precise / balanced / expansive
)
```

Or via Hermes config UI — SAGE-lite exposes `get_config_schema()` for interactive setup.

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Roadmap

- [ ] Async prefetch (background graph traversal)
- [ ] LLM-assisted triple extraction (optional, pluggable)
- [ ] Neo4j backend adapter
- [ ] Distributed graph store for multi-agent setups
- [ ] Web UI for memory graph visualization

---

## License

MIT © 2026