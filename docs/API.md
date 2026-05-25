# SAGE-lite API Reference

## SAGELiteProvider

### Constructor

```python
SAGELiteProvider(
    top_k: int = 5,
    max_hops: int = 2,
    max_tokens: int = 800,
    recall_mode: str = "balanced"
)
```

### Hermes ABC Methods

| Method | Required | Description |
|---|---|---|
| `name` | yes | Returns `"sage_lite"` |
| `is_available()` | yes | Checks networkx import |
| `initialize(session_id, **kwargs)` | yes | Sets up store, writer, reader |
| `get_tool_schemas()` | yes | Returns 4 tool definitions |
| `system_prompt_block()` | optional | Stats summary string |
| `prefetch(query, *, session_id)` | optional | Returns compressed context |
| `sync_turn(user, assistant, *, session_id)` | optional | Extracts + writes facts |
| `handle_tool_call(name, args)` | optional | Routes to sage_* tools |
| `on_memory_write(action, target, content, metadata)` | optional | Mirrors builtin writes |
| `on_pre_compress(messages)` | optional | Informs compressor of graph state |
| `on_session_switch(new_id, **kwargs)` | optional | Handles profile changes |
| `on_session_end(messages)` | optional | Flushes SQLite |
| `shutdown()` | optional | Flushes + closes DB |

### Tools

#### `sage_add_fact`

```json
{
  "subject": "Alice",
  "predicate": "likes",
  "object": "coffee",
  "weight": 1.0
}
```

Returns: `{"status": "ok", "fact_id": "<uuid>"}`

#### `sage_correct`

```json
{
  "fact_id": "<uuid>",
  "action": "decay" | "prune" | "merge" | "conflict_flag",
  "target_id": "<uuid>",
  "delta": 0.1,
  "reason": "user_correction"
}
```

Returns: `{"status": "ok" | "not_found"}`

#### `sage_recall`

```json
{
  "query": "what does Alice like",
  "top_k": 5,
  "mode": "precise" | "balanced" | "expansive"
}
```

Returns: `{"summary": "...", "fact_count": 3, "chain_count": 1, "token_estimate": 45}`

#### `sage_stats`

Returns:

```json
{
  "total_facts": 42,
  "active_facts": 38,
  "pruned_facts": 4,
  "avg_weight": 0.823,
  "node_count": 25,
  "edge_count": 38,
  "source_breakdown": {"user": 30, "inference": 8},
  "oldest_fact_days": 3.2,
  "db_path": "~/.hermes/profiles/default/sage_memory/graph.sqlite"
}
```