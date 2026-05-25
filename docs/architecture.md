# Architecture Deep Dive

## Core Design Principles

1. **Zero external dependencies** — NetworkX + SQLite (both pure Python / stdlib)
2. **Hermes-native** — implements `MemoryProvider` ABC, registered via `register(ctx)`
3. **Causal over semantic** — stores relationships, not embeddings
4. **Self-correcting** — facts decay over time, conflicts auto-resolve

---

## Data Model

```
Fact {
    subject: str     # Entity A ("Alice", "user")
    predicate: str    # Relationship ("likes", "works_at")
    object: str      # Entity B ("coffee", "TechCorp")
    timestamp: float  # Unix time of creation
    weight: float    # Confidence 0.0–1.0
    source: str      # "user" | "inference" | "correction"
    fact_id: str     # UUID
    session_id: str  # Originating session
}
```

## Graph Store

- **In-memory**: NetworkX `MultiDiGraph` for fast traversal
- **On-disk**: SQLite with WAL mode, batch commits (every 20 writes)
- **Schema v2**: `facts` table + `schema_meta` for migration tracking
- **Indexes**: `subject`, `object`, `weight`, `session_id`

## Scoring Formula

```
composite_score(fact) =
    weight
    × exp(-age_days / 30)   # recency half-life 30 days
    × (0.4 + 0.6 × relevance)   # relevance: keyword hit ratio
```

## Evolution Rules

| Source | Decay Rate | Rationale |
|---|---|---|
| `user` | 0.03/cycle | User statements are authoritative |
| `inference` | 0.08/cycle | Model inferences are less certain |
| `correction` | 0.02/cycle | Corrected facts should persist |

Auto-prune threshold: `weight < 0.05`

## Profile Isolation

Each Hermes profile gets its own SQLite file:

```
~/.hermes/profiles/<name>/sage_memory/graph.sqlite
```

Session switches within the same profile reuse the store.
Profile switches trigger full store reload.