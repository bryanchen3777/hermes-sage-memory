"""
SAGELiteProvider — 實作 Hermes MemoryProvider ABC v8
升級：on_memory_retrieved hook、write health、confidence/sigmoid 正規化
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Optional

from .graph_store import GraphStore
from .writer import MemoryWriter
from .reader import MemoryReader
from .evolution import MemoryEvolution
from .models import Fact, ContextResult
from .token_utils import TokenBudget, SummaryCompressor, PrefetchCache

try:
    from agent.memory_provider import MemoryProvider
except ImportError:
    class MemoryProvider:  # type: ignore
        @property
        def name(self): raise NotImplementedError
        def is_available(self): raise NotImplementedError
        def initialize(self, session_id, **kwargs): raise NotImplementedError
        def get_tool_schemas(self): raise NotImplementedError


class SAGELiteProvider(MemoryProvider):
    """Hermes-compatible SAGE-lite Memory Provider v8"""

    PROVIDER_NAME = "sage_lite"

    def __init__(
        self,
        top_k: int = 5,
        max_hops: int = 2,
        max_tokens: int = 800,
        recall_mode: str = "balanced",
    ):
        self.top_k = top_k
        self.max_hops = max_hops
        self.max_tokens = max_tokens
        self.recall_mode = recall_mode
        self._hermes_home: Optional[Path] = None
        self._session_id: str = ""
        self._profile_name: str = "default"
        self._store: Optional[GraphStore] = None
        self._writer: Optional[MemoryWriter] = None
        self._reader: Optional[MemoryReader] = None
        self._evolution: Optional[MemoryEvolution] = None
        self._turn_count: int = 0
        self._compressor = SummaryCompressor()
        self._cache = PrefetchCache(ttl_seconds=30.0, max_size=50)
        self._write_failures: list[dict] = []

    # ── 必填 ABC ─────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.PROVIDER_NAME

    def is_available(self) -> bool:
        try:
            import networkx  # noqa
            return True
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        hermes_home = kwargs.get("hermes_home")
        if hermes_home:
            self._hermes_home = Path(hermes_home)
        agent_ctx = kwargs.get("agent_context", {})
        if isinstance(agent_ctx, dict):
            self._profile_name = agent_ctx.get("profile", "default")
        self._init_components()

    def get_tool_schemas(self) -> list[dict]:
        return [
            {
                "name": "sage_add_fact",
                "description": (
                    "Manually add a structured fact to SAGE-lite memory graph. "
                    "Use when the user states something important to remember."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "subject":   {"type": "string", "description": "The entity"},
                        "predicate": {"type": "string", "description": "The relationship"},
                        "object":    {"type": "string", "description": "The target"},
                        "weight":    {"type": "number", "default": 1.0,
                                      "description": "Confidence 0.0–1.0"},
                    },
                    "required": ["subject", "predicate", "object"],
                },
            },
            {
                "name": "sage_correct",
                "description": (
                    "Correct a memory fact. Use when user says something was wrong. "
                    "Actions: decay (reduce confidence), prune (delete), merge (combine)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "fact_id":   {"type": "string"},
                        "action":    {"type": "string",
                                      "enum": ["decay", "prune", "merge", "conflict_flag"]},
                        "target_id": {"type": "string",
                                      "description": "Required for merge action"},
                        "delta":     {"type": "number",
                                      "description": "Weight reduction amount for decay"},
                        "reason":    {"type": "string", "default": "user_correction"},
                    },
                    "required": ["fact_id", "action"],
                },
            },
            {
                "name": "sage_recall",
                "description": "Manually trigger a memory recall for a given query.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query":  {"type": "string"},
                        "top_k":  {"type": "integer", "default": 5},
                        "mode":   {"type": "string",
                                   "enum": ["precise", "balanced", "expansive"],
                                   "default": "balanced"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "sage_stats",
                "description": "Return memory graph health statistics.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    # ── 可選 Hooks ────────────────────────────────────────────

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        s = self._store.stats()
        return (
            f"[SAGE-lite Memory] "
            f"{s['active_facts']} active facts | "
            f"{s['node_count']} entities | "
            f"avg confidence {s['avg_weight']:.2f} | "
            f"profile: {self._profile_name}"
        )

    def prefetch(
        self,
        query: str,
        *,
        session_id: str,
        boost_tags: Optional[list[str]] = None,
    ) -> str:
        cached = self._cache.get(query)
        if cached is not None:
            return cached

        result = self._reader.retrieve_context(
            query,
            top_k=self.top_k,
            max_hops=self.max_hops,
            max_tokens=self.max_tokens,
            mode=self.recall_mode,
            boost_tags=boost_tags,
        )
        if result.is_empty:
            return ""

        budget = TokenBudget(self.max_tokens)
        summary = self._compressor.compress(result, budget)
        self._cache.set(query, summary)
        return summary

    def queue_prefetch(self, query: str, *, session_id: str) -> None:
        t = threading.Thread(
            target=self.prefetch,
            kwargs={"query": query, "session_id": session_id},
            daemon=True,
        )
        t.start()

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str,
    ) -> None:
        if not self._writer:
            return
        self._writer.write_turn(
            user_content, assistant_content, session_id=session_id
        )
        self._turn_count += 1
        self._cache.invalidate()
        if self._turn_count % 20 == 0:
            self._evolution.run_scheduled_decay()
            self._evolution.auto_resolve_conflicts()

    async def post_reply_commit(
        self,
        session_id: str,
        last_user_msg: str,
        agent_reply: str,
    ) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._writer.write_turn,
            last_user_msg,
            agent_reply,
            session_id,
        )
        self._cache.invalidate()

        if self._turn_count % 20 == 0:
            await loop.run_in_executor(
                None, self._evolution.run_scheduled_decay
            )
            await loop.run_in_executor(
                None, self._evolution.auto_resolve_conflicts
            )

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        try:
            return self._dispatch_tool(tool_name, args)
        except Exception as e:
            return json.dumps({"error": str(e), "tool": tool_name})

    def on_memory_write(
        self, action: str, target: str, content: str, metadata: dict
    ) -> None:
        if action in ("append", "write") and content and self._writer:
            self._writer.extract_and_write(
                content,
                subject_hint="system",
                session_id=self._session_id,
                source="inference",
            )
        elif action in ("correct", "update") and self._evolution:
            # content format: fact_id|new_weight or fact_id|delta|reason
            parts = content.split("|")
            if len(parts) >= 1:
                fact_id = parts[0]
                delta = float(parts[1]) if len(parts) > 1 else 0.1
                reason = parts[2] if len(parts) > 2 else "user_correction"
                self._evolution.apply_correction(
                    fact_id, "decay", delta=delta, reason=reason
                )
        elif action in ("delete", "forget") and self._evolution:
            self._evolution.apply_correction(target, "prune", reason="user_delete")

    def on_pre_compress(self, messages: list) -> str:
        if not self._store or self._store.edge_count == 0:
            return ""
        s = self._store.stats()
        return (
            f"[SAGE-lite] {s['active_facts']} facts persisted in graph memory. "
            f"Entities: {s['node_count']}. "
            "Long-term facts are safely stored and will not be lost during compression."
        )

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        if self._store:
            self._store.flush()
        self._session_id = new_session_id
        self._turn_count = 0
        agent_ctx = kwargs.get("agent_context", {})
        if isinstance(agent_ctx, dict):
            new_profile = agent_ctx.get("profile", self._profile_name)
            if new_profile != self._profile_name:
                self._profile_name = new_profile
                self._init_components()
                return
        if self._writer:
            self._writer.default_session_id = new_session_id

    def on_session_end(self, messages: list) -> None:
        if self._store:
            self._store.flush()

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        pass

    def shutdown(self) -> None:
        if self._store:
            self._store.flush()
            self._store.close()

    def get_config_schema(self) -> list[dict]:
        return [
            {"key": "top_k", "label": "Top-K results",
             "type": "int", "default": 5,
             "description": "Number of facts to retrieve per query"},
            {"key": "max_hops", "label": "Max graph hops",
             "type": "int", "default": 2,
             "description": "Depth of causal chain traversal (1–4)"},
            {"key": "max_tokens", "label": "Max context tokens",
             "type": "int", "default": 800,
             "description": "Token budget for injected memory context"},
            {"key": "recall_mode", "label": "Recall mode",
             "type": "str", "default": "balanced",
             "description": "precise / balanced / expansive"},
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        self.top_k       = int(values.get("top_k",       self.top_k))
        self.max_hops    = int(values.get("max_hops",    self.max_hops))
        self.max_tokens  = int(values.get("max_tokens",  self.max_tokens))
        self.recall_mode = str(values.get("recall_mode", self.recall_mode))

    # ── 工具分派 ──────────────────────────────────────────────

    def _dispatch_tool(self, tool_name: str, args: dict) -> str:
        if tool_name == "sage_add_fact":
            fact = Fact(
                subject=args["subject"],
                predicate=args["predicate"],
                object=args["object"],
                weight=float(args.get("weight", 1.0)),
                session_id=self._session_id,
                source="user",
            )
            result = self._writer.write_with_confirmation(fact)
            if result.has_failures:
                self._write_failures.append({
                    "fact": fact.to_dict(),
                    "errors": result.rejected,
                })
            fact_id = result.written[0] if result.written else (
                result.merged[0] if result.merged else ""
            )
            return json.dumps({"status": "ok" if fact_id else "error",
                               "fact_id": fact_id})

        elif tool_name == "sage_correct":
            ok = self._evolution.apply_correction(
                fact_id=args["fact_id"],
                action=args["action"],
                target_id=args.get("target_id"),
                delta=args.get("delta"),
                reason=args.get("reason", "user_correction"),
            )
            return json.dumps({"status": "ok" if ok else "not_found"})

        elif tool_name == "sage_recall":
            result = self._reader.retrieve_context(
                query=args["query"],
                top_k=int(args.get("top_k", self.top_k)),
                mode=args.get("mode", self.recall_mode),
            )
            return json.dumps({
                "summary":        result.summary,
                "fact_count":     len(result.facts),
                "chain_count":    len(result.chains),
                "token_estimate": result.token_estimate,
            })

        elif tool_name == "sage_stats":
            return json.dumps(self._store.stats())

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # ── 內部方法 ──────────────────────────────────────────────

    def _db_path(self) -> Path:
        base = self._hermes_home or Path.home() / ".hermes"
        return (
            base / "profiles" / self._profile_name
            / "sage_memory" / "graph.sqlite"
        )

    def _init_components(self) -> None:
        if self._store:
            self._store.close()
        self._store     = GraphStore(db_path=self._db_path())
        self._writer    = MemoryWriter(
            self._store, default_session_id=self._session_id
        )
        self._reader    = MemoryReader(
            self._store,
            on_retrieved=self._on_memory_retrieved,
        )
        self._evolution = MemoryEvolution(self._store)

    def _on_memory_retrieved(self, result: ContextResult) -> None:
        """
        Post-retrieval hook: 低分 facts 自動輕微 decay。
        """
        for fact in result.facts:
            score = result.retrieval_scores.get(fact.fact_id, 0.0)
            if score < 0.2 and not fact.is_anchor:
                self._evolution.apply_correction(
                    fact.fact_id, "decay",
                    delta=0.02,
                    reason="low_retrieval_score",
                )

    def get_write_health(self) -> dict:
        return {
            "total_write_failures": len(self._write_failures),
            "recent_failures": self._write_failures[-5:],
            "store_stats": self._store.stats() if self._store else {},
        }