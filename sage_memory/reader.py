from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Literal, Optional

import networkx as nx

from .models import Fact, ContextResult
from .graph_store import GraphStore

MAX_TOKENS_DEFAULT = 800
CHARS_PER_TOKEN = 4
RecallMode = Literal["precise", "balanced", "expansive"]


class MemoryReader:
    """Reader v3：三維評分 + recall_mode + 結構化 summary"""

    def __init__(self, graph_store: GraphStore):
        self.store = graph_store

    # ── 主要公開 API ──────────────────────────────────────────

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        max_hops: int = 2,
        max_tokens: int = MAX_TOKENS_DEFAULT,
        min_weight: float = 0.1,
        mode: RecallMode = "balanced",
    ) -> ContextResult:
        """
        主要檢索入口，對應 Hermes prefetch() hook。
        三步驟：
          1. 關鍵詞提取 → 相關 facts 候選集
          2. 三維評分排序（weight × recency × relevance）
          3. 依 mode 組裝 summary（token 受控）
        """
        if self.store.edge_count == 0:
            return ContextResult(facts=[], chains=[], summary="", token_estimate=0)

        keywords = self._extract_keywords(query)
        if not keywords:
            # query 太短或全是停用詞 → fallback 到最新高 weight facts
            return self._fallback_recent(top_k, max_tokens)

        # Step 1：候選集
        candidates = self._gather_candidates(keywords, min_weight)
        if not candidates:
            return self._fallback_recent(top_k, max_tokens)

        # Step 2：三維評分
        scored = self._score_facts(candidates, keywords)

        # Step 3：依 mode 決定數量與展開深度
        if mode == "precise":
            top_facts = scored[:min(top_k, 3)]
            chains = []
        elif mode == "expansive":
            top_facts = scored[:top_k * 2]
            chains = self._build_chains(keywords, max_hops=3)
        else:  # balanced（預設）
            top_facts = scored[:top_k]
            chains = self._build_chains(keywords, max_hops)

        # Step 4：組裝 summary
        summary = self._build_summary(top_facts, chains, max_tokens, mode)
        token_est = len(summary) // CHARS_PER_TOKEN

        return ContextResult(
            facts=top_facts,
            chains=chains,
            summary=summary,
            token_estimate=token_est,
        )

    # ── 關鍵詞提取 ────────────────────────────────────────────

    def _extract_keywords(self, query: str) -> list[str]:
        import re
        stopwords = {
            # 英文
            "what", "who", "where", "when", "how", "why", "is", "are", "was",
            "were", "the", "a", "an", "i", "you", "he", "she", "they", "we",
            "do", "did", "does", "tell", "me", "about", "know", "can", "could",
            "would", "should", "has", "have", "had", "been", "be", "my", "your",
            "his", "her", "its", "our", "their", "this", "that", "which",
            # 中文
            "嗎", "的", "我", "你", "他", "她", "是", "在", "有", "了",
            "嗯", "呢", "吧", "啊", "哦", "喔", "什麼", "怎麼", "為什麼",
        }
        # 同時抓英文詞和中文字符串
        tokens = re.findall(r"[一-鿿]{2,}|[a-zA-Z]{2,}", query)
        result = [t for t in tokens if t.lower() not in stopwords]
        return result[:6]  # 最多 6 個關鍵詞

    # ── 候選集蒐集 ────────────────────────────────────────────

    def _gather_candidates(
        self, keywords: list[str], min_weight: float
    ) -> list[Fact]:
        seen_ids: set[str] = set()
        candidates: list[Fact] = []
        for kw in keywords:
            for fact in self.store.search_by_entity(kw, min_weight=min_weight):
                if fact.fact_id not in seen_ids:
                    candidates.append(fact)
                    seen_ids.add(fact.fact_id)
        return candidates

    # ── 三維評分 ──────────────────────────────────────────────

    def _score_facts(self, facts: list[Fact], keywords: list[str]) -> list[Fact]:
        """
        綜合分數 = weight × recency_score × relevance_score
        - weight：圖上儲存的信度（0~1）
        - recency：越新越高，半衰期 30 天
        - relevance：關鍵詞命中數 / 關鍵詞總數（TF 風格）
        """
        now = time.time()
        kw_lower = [k.lower() for k in keywords]

        def composite_score(f: Fact) -> float:
            # Recency：半衰期 30 天
            age_days = (now - f.timestamp) / 86400
            recency = math.exp(-age_days / 30.0)

            # Relevance：在 subject + predicate + object 中命中的關鍵詞比例
            text = f"{f.subject} {f.predicate} {f.object}".lower()
            hits = sum(1 for kw in kw_lower if kw in text)
            relevance = hits / max(len(kw_lower), 1)

            return f.weight * recency * (0.4 + 0.6 * relevance)

        return sorted(facts, key=composite_score, reverse=True)

    # ── 多跳鏈建構 ────────────────────────────────────────────

    def _build_chains(
        self, keywords: list[str], max_hops: int
    ) -> list[list[Fact]]:
        chains: list[list[Fact]] = []
        seen_chain_keys: set[frozenset] = set()

        for kw in keywords[:3]:
            ego = self.store.get_ego_graph(kw, radius=max_hops)
            if ego.number_of_edges() == 0:
                continue

            # 依 weight 排序取最強的 edges
            edges_sorted = sorted(
                ego.edges(data=True),
                key=lambda x: x[2].get("weight", 0),
                reverse=True,
            )[:6]

            chain: list[Fact] = []
            chain_key_parts: set[str] = set()

            for u, v, data in edges_sorted:
                fid = data.get("fact_id", "")
                if fid in chain_key_parts:
                    continue
                chain_key_parts.add(fid)
                chain.append(Fact(
                    subject=u,
                    predicate=data.get("predicate", "related_to"),
                    object=v,
                    timestamp=data.get("timestamp", 0.0),
                    weight=data.get("weight", 1.0),
                    source=data.get("source", "user"),
                    fact_id=fid,
                    session_id=data.get("session_id", ""),
                ))

            # 去重：相同邊集合的 chain 不重複加入
            chain_key = frozenset(f.fact_id for f in chain)
            if chain and chain_key not in seen_chain_keys:
                chains.append(chain)
                seen_chain_keys.add(chain_key)

        return chains

    # ── Summary 組裝 ──────────────────────────────────────────

    def _build_summary(
        self,
        facts: list[Fact],
        chains: list[list[Fact]],
        max_tokens: int,
        mode: RecallMode,
    ) -> str:
        budget_chars = max_tokens * CHARS_PER_TOKEN
        lines: list[str] = []
        used = 0

        # === 記憶區 ===
        if facts:
            header = "## Recalled Memory"
            lines.append(header)
            used += len(header)

            for f in facts:
                confidence = (
                    "high"   if f.weight >= 0.8 else
                    "medium" if f.weight >= 0.5 else
                    "low"
                )
                line = f"- [{confidence}] {f.subject} {f.predicate} {f.object}"
                if used + len(line) > budget_chars * 0.65:
                    lines.append(f"  ... (+{len(facts) - facts.index(f)} more)")
                    break
                lines.append(line)
                used += len(line)

        # === 因果鏈區（precise mode 跳過）===
        if chains and mode != "precise":
            header = "\n## Causal Chains"
            if used + len(header) < budget_chars:
                lines.append(header)
                used += len(header)

                for chain in chains[:2]:
                    parts = [
                        f"{f.subject}→[{f.predicate}]→{f.object}"
                        for f in chain[:4]
                    ]
                    line = "- " + " ⟹ ".join(parts)
                    if used + len(line) > budget_chars:
                        break
                    lines.append(line)
                    used += len(line)

        return "\n".join(lines)

    # ── Fallback ──────────────────────────────────────────────

    def _fallback_recent(self, top_k: int, max_tokens: int) -> ContextResult:
        """query 無有效關鍵詞時，回傳最近高 weight 的 facts"""
        facts = self.store.get_all_facts(min_weight=0.5)[:top_k]
        summary = self._build_summary(facts, [], max_tokens, "precise")
        return ContextResult(
            facts=facts,
            chains=[],
            summary=summary,
            token_estimate=len(summary) // CHARS_PER_TOKEN,
        )