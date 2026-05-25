from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from .graph_store import GraphStore
from .models import Fact

DecayAction = Literal["decay", "prune", "merge", "conflict_flag"]


@dataclass
class EvolutionEvent:
    """演化操作日誌單筆記錄"""
    action: str
    fact_id: str
    target_id: Optional[str]
    old_weight: float
    new_weight: float
    timestamp: float = field(default_factory=time.time)
    reason: str = ""

    def to_log_line(self) -> str:
        return (
            f"[{self.action.upper()}] fact={self.fact_id[:8]} "
            f"w:{self.old_weight:.2f}→{self.new_weight:.2f} "
            f"reason={self.reason}"
        )


# 依 source 差異化的衰減率
_DECAY_RATES: dict[str, float] = {
    "user":        0.03,   # 用戶親口說的衰減慢
    "inference":   0.08,   # 推斷的衰減快
    "correction":  0.02,   # 已修正的保留久
}

# 矛盾謂詞對（互斥關係）
_CONFLICT_PREDICATES: list[frozenset[str]] = [
    frozenset({"likes",    "hates"}),
    frozenset({"likes",    "dislikes"}),
    frozenset({"loves",    "hates"}),
    frozenset({"is",       "was"}),
    frozenset({"lives_in", "lived_in"}),
    frozenset({"喜歡",      "討厭"}),
]


class MemoryEvolution:
    """Self-Correction Loop v4：動態衰減 + 衝突偵測 + 演化日誌"""

    PRUNE_THRESHOLD  = 0.05
    CONFLICT_PENALTY = 0.3   # 矛盾事實的 weight 懲罰

    def __init__(self, graph_store: GraphStore):
        self.store = graph_store
        self._log: list[EvolutionEvent] = []

    # ── 主要公開 API ──────────────────────────────────────────

    def apply_correction(
        self,
        fact_id: str,
        action: DecayAction,
        target_id: Optional[str] = None,
        delta: Optional[float] = None,
        reason: str = "manual",
    ) -> bool:
        """
        對應 Hermes feedback_correction hook。
        action:
          decay        → 降低 weight（delta 可覆寫預設衰減率）
          prune        → 直接刪除
          merge        → 合併到 target_id（語意相容性檢查）
          conflict_flag→ 標記矛盾並降權
        """
        if action == "decay":
            return self._decay(fact_id, delta, reason)
        elif action == "prune":
            return self._prune(fact_id, reason)
        elif action == "merge":
            return self._merge(fact_id, target_id, reason) if target_id else False
        elif action == "conflict_flag":
            return self._conflict_flag(fact_id, reason)
        return False

    def run_scheduled_decay(
        self,
        age_days_threshold: float = 7.0,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """
        定期排程執行：
        - 對超過 age_days_threshold 天的 facts 依 source 差異化衰減
        - weight 低於 PRUNE_THRESHOLD 自動剪除
        - dry_run=True 只回報不執行

        回傳：{"decayed": n, "pruned": n, "skipped": n}
        """
        now = time.time()
        stats = {"decayed": 0, "pruned": 0, "skipped": 0}

        for fact in self.store.get_all_facts(min_weight=0.0):
            age_days = (now - fact.timestamp) / 86400
            if age_days <= age_days_threshold:
                stats["skipped"] += 1
                continue

            rate = _DECAY_RATES.get(fact.source, 0.05)
            # 每超過一天額外衰減一次（補償長時間未運行）
            extra_days = max(0, age_days - age_days_threshold)
            effective_rate = min(0.5, rate * (1 + extra_days / 30))
            new_weight = fact.weight * (1.0 - effective_rate)

            if dry_run:
                stats["decayed"] += 1
                continue

            if new_weight < self.PRUNE_THRESHOLD:
                self._prune(fact.fact_id, reason="scheduled_decay_threshold")
                stats["pruned"] += 1
            else:
                self.store.update_weight(fact.fact_id, new_weight)
                self._record(EvolutionEvent(
                    action="decay",
                    fact_id=fact.fact_id,
                    target_id=None,
                    old_weight=fact.weight,
                    new_weight=new_weight,
                    reason=f"scheduled(age={age_days:.1f}d,rate={effective_rate:.2f})",
                ))
                stats["decayed"] += 1

        return stats

    def detect_conflicts(self) -> list[tuple[Fact, Fact]]:
        """
        掃描全圖，找出所有矛盾事實對（相同 subject，互斥 predicate）。
        回傳 list of (fact_a, fact_b) 供外部決策。
        """
        conflicts: list[tuple[Fact, Fact]] = []
        all_facts = self.store.get_all_facts(min_weight=self.PRUNE_THRESHOLD)

        # 按 subject 分組
        by_subject: dict[str, list[Fact]] = {}
        for f in all_facts:
            by_subject.setdefault(f.subject.lower(), []).append(f)

        for subj_facts in by_subject.values():
            for i, fa in enumerate(subj_facts):
                for fb in subj_facts[i + 1:]:
                    if self._are_conflicting(fa, fb):
                        conflicts.append((fa, fb))

        return conflicts

    def auto_resolve_conflicts(self) -> int:
        """
        自動解決衝突：
        - 保留 weight 較高的，降權 weight 較低的
        - 回傳處理的衝突對數
        """
        resolved = 0
        for fa, fb in self.detect_conflicts():
            loser = fa if fa.weight <= fb.weight else fb
            self._conflict_flag(loser.fact_id, reason="auto_resolve")
            resolved += 1
        return resolved

    @property
    def evolution_log(self) -> list[EvolutionEvent]:
        return list(self._log)

    def log_summary(self) -> str:
        if not self._log:
            return "No evolution events recorded."
        counts: dict[str, int] = {}
        for e in self._log:
            counts[e.action] = counts.get(e.action, 0) + 1
        parts = [f"{a}×{n}" for a, n in sorted(counts.items())]
        return f"Evolution log: {len(self._log)} events — " + ", ".join(parts)

    # ── 內部操作 ──────────────────────────────────────────────

    def _decay(
        self,
        fact_id: str,
        delta: Optional[float],
        reason: str,
    ) -> bool:
        fact = self.store.get_fact(fact_id)
        if not fact:
            return False
        rate = delta if delta is not None else _DECAY_RATES.get(fact.source, 0.05)
        new_weight = max(0.0, fact.weight - rate)
        if new_weight < self.PRUNE_THRESHOLD:
            return self._prune(fact_id, reason=f"{reason}→below_threshold")
        ok = self.store.update_weight(fact_id, new_weight)
        if ok:
            self._record(EvolutionEvent("decay", fact_id, None,
                                        fact.weight, new_weight, reason=reason))
        return ok

    def _prune(self, fact_id: str, reason: str = "manual") -> bool:
        fact = self.store.get_fact(fact_id)
        old_w = fact.weight if fact else 0.0
        ok = self.store.remove_fact(fact_id)
        if ok:
            self._record(EvolutionEvent("prune", fact_id, None,
                                        old_w, 0.0, reason=reason))
        return ok

    def _merge(
        self,
        source_id: str,
        target_id: str,
        reason: str,
    ) -> bool:
        source = self.store.get_fact(source_id)
        target = self.store.get_fact(target_id)
        if not source or not target:
            return False

        # 語意相容性檢查：矛盾謂詞不允許合併
        if self._are_conflicting(source, target):
            # 改為 conflict_flag 而非合併
            self._conflict_flag(source_id, reason=f"merge_rejected_conflict→{reason}")
            return False

        merged_weight = min(1.0, target.weight + source.weight * 0.5)
        self.store.update_weight(target_id, merged_weight)
        self._record(EvolutionEvent("merge", source_id, target_id,
                                    source.weight, 0.0, reason=reason))
        return self._prune(source_id, reason=f"merged_into:{target_id[:8]}")

    def _conflict_flag(self, fact_id: str, reason: str = "conflict") -> bool:
        fact = self.store.get_fact(fact_id)
        if not fact:
            return False
        new_weight = max(0.0, fact.weight - self.CONFLICT_PENALTY)
        if new_weight < self.PRUNE_THRESHOLD:
            return self._prune(fact_id, reason=f"{reason}→pruned")
        ok = self.store.update_weight(fact_id, new_weight)
        if ok:
            self._record(EvolutionEvent("conflict_flag", fact_id, None,
                                        fact.weight, new_weight, reason=reason))
        return ok

    def _are_conflicting(self, fa: Fact, fb: Fact) -> bool:
        """判斷兩條 fact 是否語意矛盾"""
        if fa.subject.lower() != fb.subject.lower():
            return False
        pair = frozenset({fa.predicate, fb.predicate})
        return pair in _CONFLICT_PREDICATES

    def _record(self, event: EvolutionEvent) -> None:
        self._log.append(event)
        # 日誌上限 1000 條（避免無限增長）
        if len(self._log) > 1000:
            self._log = self._log[-800:]