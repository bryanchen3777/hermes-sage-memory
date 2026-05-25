import re
import time
from typing import Optional
from .models import Fact
from .graph_store import GraphStore

# ── 關係 Pattern 庫（中英文擴展版）────────────────────────────
_RELATION_PATTERNS: list[tuple[str, str, float]] = [
    # (pattern, predicate, base_weight)

    # 身份 / 狀態
    (r"(.+?)\s+is\s+(?:a\s+|an\s+|the\s+)?(.+)",           "is",          1.0),
    (r"(.+?)\s+are\s+(.+)",                                  "is",          1.0),
    (r"(.+?)\s+was\s+(?:a\s+|an\s+)?(.+)",                  "was",         0.8),

    # 情感 / 偏好
    (r"(.+?)\s+(?:really\s+)?likes?\s+(.+)",                 "likes",       1.0),
    (r"(.+?)\s+loves?\s+(.+)",                               "loves",       1.0),
    (r"(.+?)\s+(?:really\s+)?hates?\s+(.+)",                 "hates",       1.0),
    (r"(.+?)\s+(?:enjoys?|enjoyed)\s+(.+)",                  "enjoys",      0.9),
    (r"(.+?)\s+(?:prefers?)\s+(.+)",                         "prefers",     0.9),
    (r"(.+?)\s+(?:dislikes?|doesn'?t like)\s+(.+)",          "dislikes",    1.0),

    # 職業 / 地點
    (r"(.+?)\s+(?:works?|worked)\s+(?:at|for|in)\s+(.+)",   "works_at",    1.0),
    (r"(.+?)\s+(?:lives?|lived)\s+(?:in|at|near)\s+(.+)",   "lives_in",    1.0),
    (r"(.+?)\s+(?:is\s+from|comes?\s+from)\s+(.+)",         "from",        0.9),
    (r"(.+?)\s+(?:studies?|studied)\s+(?:at\s+)?(.+)",      "studies_at",  0.9),

    # 關係
    (r"(.+?)\s+(?:knows?|knew)\s+(.+)",                      "knows",       0.9),
    (r"(.+?)\s+(?:has|have|had)\s+(?:a\s+|an\s+)?(.+)",     "has",         0.9),
    (r"(.+?)\s+(?:wants?|wanted|need[s]?)\s+(.+)",           "wants",       0.8),
    (r"(.+?)\s+(?:remembers?)\s+(.+)",                       "remembers",   0.9),
    (r"(.+?)\s+(?:said|says|told)\s+(.+)",                   "said",        0.7),
    (r"(.+?)\s+(?:feels?|felt)\s+(.+)",                      "feels",       0.8),
    (r"(.+?)\s+(?:thinks?|thought)\s+(.+)",                  "thinks",      0.7),
    (r"(.+?)\s+(?:believes?)\s+(.+)",                        "believes",    0.8),

    # 目標 / 計畫
    (r"(.+?)\s+(?:plans?\s+to|wants?\s+to|going\s+to)\s+(.+)", "plans_to", 0.8),
    (r"(.+?)\s+(?:will|would)\s+(.+)",                       "will",        0.6),

    # 中文模式
    (r"(.+?)喜歡(.+)",   "喜歡",  1.0),
    (r"(.+?)討厭(.+)",   "討厭",  1.0),
    (r"(.+?)住在(.+)",   "住在",  1.0),
    (r"(.+?)在(.+?)工作", "工作於", 1.0),
    (r"(.+?)是(.+)",     "是",    1.0),
    (r"(.+?)有(.+)",     "有",    0.9),
    (r"(.+?)想要(.+)",   "想要",  0.8),
    (r"(.+?)記得(.+)",   "記得",  0.9),
    (r"(.+?)認識(.+)",   "認識",  0.9),
    (r"(.+?)覺得(.+)",   "覺得",  0.8),
]

# 停用的 object 片段（過於模糊，不值得存）
_NOISE_OBJECTS = {
    "it", "this", "that", "things", "something", "anything",
    "him", "her", "them", "there", "here", "now", "then",
}

# subject 別名正規化（第一人稱統一為 user）
_FIRST_PERSON = {"i", "me", "my", "myself", "i'm", "i've", "i'd"}


class MemoryWriter:
    """Writer v2：擴展 pattern + 實體正規化 + 近似重複偵測"""

    def __init__(self, graph_store: GraphStore, default_session_id: str = ""):
        self.store = graph_store
        self.default_session_id = default_session_id

    # ── 公開 API（維持 Phase 1 相同簽名）────────────────────────

    def add_fact(self, fact: Fact) -> str:
        if not fact.session_id:
            fact.session_id = self.default_session_id
        # 近似重複偵測：同 subject+predicate 的舊 fact 進行 weight 疊加而非新增
        existing = self._find_similar(fact)
        if existing:
            merged_weight = min(1.0, existing.weight + fact.weight * 0.3)
            self.store.update_weight(existing.fact_id, merged_weight)
            return existing.fact_id
        return self.store.add_fact(fact)

    def add_facts_batch(self, facts: list[Fact]) -> list[str]:
        return [self.add_fact(f) for f in facts]

    def extract_and_write(
        self,
        text: str,
        subject_hint: Optional[str] = None,
        session_id: Optional[str] = None,
        source: str = "user",
    ) -> list[str]:
        sid = session_id or self.default_session_id
        facts = self._extract_facts(text, subject_hint, sid, source)
        return self.add_facts_batch(facts)

    def write_turn(
        self,
        user_content: str,
        assistant_content: str,
        session_id: Optional[str] = None,
    ) -> list[str]:
        sid = session_id or self.default_session_id
        # user 說的話 weight 較高（直接陳述），assistant 推斷 weight 較低
        user_ids = self.extract_and_write(
            user_content, subject_hint="user", session_id=sid, source="user"
        )
        assistant_ids = self.extract_and_write(
            assistant_content, subject_hint="assistant", session_id=sid, source="inference"
        )
        return user_ids + assistant_ids

    # ── 內部方法 ──────────────────────────────────────────────

    def _extract_facts(
        self,
        text: str,
        subject_hint: Optional[str],
        session_id: str,
        source: str,
    ) -> list[Fact]:
        facts: list[Fact] = []
        seen: set[tuple[str, str, str]] = set()  # (subject, predicate, object) 去重

        # 分句：支援標點、連接詞、換行
        sentences = re.split(
            r"[.!?。！？\n]+|(?:\s+(?:and|but|however|although|so|then)\s+)",
            text,
            flags=re.IGNORECASE,
        )

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 4:
                continue

            for pattern, predicate, base_weight in _RELATION_PATTERNS:
                m = re.search(pattern, sentence, re.IGNORECASE)
                if not m:
                    continue

                subj_raw = m.group(1).strip()
                obj_raw  = m.group(2).strip()

                # 正規化
                subj = self._normalize_entity(subj_raw, subject_hint)
                obj  = self._normalize_object(obj_raw)

                # 過濾無效片段
                if not subj or not obj:
                    continue
                if obj.lower() in _NOISE_OBJECTS:
                    continue
                if len(subj) > 60 or len(obj) > 100:
                    continue

                key = (subj.lower(), predicate, obj.lower())
                if key in seen:
                    continue
                seen.add(key)

                # source 影響初始 weight
                weight = base_weight
                if source == "inference":
                    weight *= 0.8

                facts.append(Fact(
                    subject=subj,
                    predicate=predicate,
                    object=obj,
                    timestamp=time.time(),
                    weight=weight,
                    source=source,
                    session_id=session_id,
                ))
                # 一句可以匹配多個 pattern（不 break）

        return facts

    def _normalize_entity(self, raw: str, subject_hint: Optional[str]) -> str:
        """正規化主語：第一人稱統一、去除冠詞、首字母大寫"""
        cleaned = raw.strip().lower()
        # 第一人稱 → subject_hint（通常是 "user"）
        if cleaned in _FIRST_PERSON:
            return subject_hint or "user"
        # 去除開頭冠詞
        cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
        # 首字母大寫
        return cleaned.capitalize() if cleaned else ""

    def _normalize_object(self, raw: str) -> str:
        """正規化受詞：去除尾部標點、冠詞"""
        cleaned = raw.strip()
        cleaned = re.sub(r"[,.;:]+$", "", cleaned)  # 去除尾部標點
        cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _find_similar(self, fact: Fact) -> Optional[Fact]:
        """
        檢查是否已有 subject+predicate 相同的 fact。
        相同 → 合併 weight，不重複寫入。
        """
        existing = self.store.search_by_entity(fact.subject, min_weight=0.01)
        for e in existing:
            if (e.subject.lower() == fact.subject.lower()
                    and e.predicate == fact.predicate
                    and e.object.lower() == fact.object.lower()):
                return e
        return None