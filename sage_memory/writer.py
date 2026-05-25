import re
import time
import datetime
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

# 時間詞解析（簡易版，無需 dateparser）
_DATE_PATTERNS = [
    (r"(\d{1,2})[月/\-](\d{1,2})[日號]?", "month_day"),
    (r"(\d{4})[年/\-](\d{1,2})[月/\-](\d{1,2})", "full_date"),
    (r"明天|tomorrow",   "tomorrow"),
    (r"後天",             "day_after_tomorrow"),
    (r"下週|next week",   "next_week"),
    (r"下個月|next month","next_month"),
]

# 停用的 object 片段（過於模糊，不值得存）
_NOISE_OBJECTS = {
    "it", "this", "that", "things", "something", "anything",
    "him", "her", "them", "there", "here", "now", "then",
}

# subject 別名正規化（第一人稱統一為 user）
_FIRST_PERSON = {"i", "me", "my", "myself", "i'm", "i've", "i'd"}


class MemoryWriter:
    """Writer v3：entity 對齊 + event_time 解析 + anchor auto-set"""

    def __init__(self, graph_store: GraphStore, default_session_id: str = ""):
        self.store = graph_store
        self.default_session_id = default_session_id

    # ── 公開 API ──────────────────────────────────────────────

    def add_fact(self, fact: Fact) -> str:
        if not fact.session_id:
            fact.session_id = self.default_session_id

        # W-1：entity 對齊（零依賴版）
        fact.subject = self._align_entity(fact.subject)
        fact.object  = self._align_entity(fact.object)

        # 重複偵測
        existing = self._find_similar(fact)
        if existing:
            new_weight = min(1.0, existing.weight + fact.weight * 0.3)
            if new_weight >= 1.8 and not existing.is_anchor:
                self.store.set_anchor(existing.fact_id, True)
            else:
                self.store.update_weight(existing.fact_id, new_weight)
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
        user_ids = self.extract_and_write(
            user_content, subject_hint="user", session_id=sid, source="user"
        )
        assistant_ids = self.extract_and_write(
            assistant_content, subject_hint="assistant", session_id=sid, source="inference"
        )
        return user_ids + assistant_ids

    # ── 內部方法 ──────────────────────────────────────────────

    def _align_entity(self, name: str) -> str:
        """若圖中有相似實體則對齊，防止碎片化"""
        similar = self.store.find_similar_entity(name, threshold=0.75)
        return similar if similar else name

    def _extract_event_time(self, text: str) -> Optional[float]:
        """從句子中嘗試解析事件時間，找不到回傳 None"""
        now = time.time()
        for pattern, tag in _DATE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if tag == "tomorrow":
                    return now + 86400
                elif tag == "day_after_tomorrow":
                    return now + 86400 * 2
                elif tag == "next_week":
                    return now + 86400 * 7
                elif tag == "next_month":
                    return now + 86400 * 30
                elif tag == "month_day":
                    m = re.search(pattern, text)
                    month, day = int(m.group(1)), int(m.group(2))
                    try:
                        t = datetime.datetime(
                            datetime.datetime.now().year, month, day
                        )
                        return t.timestamp()
                    except ValueError:
                        pass
                elif tag == "full_date":
                    m = re.search(pattern, text)
                    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    try:
                        t = datetime.datetime(year, month, day)
                        return t.timestamp()
                    except ValueError:
                        pass
        return None

    def _extract_facts(
        self,
        text: str,
        subject_hint: Optional[str],
        session_id: str,
        source: str,
    ) -> list[Fact]:
        facts: list[Fact] = []
        seen: set[tuple[str, str, str]] = set()

        sentences = re.split(
            r"[.!?。！？\n]+|(?:\s+(?:and|but|however|although|so|then)\s+)",
            text,
            flags=re.IGNORECASE,
        )

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 4:
                continue

            event_time = self._extract_event_time(sentence)

            for pattern, predicate, base_weight in _RELATION_PATTERNS:
                m = re.search(pattern, sentence, re.IGNORECASE)
                if not m:
                    continue

                subj_raw = m.group(1).strip()
                obj_raw  = m.group(2).strip()

                subj = self._normalize_entity(subj_raw, subject_hint)
                obj  = self._normalize_object(obj_raw)

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

                weight = base_weight
                if source == "inference":
                    weight *= 0.8

                facts.append(Fact(
                    subject=subj,
                    predicate=predicate,
                    object=obj,
                    timestamp=time.time(),
                    event_time=event_time,
                    weight=weight,
                    source=source,
                    session_id=session_id,
                ))

        return facts

    def _normalize_entity(self, raw: str, subject_hint: Optional[str]) -> str:
        """正規化主語：第一人稱統一、去除冠詞、首字母大寫"""
        cleaned = raw.strip().lower()
        if cleaned in _FIRST_PERSON:
            return subject_hint or "user"
        cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
        return cleaned.capitalize() if cleaned else ""

    def _normalize_object(self, raw: str) -> str:
        """正規化受詞：去除尾部標點、冠詞"""
        cleaned = raw.strip()
        cleaned = re.sub(r"[,.;:]+$", "", cleaned)
        cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _find_similar(self, fact: Fact) -> Optional[Fact]:
        existing = self.store.search_by_entity(fact.subject, min_weight=0.01)
        for e in existing:
            if (e.subject.lower() == fact.subject.lower()
                    and e.predicate == fact.predicate
                    and e.object.lower() == fact.object.lower()):
                return e
        return None