from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import networkx as nx

from .models import Fact

# 每累積 N 次寫入才 commit（WAL 模式下安全）
_BATCH_SIZE = 20
_SCHEMA_VERSION = 2


class GraphStore:
    """
    NetworkX MultiDiGraph + SQLite 持久化 v5
    新增：WAL 模式、schema migration、export/import、stats、context manager
    """

    def __init__(self, db_path: Path, batch_size: int = _BATCH_SIZE):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._conn: Optional[sqlite3.Connection] = None
        self._pending_writes: int = 0
        self._init_db()
        self._load_from_db()

    # ── Context Manager ───────────────────────────────────────

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── 初始化 ────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=10.0,
            )
            self._conn.row_factory = sqlite3.Row
            # WAL 模式：允許讀寫並行，不鎖整個 DB
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=-8000")   # 8MB cache
            self._conn.execute("PRAGMA temp_store=MEMORY")
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        # Schema version 表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        # 取得目前版本
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key='version'"
        ).fetchone()
        current_version = int(row["value"]) if row else 0

        # Migration
        self._migrate(conn, current_version)

        conn.execute(
            "INSERT OR REPLACE INTO schema_meta VALUES ('version', ?)",
            (str(_SCHEMA_VERSION),),
        )
        conn.commit()

    def _migrate(self, conn: sqlite3.Connection, from_version: int) -> None:
        """依序執行所有需要的 migration"""
        if from_version < 1:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    fact_id    TEXT PRIMARY KEY,
                    subject    TEXT NOT NULL,
                    predicate  TEXT NOT NULL,
                    object     TEXT NOT NULL,
                    timestamp  REAL NOT NULL,
                    weight     REAL NOT NULL DEFAULT 1.0,
                    source     TEXT NOT NULL DEFAULT 'user',
                    session_id TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_subject ON facts(subject)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_object ON facts(object)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_weight ON facts(weight)"
            )

        if from_version < 2:
            # v2：新增 tags 欄位、session_id 索引
            try:
                conn.execute(
                    "ALTER TABLE facts ADD COLUMN tags TEXT NOT NULL DEFAULT ''"
                )
            except sqlite3.OperationalError:
                pass  # 欄位已存在（重跑 migration 時）
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session "
                "ON facts(session_id)"
            )

    def _load_from_db(self) -> None:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM facts WHERE weight > 0.01"
        ).fetchall()
        for row in rows:
            d = dict(row)
            d.pop("tags", None)   # tags 欄位不在 Fact dataclass 中
            fact = Fact(**d)
            self._add_to_graph(fact)

    # ── Graph 操作 ────────────────────────────────────────────

    def _add_to_graph(self, fact: Fact) -> None:
        self.graph.add_node(fact.subject)
        self.graph.add_node(fact.object)
        self.graph.add_edge(
            fact.subject, fact.object,
            key=fact.fact_id,
            predicate=fact.predicate,
            timestamp=fact.timestamp,
            weight=fact.weight,
            source=fact.source,
            session_id=fact.session_id,
            fact_id=fact.fact_id,
        )

    def add_fact(self, fact: Fact) -> str:
        self._add_to_graph(fact)
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO facts
               (fact_id, subject, predicate, object,
                timestamp, weight, source, session_id, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (fact.fact_id, fact.subject, fact.predicate, fact.object,
             fact.timestamp, fact.weight, fact.source, fact.session_id, ""),
        )
        self._pending_writes += 1
        if self._pending_writes >= self.batch_size:
            conn.commit()
            self._pending_writes = 0
        return fact.fact_id

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d.pop("tags", None)
        return Fact(**d)

    def update_weight(self, fact_id: str, new_weight: float) -> bool:
        found = False
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            if data.get("fact_id") == fact_id:
                self.graph[u][v][k]["weight"] = new_weight
                found = True
                break
        if not found:
            return False
        conn = self._get_conn()
        conn.execute(
            "UPDATE facts SET weight = ? WHERE fact_id = ?",
            (new_weight, fact_id),
        )
        self._pending_writes += 1
        if self._pending_writes >= self.batch_size:
            conn.commit()
            self._pending_writes = 0
        return True

    def remove_fact(self, fact_id: str) -> bool:
        for u, v, k, data in list(self.graph.edges(keys=True, data=True)):
            if data.get("fact_id") == fact_id:
                self.graph.remove_edge(u, v, k)
                conn = self._get_conn()
                conn.execute(
                    "DELETE FROM facts WHERE fact_id = ?", (fact_id,)
                )
                conn.commit()
                self._pending_writes = 0
                return True
        return False

    def get_ego_graph(self, node: str, radius: int = 2) -> nx.MultiDiGraph:
        if node not in self.graph:
            return nx.MultiDiGraph()
        return nx.ego_graph(
            self.graph, node, radius=radius, undirected=True
        )

    def get_all_facts(self, min_weight: float = 0.05) -> list[Fact]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM facts WHERE weight >= ? ORDER BY timestamp DESC",
            (min_weight,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d.pop("tags", None)
            result.append(Fact(**d))
        return result

    def search_by_entity(
        self, entity: str, min_weight: float = 0.1
    ) -> list[Fact]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM facts
               WHERE (subject LIKE ? OR predicate LIKE ? OR object LIKE ?)
                 AND weight >= ?
               ORDER BY weight DESC, timestamp DESC""",
            (f"%{entity}%", f"%{entity}%", f"%{entity}%", min_weight),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d.pop("tags", None)
            result.append(Fact(**d))
        return result

    # ── 統計 ──────────────────────────────────────────────────

    def stats(self) -> dict:
        """回傳圖健康指標，供 adapter.system_prompt_block() 使用"""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        active = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE weight >= 0.1"
        ).fetchone()[0]
        avg_w = conn.execute(
            "SELECT AVG(weight) FROM facts WHERE weight >= 0.1"
        ).fetchone()[0] or 0.0
        by_source = conn.execute(
            "SELECT source, COUNT(*) as cnt FROM facts GROUP BY source"
        ).fetchall()
        oldest = conn.execute(
            "SELECT MIN(timestamp) FROM facts WHERE weight >= 0.1"
        ).fetchone()[0] or time.time()

        return {
            "total_facts":    total,
            "active_facts":   active,
            "pruned_facts":   total - active,
            "avg_weight":     round(avg_w, 3),
            "node_count":     self.node_count,
            "edge_count":     self.edge_count,
            "source_breakdown": {r["source"]: r["cnt"] for r in by_source},
            "oldest_fact_days": round((time.time() - oldest) / 86400, 1),
            "db_path":        str(self.db_path),
        }

    # ── Export / Import ───────────────────────────────────────

    def export_json(self, path: Path) -> int:
        """匯出所有 facts 為 JSON Lines 格式，回傳匯出筆數"""
        facts = self.get_all_facts(min_weight=0.0)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for fact in facts:
                f.write(json.dumps(fact.to_dict(), ensure_ascii=False) + "\n")
        return len(facts)

    def import_json(
        self,
        path: Path,
        overwrite: bool = False,
        min_weight: float = 0.05,
    ) -> dict[str, int]:
        """
        從 JSON Lines 匯入 facts。
        overwrite=True：清空現有資料再匯入
        回傳：{"imported": n, "skipped": n}
        """
        if not path.exists():
            return {"imported": 0, "skipped": 0}

        if overwrite:
            conn = self._get_conn()
            conn.execute("DELETE FROM facts")
            conn.commit()
            self.graph.clear()

        imported = skipped = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    fact = Fact.from_dict(d)
                    if fact.weight >= min_weight:
                        self.add_fact(fact)
                        imported += 1
                    else:
                        skipped += 1
                except (json.JSONDecodeError, TypeError, KeyError):
                    skipped += 1
        self.flush()
        return {"imported": imported, "skipped": skipped}

    # ── Lifecycle ─────────────────────────────────────────────

    def flush(self) -> None:
        """強制 commit 所有 pending writes"""
        if self._conn and self._pending_writes > 0:
            self._conn.commit()
            self._pending_writes = 0

    def close(self) -> None:
        if self._conn:
            self.flush()
            self._conn.close()
            self._conn = None

    def vacuum(self) -> None:
        """釋放 SQLite 碎片空間（建議在大量 prune 後呼叫）"""
        self.flush()
        conn = self._get_conn()
        conn.execute("VACUUM")

    # ── Properties ────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()