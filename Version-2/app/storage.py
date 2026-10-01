"""SQLite persistence: conversations, messages and the ingested-document registry."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False) if self.path == ":memory:" else None
        self._init()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = self._memory_conn or sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            if not self._memory_conn:
                conn.close()

    def _init(self) -> None:
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations(
                    id TEXT PRIMARY KEY, title TEXT, created_at TEXT, updated_at TEXT);
                CREATE TABLE IF NOT EXISTS messages(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT,
                    role TEXT, content TEXT, meta TEXT, created_at TEXT);
                CREATE TABLE IF NOT EXISTS documents(
                    document_id TEXT PRIMARY KEY, document_name TEXT, source_file TEXT,
                    revision_date TEXT, page_count INTEGER, chunk_count INTEGER,
                    ingested_at TEXT, meta TEXT);
                """
            )

    # ---------------------------------------------------------------- conversations
    def create_conversation(self, title: str = "New chat") -> str:
        cid = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute("INSERT INTO conversations VALUES (?,?,?,?)", (cid, title[:80], _now(), _now()))
        return cid

    def list_conversations(self, limit: int = 30) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def rename_conversation(self, cid: str, title: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE conversations SET title=? WHERE id=?", (title[:80], cid))

    def delete_conversation(self, cid: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
            c.execute("DELETE FROM conversations WHERE id=?", (cid,))

    def add_message(self, cid: str, role: str, content: str, meta: dict[str, Any] | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO messages(conversation_id, role, content, meta, created_at) VALUES (?,?,?,?,?)",
                (cid, role, content, json.dumps(meta or {}, default=str), _now()),
            )
            c.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now(), cid))

    def get_messages(self, cid: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY id", (cid,)).fetchall()
        return [{"role": r["role"], "content": r["content"], "meta": json.loads(r["meta"] or "{}")} for r in rows]

    # -------------------------------------------------------------------- documents
    def upsert_document(self, info: dict[str, Any], chunk_count: int) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?)",
                (
                    info["document_id"], info["document_name"], info["source_file"], info["revision_date"],
                    info.get("page_count", 0), chunk_count, _now(), json.dumps(info.get("pdf_metadata", {}), default=str),
                ),
            )

    def list_documents(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM documents ORDER BY ingested_at DESC").fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, document_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM documents WHERE document_id=?", (document_id,))
