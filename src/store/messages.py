"""Persist full conversation transcripts and their rolling summaries.

The graph no longer carries the transcript in its state. It is the authoritative
store's job to hold every message; the graph only ever sees a bounded window
assembled from this store (see ``src/agent/context_window.py``).
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class StoredMessage:
    """One persisted conversation message."""

    seq: int
    role: str  # "user" | "assistant"
    content: str
    created_at: str


@dataclass(frozen=True)
class ThreadSummary:
    """A rolling summary covering every message with ``seq <= upto_seq``."""

    text: str | None
    upto_seq: int


_EMPTY_SUMMARY = ThreadSummary(text=None, upto_seq=0)


@runtime_checkable
class ConversationStore(Protocol):
    """Read/write access to a thread's transcript and rolling summary."""

    def append_turn(
        self, thread_id: str, user_message: str, assistant_message: str
    ) -> None:
        """Append one user message and the assistant reply, in order."""

    def history(self, thread_id: str) -> list[StoredMessage]:
        """Return every message for a thread, oldest first."""

    def message_count(self, thread_id: str) -> int:
        """Return the number of messages stored for a thread."""

    def summary(self, thread_id: str) -> ThreadSummary:
        """Return the thread's rolling summary, or an empty summary."""

    def update_summary(self, thread_id: str, text: str, upto_seq: int) -> None:
        """Replace the thread's rolling summary."""


class InMemoryConversationStore:
    """Non-persistent store for the CLI, evaluation, and tests."""

    def __init__(self) -> None:
        """Create empty per-thread message and summary maps."""
        self._messages: dict[str, list[StoredMessage]] = {}
        self._summaries: dict[str, ThreadSummary] = {}
        self._lock = threading.Lock()

    def append_turn(
        self, thread_id: str, user_message: str, assistant_message: str
    ) -> None:
        """Append the user message and assistant reply for a thread."""
        now = datetime.now(UTC).isoformat()
        with self._lock:
            bucket = self._messages.setdefault(thread_id, [])
            for role, content in (
                ("user", user_message),
                ("assistant", assistant_message),
            ):
                bucket.append(
                    StoredMessage(
                        seq=len(bucket) + 1,
                        role=role,
                        content=content,
                        created_at=now,
                    )
                )

    def history(self, thread_id: str) -> list[StoredMessage]:
        """Return a copy of the thread's message list."""
        with self._lock:
            return list(self._messages.get(thread_id, []))

    def message_count(self, thread_id: str) -> int:
        """Return the thread's message count."""
        with self._lock:
            return len(self._messages.get(thread_id, []))

    def summary(self, thread_id: str) -> ThreadSummary:
        """Return the thread's rolling summary."""
        with self._lock:
            return self._summaries.get(thread_id, _EMPTY_SUMMARY)

    def update_summary(self, thread_id: str, text: str, upto_seq: int) -> None:
        """Replace the thread's rolling summary."""
        with self._lock:
            self._summaries[thread_id] = ThreadSummary(text=text, upto_seq=upto_seq)


class SqliteConversationStore:
    """SQLite-backed store, sharing the sidecar database file."""

    def __init__(self, db_path: Path | str) -> None:
        """Open (and if needed create) the transcript tables."""
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a short-lived connection to the sidecar database."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self._db_path)

    def _ensure_schema(self) -> None:
        """Create the messages and thread_summaries tables if absent."""
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "thread_id TEXT NOT NULL, "
                "seq INTEGER NOT NULL, "
                "role TEXT NOT NULL, "
                "content TEXT NOT NULL, "
                "created_at TEXT NOT NULL, "
                "UNIQUE(thread_id, seq))"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_thread "
                "ON messages (thread_id, seq)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS thread_summaries ("
                "thread_id TEXT PRIMARY KEY, "
                "summary TEXT NOT NULL, "
                "upto_seq INTEGER NOT NULL, "
                "updated_at TEXT NOT NULL)"
            )

    def append_turn(
        self, thread_id: str, user_message: str, assistant_message: str
    ) -> None:
        """Append the user message and assistant reply with consecutive seq values."""
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            next_seq = int(row[0]) + 1
            connection.executemany(
                "INSERT INTO messages (thread_id, seq, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (thread_id, next_seq, "user", user_message, now),
                    (thread_id, next_seq + 1, "assistant", assistant_message, now),
                ],
            )

    def history(self, thread_id: str) -> list[StoredMessage]:
        """Return every message for a thread, ordered by seq."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT seq, role, content, created_at FROM messages "
                "WHERE thread_id = ? ORDER BY seq",
                (thread_id,),
            ).fetchall()
        return [
            StoredMessage(seq=row[0], role=row[1], content=row[2], created_at=row[3])
            for row in rows
        ]

    def message_count(self, thread_id: str) -> int:
        """Return the thread's message count."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM messages WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        return int(row[0])

    def summary(self, thread_id: str) -> ThreadSummary:
        """Return the thread's rolling summary, or an empty one."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary, upto_seq FROM thread_summaries WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        if row is None:
            return _EMPTY_SUMMARY
        return ThreadSummary(text=row[0], upto_seq=int(row[1]))

    def update_summary(self, thread_id: str, text: str, upto_seq: int) -> None:
        """Insert or replace the thread's rolling summary."""
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO thread_summaries (thread_id, summary, upto_seq, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(thread_id) DO UPDATE SET "
                "summary = excluded.summary, "
                "upto_seq = excluded.upto_seq, "
                "updated_at = excluded.updated_at",
                (thread_id, text, upto_seq, now),
            )
