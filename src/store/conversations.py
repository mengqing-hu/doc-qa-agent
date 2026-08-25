"""Track per-visitor conversation metadata for the Streamlit sidebar."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


TITLE_MAX_LENGTH = 50


@dataclass(frozen=True)
class ConversationSummary:
    """Describe one conversation thread for the sidebar list."""

    thread_id: str
    title: str
    created_at: str


def create_conversation(db_path: Path, owner_uid: str, first_question: str) -> str:
    """Register a new conversation thread and return its thread_id."""
    normalized_owner_uid = owner_uid.strip()
    normalized_question = first_question.strip()
    if not normalized_owner_uid:
        raise ValueError("owner_uid must not be empty")
    if not normalized_question:
        raise ValueError("first_question must not be empty")

    thread_id = f"{normalized_owner_uid}:{uuid.uuid4().hex}"
    title = normalized_question[:TITLE_MAX_LENGTH]
    created_at = datetime.now(UTC).isoformat()

    with _connect(db_path) as connection:
        connection.execute(
            "INSERT INTO conversations (thread_id, owner_uid, title, created_at) "
            "VALUES (?, ?, ?, ?)",
            (thread_id, normalized_owner_uid, title, created_at),
        )
    return thread_id


def list_conversations(db_path: Path, owner_uid: str) -> list[ConversationSummary]:
    """Return this visitor's conversations, most recently created first."""
    normalized_owner_uid = owner_uid.strip()
    if not normalized_owner_uid:
        raise ValueError("owner_uid must not be empty")

    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT thread_id, title, created_at FROM conversations "
            "WHERE owner_uid = ? ORDER BY created_at DESC",
            (normalized_owner_uid,),
        ).fetchall()
    return [
        ConversationSummary(thread_id=row[0], title=row[1], created_at=row[2])
        for row in rows
    ]


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open the sidecar sqlite database, creating its table if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS conversations ("
        "thread_id TEXT PRIMARY KEY, "
        "owner_uid TEXT NOT NULL, "
        "title TEXT NOT NULL, "
        "created_at TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_owner_uid "
        "ON conversations (owner_uid)"
    )
    return connection
