import sqlite3
from pathlib import Path
from typing import Literal, TypedDict, List


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "chat_history.db"


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection, creating the DB file if needed."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,          -- 'user' or 'assistant'
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
            """
        )
        conn.commit()


def create_session() -> int:
    """Start a new chat session and return its ID."""
    with get_connection() as conn:
        cursor = conn.execute("INSERT INTO sessions DEFAULT VALUES")
        conn.commit()
        return int(cursor.lastrowid)


Role = Literal["user", "assistant"]


def add_message(session_id: int, role: Role, content: str) -> None:
    """Persist a single message for a session. Auto-creates the session if it doesn't exist."""
    with get_connection() as conn:
        # Auto-create session if it doesn't exist (check in same connection to avoid extra query)
        cursor = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,))
        if cursor.fetchone() is None:
            conn.execute("INSERT INTO sessions (id) VALUES (?)", (session_id,))
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()


class Message(TypedDict):
    role: Role
    content: str


def get_messages(session_id: int) -> list[Message]:
    """Load all messages for a given session, ordered by time."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        rows = cursor.fetchall()
    return [Message(role=row["role"], content=row["content"]) for row in rows]


class Session(TypedDict):
    id: int
    created_at: str


def list_sessions() -> List[Session]:
    """Return all chat sessions."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT id, created_at FROM sessions ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
    return [Session(id=row["id"], created_at=row["created_at"]) for row in rows]


def delete_session(session_id: int) -> bool:
    """Delete a chat session and all its messages. Returns True if session or messages existed."""
    with get_connection() as conn:
        # Delete messages first (cascade), then session
        msg_cursor = conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        session_cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        # Return True if either messages or session existed
        return msg_cursor.rowcount > 0 or session_cursor.rowcount > 0

