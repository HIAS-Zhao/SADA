# -*- coding: utf-8 -*-
import json
import os
import sqlite3
from typing import Any, Optional


class GPTCache:
    """
    Simple SQLite cache for GPT-judge calls.

    Stores arbitrary JSON-serializable objects by string key.
    """

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS kv (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[Any]:
        cur = self._conn.cursor()
        cur.execute("SELECT v FROM kv WHERE k = ?", (key,))
        row = cur.fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
