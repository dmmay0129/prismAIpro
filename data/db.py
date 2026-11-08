"""统一数据库访问层。

后端选择（按优先级）：

1. 配置了 ``DATABASE_URL``（Postgres/Supabase 连接串）时使用 Postgres，
   供 Streamlit Cloud 等无持久化文件系统的部署环境使用；
2. 未配置时退回本地 SQLite（``prism.db``），保证本地开发零配置可跑。

``DATABASE_URL`` 的来源：先读环境变量，再兜底读 Streamlit secrets
（Streamlit Cloud 的 secrets 不会自动注入到环境变量）。
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

# 本地 SQLite 文件路径：与历史版本一致，位于项目根目录。
SQLITE_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "prism.db")

_database_url: str | None = None
_url_resolved: bool = False


def get_database_url() -> str | None:
    """返回 Postgres 连接串；未配置时返回 None（回退 SQLite）。"""
    global _database_url, _url_resolved
    if _url_resolved:
        return _database_url
    _url_resolved = True

    url = (os.environ.get("DATABASE_URL") or "").strip() or None
    if url:
        _database_url = url
        return _database_url

    # Streamlit Cloud 的 secrets 只在 st.secrets 里可见，不会成为环境变量。
    try:
        import streamlit as st
    except Exception:
        return None
    try:
        value = st.secrets.get("DATABASE_URL", "") if hasattr(st, "secrets") else ""
    except Exception:
        value = ""
    _database_url = (value or "").strip() or None
    return _database_url


def is_postgres() -> bool:
    """当前是否使用 Postgres 后端。"""
    return get_database_url() is not None


def sql(query: str) -> str:
    """把源码里统一的 ``?`` 占位符按后端转换（Postgres 用 ``%s``）。"""
    if is_postgres():
        return query.replace("?", "%s")
    return query


def is_unique_violation(exc: BaseException) -> bool:
    """判断异常是否为「唯一约束冲突」（重复用户名等）。兼容 SQLite 与 psycopg2。"""
    name = type(exc).__name__
    module = type(exc).__module__ or ""
    # psycopg2 抛出的唯一约束冲突：psycopg2.errors.UniqueViolation
    if name == "UniqueViolation":
        return True
    # SQLite 抛出的唯一约束冲突：sqlite3.IntegrityError，消息含 "UNIQUE"
    if module == "sqlite3" and name == "IntegrityError":
        return "unique" in str(exc).lower()
    return False


class _PgCursor:
    """让 psycopg2 游标呈现与 sqlite3 游标一致的接口。"""

    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount


class _PgConnection:
    """让 psycopg2 连接呈现 ``conn.execute(sql, params)`` 接口。"""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def execute(self, query: str, params=()):
        cursor = self._conn.cursor()
        cursor.execute(query, params)
        return _PgCursor(cursor)


@contextmanager
def connect() -> Iterator[Any]:
    """返回统一连接：支持 ``conn.execute(sql, params)`` 及游标的
    ``fetchone()`` / ``fetchall()`` / ``rowcount``。

    成功退出时提交，异常时回滚。
    """
    if is_postgres():
        import psycopg2

        conn = psycopg2.connect(get_database_url())
        conn.autocommit = False
        wrapped = _PgConnection(conn)
        try:
            yield wrapped
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(SQLITE_DB_PATH, timeout=10)
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            conn.close()
