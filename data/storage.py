"""数据管理层：持久化评估结果、完整报告与按用户隔离的历史。

后端选择：配置了 ``DATABASE_URL``（Postgres/Supabase）时使用 Postgres，
否则退回本地 SQLite（``prism.db``）。详见 :mod:`data.db`。

原始音视频不入库；仅保存结构化分析结果和完整报告 JSON。
``user_id`` 对应登录用户名；首次登录时可把升级前的旧记录认领给该用户。
用户名比较统一使用 ``LOWER()``，跨两种后端保持大小写不敏感。
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

from core.settings import get_settings
from data.db import SQLITE_DB_PATH, connect, is_postgres, sql


def _ensure_schema_sqlite(conn) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS results (
               session_id  TEXT PRIMARY KEY,
               category    TEXT,
               overall     REAL,
               dims        TEXT,
               radar       TEXT,
               created_at  REAL,
               user_id     TEXT,
               report_json TEXT
           )"""
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    if "user_id" not in cols:
        conn.execute("ALTER TABLE results ADD COLUMN user_id TEXT")
    if "report_json" not in cols:
        conn.execute("ALTER TABLE results ADD COLUMN report_json TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_results_user_created "
        "ON results(user_id, created_at)"
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS knowledge_base (
               user_id TEXT PRIMARY KEY COLLATE NOCASE,
               payload TEXT NOT NULL,
               updated_at REAL NOT NULL
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS interview_turns (
               session_id TEXT NOT NULL,
               turn_index INTEGER NOT NULL,
               user_id TEXT NOT NULL COLLATE NOCASE,
               question TEXT NOT NULL,
               transcript TEXT NOT NULL,
               overall REAL NOT NULL,
               report_json TEXT NOT NULL,
               created_at REAL NOT NULL,
               PRIMARY KEY(session_id, turn_index)
           )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_interview_turns_user_created "
        "ON interview_turns(user_id, created_at)"
    )


def _ensure_schema_postgres(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS results ("
        " session_id TEXT PRIMARY KEY,"
        " category TEXT,"
        " overall DOUBLE PRECISION,"
        " dims TEXT,"
        " radar TEXT,"
        " created_at DOUBLE PRECISION,"
        " user_id TEXT,"
        " report_json TEXT)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_results_user_created "
        "ON results(user_id, created_at)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS knowledge_base ("
        " user_id TEXT PRIMARY KEY,"
        " payload TEXT NOT NULL,"
        " updated_at DOUBLE PRECISION NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS interview_turns ("
        " session_id TEXT NOT NULL,"
        " turn_index INTEGER NOT NULL,"
        " user_id TEXT NOT NULL,"
        " question TEXT NOT NULL,"
        " transcript TEXT NOT NULL,"
        " overall DOUBLE PRECISION NOT NULL,"
        " report_json TEXT NOT NULL,"
        " created_at DOUBLE PRECISION NOT NULL,"
        " PRIMARY KEY(session_id, turn_index))"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_interview_turns_user_created "
        "ON interview_turns(user_id, created_at)"
    )


def _ensure_schema(conn) -> None:
    if is_postgres():
        _ensure_schema_postgres(conn)
    else:
        _ensure_schema_sqlite(conn)


@contextmanager
def _conn():
    with connect() as conn:
        _ensure_schema(conn)
        yield conn


def _has_local_db() -> bool:
    """读取类操作在无本地库且未配 Postgres 时直接返回空，避免创建空文件。"""
    return is_postgres() or os.path.exists(SQLITE_DB_PATH)


def save_result(report, username: str | None = None) -> None:
    """保存评估结果和完整报告；username 为空时保留为历史遗留记录。"""
    report_data = report.to_dict()
    report_data.pop("transcript", None)
    payload = json.dumps(report_data, ensure_ascii=False)
    with _conn() as conn:
        conn.execute(
            sql(
                "INSERT INTO results "
                "(session_id, category, overall, dims, radar, created_at, "
                " user_id, report_json) "
                "VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "category=excluded.category, overall=excluded.overall, "
                "dims=excluded.dims, radar=excluded.radar, "
                "created_at=excluded.created_at, user_id=excluded.user_id, "
                "report_json=excluded.report_json"
            ),
            (
                report.session_id,
                report.question.get("category", ""),
                report.overall_score,
                json.dumps(report.dimension_scores, ensure_ascii=False),
                json.dumps(report.radar, ensure_ascii=False),
                report.created_at,
                username,
                payload,
            ),
        )


def save_history_entry(
    username: str,
    session_id: str,
    category: str,
    overall: float,
    dims: dict[str, float],
    created_at: float | None = None,
) -> None:
    """通话面试整场结束后写入一条历史趋势记录（无完整报告 JSON）。

    每场通话只写一条（整场平均分），与单题评估的历史记录共用 results 表。
    """
    if created_at is None:
        created_at = time.time()
    with _conn() as conn:
        conn.execute(
            sql(
                "INSERT INTO results "
                "(session_id, category, overall, dims, radar, created_at, "
                " user_id, report_json) "
                "VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "category=excluded.category, overall=excluded.overall, "
                "dims=excluded.dims, radar=excluded.radar, "
                "created_at=excluded.created_at, user_id=excluded.user_id, "
                "report_json=excluded.report_json"
            ),
            (
                session_id,
                category,
                round(float(overall), 1),
                json.dumps(dims, ensure_ascii=False),
                json.dumps(dims, ensure_ascii=False),
                created_at,
                username,
                None,
            ),
        )


def load_history(limit: int | None = None, username: str | None = None) -> list:
    """历史趋势数据：按时间正序返回近期练习记录；有用户时只返回该用户。"""
    if limit is None:
        limit = int(get_settings()["storage"]["history_limit"])
    if not _has_local_db():
        return []
    with _conn() as conn:
        if username:
            rows = conn.execute(
                sql(
                    "SELECT session_id, overall, dims, created_at FROM results "
                    "WHERE LOWER(user_id) = LOWER(?) "
                    "ORDER BY created_at DESC LIMIT ?"
                ),
                (username, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                sql(
                    "SELECT session_id, overall, dims, created_at FROM results "
                    "ORDER BY created_at DESC LIMIT ?"
                ),
                (limit,),
            ).fetchall()

    history = []
    for session_id, overall, dims, created_at in reversed(rows):
        try:
            parsed_dims = json.loads(dims)
        except (TypeError, json.JSONDecodeError):
            parsed_dims = {}
        history.append(
            {
                "session_id": session_id,
                "overall": overall,
                "dims": parsed_dims,
                "created_at": created_at,
            }
        )
    return history


def claim_legacy_results(username: str) -> int:
    """把升级前没有 user_id 的记录认领给当前用户（首次登录时调用）。"""
    if not username:
        return 0
    with _conn() as conn:
        cursor = conn.execute(
            sql("UPDATE results SET user_id = ? WHERE user_id IS NULL"),
            (username,),
        )
        return cursor.rowcount


def load_report(session_id: str, username: str | None = None) -> dict | None:
    """读取完整报告 JSON；用于历史详情等扩展。"""
    query = "SELECT report_json FROM results WHERE session_id = ?"
    params: list = [session_id]
    if username:
        query += " AND LOWER(user_id) = LOWER(?)"
        params.append(username)
    with _conn() as conn:
        row = conn.execute(sql(query), params).fetchone()
    if not row or not row[0]:
        return None
    try:
        return json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return None


def load_knowledge(username: str) -> dict:
    """读取按用户隔离的面试知识库。"""
    if not username or not _has_local_db():
        return {}
    with _conn() as conn:
        row = conn.execute(
            sql(
                "SELECT payload FROM knowledge_base "
                "WHERE LOWER(user_id) = LOWER(?)"
            ),
            (username,),
        ).fetchone()
    if not row:
        return {}
    try:
        payload = json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_knowledge(username: str, payload: dict) -> None:
    """持久化候选人背景、项目证据、岗位要求和追问风格。"""
    if not username:
        raise ValueError("保存知识库需要已登录用户")
    fields = ("candidate_profile", "project_evidence", "job_description",
              "interviewer_style", "interview_review")
    normalized = {key: str(payload.get(key, "")).strip()[:6000] for key in fields}
    encoded = json.dumps(normalized, ensure_ascii=False)
    with _conn() as conn:
        cursor = conn.execute(
            sql(
                "UPDATE knowledge_base SET payload = ?, updated_at = ? "
                "WHERE LOWER(user_id) = LOWER(?)"
            ),
            (encoded, time.time(), username),
        )
        if cursor.rowcount == 0:
            conn.execute(
                sql(
                    "INSERT INTO knowledge_base(user_id, payload, updated_at) "
                    "VALUES(?,?,?)"
                ),
                (username, encoded, time.time()),
            )


def save_interview_turn(
    session_id: str, turn_index: int, username: str, question: str,
    transcript: str, report,
) -> None:
    """保存问答文本和评分，不保存原始音视频。"""
    if not username or not session_id:
        raise ValueError("保存面试记录需要用户和会话标识")
    with _conn() as conn:
        conn.execute(
            sql(
                "INSERT INTO interview_turns "
                "(session_id, turn_index, user_id, question, transcript, "
                " overall, report_json, created_at) "
                "VALUES(?,?,?,?,?,?,?,?) "
                "ON CONFLICT(session_id, turn_index) DO UPDATE SET "
                "user_id=excluded.user_id, question=excluded.question, "
                "transcript=excluded.transcript, overall=excluded.overall, "
                "report_json=excluded.report_json, created_at=excluded.created_at"
            ),
            (
                session_id, int(turn_index), username, question[:1000],
                transcript[:12000], report.overall_score,
                json.dumps(report.to_dict(), ensure_ascii=False), time.time(),
            ),
        )


def list_interview_sessions(username: str, limit: int = 20) -> list[dict]:
    """返回当前用户最近的多轮模拟面试列表。"""
    if not username or not _has_local_db():
        return []
    with _conn() as conn:
        rows = conn.execute(
            sql(
                "SELECT session_id, COUNT(*), MAX(created_at), AVG(overall) "
                "FROM interview_turns "
                "WHERE LOWER(user_id) = LOWER(?) GROUP BY session_id "
                "ORDER BY MAX(created_at) DESC LIMIT ?"
            ),
            (username, max(1, int(limit))),
        ).fetchall()
    result = []
    for row in rows:
        average = None
        try:
            average = round(float(row[3]), 1)
        except (TypeError, ValueError):
            average = None
        result.append(
            {"session_id": row[0], "turn_count": row[1],
             "created_at": row[2], "average_score": average}
        )
    return result


def load_interview_turns(session_id: str, username: str) -> list[dict]:
    """按用户权限读取一场模拟面试的问答与报告。"""
    if not session_id or not username or not _has_local_db():
        return []
    with _conn() as conn:
        rows = conn.execute(
            sql(
                "SELECT turn_index, question, transcript, overall, "
                "report_json, created_at "
                "FROM interview_turns WHERE session_id = ? "
                "AND LOWER(user_id) = LOWER(?) ORDER BY turn_index"
            ),
            (session_id, username),
        ).fetchall()
    turns = []
    for row in rows:
        try:
            report = json.loads(row[4])
        except (TypeError, json.JSONDecodeError):
            report = {}
        turns.append({
            "turn_index": row[0], "question": row[1], "transcript": row[2],
            "overall": row[3], "report": report, "created_at": row[5],
        })
    return turns
