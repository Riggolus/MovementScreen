"""asyncpg connection pool and all database queries."""
from __future__ import annotations

import uuid
from typing import Optional

import asyncpg

_pool: Optional[asyncpg.Pool] = None


async def init_pool(dsn: str) -> None:
    global _pool
    _pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialised.")
    return _pool


# ── Users ────────────────────────────────────────────────

async def create_user(pool: asyncpg.Pool, email: str, name: str, password_hash: str) -> dict:
    uid = uuid.uuid4()
    row = await pool.fetchrow(
        """
        INSERT INTO users (id, email, name, password_hash)
        VALUES ($1, $2, $3, $4)
        RETURNING id, email, name, created_at
        """,
        uid, email, name, password_hash,
    )
    return dict(row)


async def get_user_by_email(pool: asyncpg.Pool, email: str) -> Optional[dict]:
    row = await pool.fetchrow("SELECT * FROM users WHERE email = $1", email)
    return dict(row) if row else None


async def get_user_by_id(pool: asyncpg.Pool, user_id: str) -> Optional[dict]:
    row = await pool.fetchrow(
        "SELECT id, email, name, created_at FROM users WHERE id = $1",
        uuid.UUID(user_id),
    )
    return dict(row) if row else None


# ── Assessments ──────────────────────────────────────────

async def save_assessment(
    pool: asyncpg.Pool,
    user_id: str,
    result: dict,
    screen: str,
    camera_angle: str,
    lead_side: str,
) -> str:
    assessment_id = uuid.uuid4()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO assessments
                  (id, user_id, screen_type, camera_angle, lead_side,
                   frame_count, worst_severity, has_findings)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                assessment_id, uuid.UUID(user_id), screen, camera_angle, lead_side,
                result["frame_count"], result["worst_severity"], result["has_findings"],
            )
            if result["findings"]:
                await conn.executemany(
                    """
                    INSERT INTO findings
                      (id, assessment_id, name, severity, description, metric_value, metric_label)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    [
                        (
                            uuid.uuid4(), assessment_id,
                            f["name"], f["severity"], f["description"],
                            f["metric_value"], f["metric_label"],
                        )
                        for f in result["findings"]
                    ],
                )
            if result["stats"]:
                await conn.executemany(
                    """
                    INSERT INTO angle_stats
                      (id, assessment_id, name, min_value, max_value, mean_value)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    [
                        (
                            uuid.uuid4(), assessment_id,
                            s["name"], s["min"], s["max"], s["mean"],
                        )
                        for s in result["stats"]
                    ],
                )
    return str(assessment_id)


async def list_assessments(
    pool: asyncpg.Pool,
    user_id: str,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT id, screen_type, camera_angle, lead_side,
               frame_count, worst_severity, has_findings, recorded_at
        FROM assessments
        WHERE user_id = $1
        ORDER BY recorded_at DESC
        LIMIT $2 OFFSET $3
        """,
        uuid.UUID(user_id), limit, offset,
    )
    return [dict(r) for r in rows]


async def get_assessment_detail(
    pool: asyncpg.Pool,
    user_id: str,
    assessment_id: str,
) -> Optional[dict]:
    row = await pool.fetchrow(
        """
        SELECT id, screen_type, camera_angle, lead_side,
               frame_count, worst_severity, has_findings, recorded_at
        FROM assessments
        WHERE id = $1 AND user_id = $2
        """,
        uuid.UUID(assessment_id), uuid.UUID(user_id),
    )
    if not row:
        return None
    detail = dict(row)
    detail["findings"] = [
        dict(r) for r in await pool.fetch(
            """SELECT name, severity, description, metric_value, metric_label
               FROM findings WHERE assessment_id = $1""",
            uuid.UUID(assessment_id),
        )
    ]
    detail["stats"] = [
        dict(r) for r in await pool.fetch(
            """SELECT name, min_value, max_value, mean_value
               FROM angle_stats WHERE assessment_id = $1""",
            uuid.UUID(assessment_id),
        )
    ]
    return detail


async def get_progress(pool: asyncpg.Pool, user_id: str) -> dict:
    rows = await pool.fetch(
        """
        SELECT screen_type, worst_severity, recorded_at
        FROM assessments
        WHERE user_id = $1
        ORDER BY recorded_at ASC
        """,
        uuid.UUID(user_id),
    )
    by_screen: dict = {}
    for r in rows:
        st = r["screen_type"]
        by_screen.setdefault(st, []).append({
            "recorded_at": r["recorded_at"].isoformat(),
            "worst_severity": r["worst_severity"],
        })
    return {"by_screen": by_screen}
