"""Apply schema changes to an existing database (safe to run multiple times).

Run from the project root:
    py db/migrate_existing.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asyncpg
from movementscreen.config import settings

MIGRATIONS = [
    (
        "Add is_admin column to users",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE",
    ),
    (
        "Create threshold_config table",
        """
        CREATE TABLE IF NOT EXISTS threshold_config (
            key         VARCHAR(100) PRIMARY KEY,
            value       DOUBLE PRECISION NOT NULL,
            description TEXT,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
    ),
]


async def main() -> None:
    conn = await asyncpg.connect(settings.database_url)
    try:
        for description, sql in MIGRATIONS:
            print(f"  {description}...", end=" ")
            await conn.execute(sql)
            print("done")
        print("Migration complete.")
    finally:
        await conn.close()


asyncio.run(main())
