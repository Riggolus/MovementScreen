"""DEV-ONLY script: create an admin user directly via the database.

Run from the project root:
    py db/seed_admin_dev.py

Delete this user once you have finished tweaking thresholds:
    py db/seed_admin_dev.py --delete
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asyncpg
from movementscreen.auth import hash_password
from movementscreen.config import settings

EMAIL = "admin@admin.com"
NAME = "Admin"
PASSWORD = "password"


async def main(delete: bool = False) -> None:
    conn = await asyncpg.connect(settings.database_url)
    try:
        if delete:
            result = await conn.execute("DELETE FROM users WHERE email = $1", EMAIL)
            print(f"Deleted admin user ({result}).")
        else:
            hashed = hash_password(PASSWORD)
            await conn.execute(
                """
                INSERT INTO users (email, name, password_hash, is_admin)
                VALUES ($1, $2, $3, TRUE)
                ON CONFLICT (email) DO UPDATE
                    SET is_admin      = TRUE,
                        password_hash = EXCLUDED.password_hash
                """,
                EMAIL, NAME, hashed,
            )
            print(f"Admin user ready — email: {EMAIL}  password: {PASSWORD}")
    finally:
        await conn.close()


asyncio.run(main(delete="--delete" in sys.argv))
