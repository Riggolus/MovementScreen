#!/usr/bin/env bash
# Container startup script.
# 1. Initialise / migrate the database schema (idempotent — safe on every start).
# 2. Launch the FastAPI server.
set -euo pipefail

echo "==> [1/2] Running schema init..."
python db/schema_init.py

echo "==> [2/2] Starting server on port ${PORT:-10000}..."
exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-10000}"
