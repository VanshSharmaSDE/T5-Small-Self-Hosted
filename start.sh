#!/usr/bin/env bash
set -e

# Run with Uvicorn (Render will set $PORT automatically)
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
