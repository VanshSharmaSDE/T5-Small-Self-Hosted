#!/usr/bin/env bash
set -e
exec gunicorn app.main:app --bind 0.0.0.0:${PORT:-8000} --workers 1
