#!/usr/bin/env bash
set -e
exec gunicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1