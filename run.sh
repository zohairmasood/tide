#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/backend"
[ -f .env ] || cp .env.example .env
pip install -q -r requirements.txt
echo "Starting Tide on http://localhost:8000  (open ../frontend/index.html)"
uvicorn server:app --reload --port 8000
