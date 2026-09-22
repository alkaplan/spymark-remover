#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -q -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
exec .venv/bin/uvicorn app.main:app --port 8765
