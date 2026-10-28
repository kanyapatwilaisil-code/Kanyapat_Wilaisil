#!/bin/bash
set -e
APP_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$APP_DIR"
if [ ! -x ".venv/bin/python" ]; then
  echo "Please install the environment first. See README_TH.md."
  read -r -p "Press Enter to close..."
  exit 1
fi
exec .venv/bin/python app.py
