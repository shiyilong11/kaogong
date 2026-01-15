#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip setuptools wheel
pip install pyinstaller
if [ -f requirements.txt ]; then
  pip install -r requirements.txt || true
fi
pyinstaller --noconfirm --onefile --name kaogong 11.py
echo "Build complete. Output: dist/kaogong"
