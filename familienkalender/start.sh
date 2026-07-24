#!/usr/bin/env bash
# Startet den Familienkalender lokal (ohne Docker).
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Erstelle virtuelle Umgebung…"
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r backend/requirements.txt

# Optional: eigene Einstellungen
export FK_TIMEZONE="${FK_TIMEZONE:-Europe/Berlin}"

echo "Familienkalender läuft auf http://localhost:8000"
cd backend
exec uvicorn app:app --host 0.0.0.0 --port 8000
