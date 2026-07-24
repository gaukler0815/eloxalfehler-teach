"""Zentrale Konfiguration des Familienkalenders.

Alle Werte lassen sich über Umgebungsvariablen anpassen, damit die App ohne
Code-Änderung auf einem eigenen Server (oder z. B. Fly.io / Railway / einem
Raspberry Pi zu Hause) betrieben werden kann.
"""
import os
from pathlib import Path

# Basisverzeichnisse -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # familienkalender/
DATA_DIR = Path(os.environ.get("FK_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Datenbank ----------------------------------------------------------------
DATABASE_URL = os.environ.get("FK_DATABASE_URL", f"sqlite:///{DATA_DIR / 'familienkalender.db'}")

# Sicherheit ---------------------------------------------------------------
# Schlüssel zum Signieren der Login-Tokens. In Produktion unbedingt setzen!
SECRET_KEY = os.environ.get("FK_SECRET_KEY", "bitte-in-produktion-aendern-langer-zufaelliger-wert")
TOKEN_TTL_SECONDS = int(os.environ.get("FK_TOKEN_TTL", 60 * 60 * 24 * 30))  # 30 Tage

# Zeitzone der Familie -----------------------------------------------------
# Alle Termine werden in dieser lokalen Zeit gespeichert und Erinnerungen
# danach berechnet.
FAMILY_TZ = os.environ.get("FK_TIMEZONE", "Europe/Berlin")

# Web-Push (VAPID) ---------------------------------------------------------
VAPID_PRIVATE_PATH = DATA_DIR / "vapid_private.pem"
VAPID_PUBLIC_PATH = DATA_DIR / "vapid_public.txt"
# Kontakt-Adresse für Push-Dienste (Pflicht laut Web-Push-Standard)
VAPID_CONTACT = os.environ.get("FK_VAPID_CONTACT", "mailto:familie@example.com")

# Uploads ------------------------------------------------------------------
MAX_UPLOAD_MB = int(os.environ.get("FK_MAX_UPLOAD_MB", 25))
ALLOWED_UPLOAD_TYPES = None  # None = alles erlaubt; sonst Set von MIME-Typen
