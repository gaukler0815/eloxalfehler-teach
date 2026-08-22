"""SQLite-Schicht.

Bewusst SQLite: laeuft ohne Server, die Datei laesst sich kopieren und
sichern. Wenn die Schaltzentrale spaeter auf SQL Server oder MySQL liegt,
wird nur diese Datei ersetzt -- das Schema ist bewusst so gehalten, dass
es sich 1:1 uebertragen laesst.

Grundsatz: die Tabelle 'event' wird nur angehaengt, nie ueberschrieben.
Der aktuelle Standort eines Gestells ist immer das juengste Event dazu.
Damit ist die Historie automatisch nachvollziehbar -- das braucht man
sowohl fuer Durchlaufzeiten als auch fuer die Fehlerrueckverfolgung.
"""

import os
import sqlite3

from . import config

STANDARD_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gestelle.db")


def db_pfad():
    """Pfad zur Datenbankdatei.

    Wird bei jedem Aufruf neu gelesen, damit sich GESTELLE_DB auch nach dem
    Import noch umstellen laesst -- das brauchen die Tests, und im Betrieb
    laesst sich damit ohne Codeaenderung auf eine andere Datei zeigen.
    """
    return os.environ.get("GESTELLE_DB", STANDARD_DB)

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS linie (
    linie_id TEXT PRIMARY KEY,
    name     TEXT NOT NULL,
    art      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS station (
    station_id     TEXT PRIMARY KEY,
    linie_id       TEXT NOT NULL REFERENCES linie(linie_id),
    name           TEXT NOT NULL,
    reihenfolge    INTEGER NOT NULL,
    art            TEXT NOT NULL,
    soll_dauer_min INTEGER
);

-- Stammdaten Gestell.
-- epc ist NULL solange das Gestell noch keinen RFID-Tag hat. Das ist beim
-- rollierenden Ausruesten von 400 Gestellen der Normalfall, nicht die
-- Ausnahme -- die Anwendung muss damit sauber umgehen koennen.
CREATE TABLE IF NOT EXISTS gestell (
    gestell_id  TEXT PRIMARY KEY,
    epc         TEXT UNIQUE,
    typ         TEXT,
    plaetze     INTEGER,
    status      TEXT NOT NULL DEFAULT 'aktiv',   -- aktiv | gesperrt | ausgemustert
    notiz       TEXT,
    angelegt_am TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS auftrag (
    auftrag_nr   TEXT PRIMARY KEY,
    kunde        TEXT,
    artikel      TEXT,
    menge        INTEGER,
    farbe        TEXT,
    schichtdicke TEXT,
    liefertermin TEXT,
    status       TEXT NOT NULL DEFAULT 'offen',  -- offen | laufend | fertig
    angelegt_am  TEXT NOT NULL
);

-- Verknuepfung Auftrag <-> Gestell, zeitlich begrenzt.
-- Ein Gestell wird immer wieder neu belegt, und es kann mehrere Auftraege
-- gleichzeitig tragen. Deshalb n:m mit von/bis und nicht einfach eine
-- Spalte 'auftrag' am Gestell.
-- bis IS NULL bedeutet: Belegung laeuft gerade.
CREATE TABLE IF NOT EXISTS belegung (
    belegung_id INTEGER PRIMARY KEY AUTOINCREMENT,
    gestell_id  TEXT NOT NULL REFERENCES gestell(gestell_id),
    auftrag_nr  TEXT NOT NULL REFERENCES auftrag(auftrag_nr),
    menge       INTEGER,
    von         TEXT NOT NULL,
    bis         TEXT,
    bemerkung   TEXT
);

-- Standort-Events. Nur anhaengen.
CREATE TABLE IF NOT EXISTS event (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    gestell_id  TEXT NOT NULL REFERENCES gestell(gestell_id),
    station_id  TEXT NOT NULL REFERENCES station(station_id),
    linie_id    TEXT NOT NULL REFERENCES linie(linie_id),
    quelle      TEXT NOT NULL,
    epc_gelesen TEXT,
    rssi        INTEGER,
    bemerkung   TEXT
);

CREATE INDEX IF NOT EXISTS idx_event_gestell   ON event(gestell_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_event_ts        ON event(ts DESC);
CREATE INDEX IF NOT EXISTS idx_belegung_offen  ON belegung(gestell_id, bis);
CREATE INDEX IF NOT EXISTS idx_belegung_auftr  ON belegung(auftrag_nr, bis);
"""


def connect(pfad=None):
    """Verbindung oeffnen. Legt Schema und Stammdaten an, falls noch nicht da."""
    pfad = pfad or db_pfad()
    if pfad != ":memory:":
        ordner = os.path.dirname(os.path.abspath(pfad))
        if ordner:
            os.makedirs(ordner, exist_ok=True)
    con = sqlite3.connect(pfad, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    _stammdaten_sichern(con)
    return con


def _stammdaten_sichern(con):
    """Linien und Stationen aus config.py in die DB spiegeln (idempotent)."""
    for l in config.LINIEN:
        con.execute(
            "INSERT INTO linie (linie_id, name, art) VALUES (?,?,?) "
            "ON CONFLICT(linie_id) DO UPDATE SET name=excluded.name, art=excluded.art",
            (l["linie_id"], l["name"], l["art"]),
        )
    for s in config.STATIONEN:
        con.execute(
            "INSERT INTO station (station_id, linie_id, name, reihenfolge, art, soll_dauer_min) "
            "VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(station_id) DO UPDATE SET "
            "  linie_id=excluded.linie_id, name=excluded.name, "
            "  reihenfolge=excluded.reihenfolge, art=excluded.art, "
            "  soll_dauer_min=excluded.soll_dauer_min",
            (s["station_id"], s["linie_id"], s["name"], s["reihenfolge"],
             s["art"], s["soll_dauer_min"]),
        )
    con.commit()


def leeren(con):
    """Bewegungsdaten loeschen, Stammdaten behalten. Fuer Demo/Test."""
    con.execute("DELETE FROM event")
    con.execute("DELETE FROM belegung")
    con.execute("DELETE FROM auftrag")
    con.execute("DELETE FROM gestell")
    con.commit()
