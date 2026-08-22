"""Fachlogik: Gestelle, Auftraege, Belegung, Standort-Events.

Diese Datei kennt keine Oberflaeche und keine Hardware. Sie ist damit
sowohl von der Streamlit-App als auch spaeter von einem echten
RFID-Reader-Dienst benutzbar -- und sie laesst sich testen.
"""

from datetime import datetime, timedelta

from . import config

ZEITFORMAT = "%Y-%m-%dT%H:%M:%S"

# RFID-Reader feuern mehrfach pro Sekunde. Ohne Entprellung stehen nach
# einer Schicht Zehntausende identischer Events in der Datenbank. Wird
# dasselbe Gestell innerhalb dieses Fensters erneut an derselben Station
# gelesen, gilt es als dieselbe Vorbeifahrt.
ENTPRELL_SEKUNDEN = 120


def jetzt():
    return datetime.now().replace(microsecond=0)


def as_ts(wert):
    """Zeitstempel -> String fuer die DB."""
    if wert is None:
        wert = jetzt()
    if isinstance(wert, str):
        return wert
    return wert.strftime(ZEITFORMAT)


def as_dt(wert):
    """String aus der DB -> datetime."""
    if wert is None:
        return None
    if isinstance(wert, datetime):
        return wert
    return datetime.strptime(wert[:19], ZEITFORMAT)


# --------------------------------------------------------------------------
# Gestelle
# --------------------------------------------------------------------------

def gestell_anlegen(con, gestell_id, typ=None, plaetze=None, epc=None,
                    notiz=None, ts=None):
    con.execute(
        "INSERT INTO gestell (gestell_id, epc, typ, plaetze, status, notiz, angelegt_am) "
        "VALUES (?,?,?,?,'aktiv',?,?) "
        "ON CONFLICT(gestell_id) DO UPDATE SET "
        "  typ=excluded.typ, plaetze=excluded.plaetze, notiz=excluded.notiz",
        (gestell_id, epc, typ, plaetze, notiz, as_ts(ts)),
    )
    con.commit()
    return gestell_id


def gestell_taggen(con, gestell_id, epc):
    """RFID-Tag einem Gestell zuordnen.

    Das passiert beim rollierenden Ausruesten am Aushaengeplatz: Gestell
    kommt leer zurueck, Tag drauf, Nummer zuordnen, fertig.
    """
    vorhanden = con.execute(
        "SELECT gestell_id FROM gestell WHERE epc = ? AND gestell_id <> ?",
        (epc, gestell_id),
    ).fetchone()
    if vorhanden:
        raise ValueError(
            f"EPC {epc} ist bereits Gestell {vorhanden['gestell_id']} zugeordnet."
        )
    con.execute("UPDATE gestell SET epc = ? WHERE gestell_id = ?", (epc, gestell_id))
    con.commit()


def gestell_by_epc(con, epc):
    return con.execute("SELECT * FROM gestell WHERE epc = ?", (epc,)).fetchone()


def gestell(con, gestell_id):
    return con.execute(
        "SELECT * FROM gestell WHERE gestell_id = ?", (gestell_id,)
    ).fetchone()


def gestelle(con, nur_aktive=True):
    sql = "SELECT * FROM gestell"
    if nur_aktive:
        sql += " WHERE status = 'aktiv'"
    sql += " ORDER BY gestell_id"
    return con.execute(sql).fetchall()


def gestelle_ohne_tag(con):
    """Noch nicht ausgeruestete Gestelle -- der Rollout-Fortschritt."""
    return con.execute(
        "SELECT * FROM gestell WHERE epc IS NULL AND status = 'aktiv' ORDER BY gestell_id"
    ).fetchall()


# --------------------------------------------------------------------------
# Auftraege
# --------------------------------------------------------------------------

def auftrag_anlegen(con, auftrag_nr, kunde=None, artikel=None, menge=None,
                    farbe=None, schichtdicke=None, liefertermin=None, ts=None):
    con.execute(
        "INSERT INTO auftrag (auftrag_nr, kunde, artikel, menge, farbe, "
        "  schichtdicke, liefertermin, status, angelegt_am) "
        "VALUES (?,?,?,?,?,?,?,'offen',?) "
        "ON CONFLICT(auftrag_nr) DO UPDATE SET "
        "  kunde=excluded.kunde, artikel=excluded.artikel, menge=excluded.menge, "
        "  farbe=excluded.farbe, schichtdicke=excluded.schichtdicke, "
        "  liefertermin=excluded.liefertermin",
        (auftrag_nr, kunde, artikel, menge, farbe, schichtdicke,
         liefertermin, as_ts(ts)),
    )
    con.commit()
    return auftrag_nr


def auftrag(con, auftrag_nr):
    return con.execute(
        "SELECT * FROM auftrag WHERE auftrag_nr = ?", (auftrag_nr,)
    ).fetchone()


def auftraege(con, status=None):
    if status:
        return con.execute(
            "SELECT * FROM auftrag WHERE status = ? ORDER BY auftrag_nr", (status,)
        ).fetchall()
    return con.execute("SELECT * FROM auftrag ORDER BY auftrag_nr").fetchall()


# --------------------------------------------------------------------------
# Standort-Events
# --------------------------------------------------------------------------

def event_erfassen(con, gestell_id, station_id, quelle="rfid", epc=None,
                   rssi=None, bemerkung=None, ts=None, entprellen=True):
    """Ein Gestell wurde an einer Station gesehen.

    Gibt die event_id zurueck, oder None wenn das Event als Doppellesung
    verworfen wurde.
    """
    if quelle not in config.QUELLEN:
        raise ValueError(f"Unbekannte Quelle: {quelle}")

    st = con.execute(
        "SELECT * FROM station WHERE station_id = ?", (station_id,)
    ).fetchone()
    if st is None:
        raise ValueError(f"Unbekannte Station: {station_id}")
    if gestell(con, gestell_id) is None:
        raise ValueError(f"Unbekanntes Gestell: {gestell_id}")

    zeitpunkt = as_dt(as_ts(ts))

    if entprellen:
        letztes = con.execute(
            "SELECT station_id, ts FROM event WHERE gestell_id = ? "
            "ORDER BY ts DESC, event_id DESC LIMIT 1",
            (gestell_id,),
        ).fetchone()
        if letztes and letztes["station_id"] == station_id:
            delta = (zeitpunkt - as_dt(letztes["ts"])).total_seconds()
            if 0 <= delta < ENTPRELL_SEKUNDEN:
                return None

    cur = con.execute(
        "INSERT INTO event (ts, gestell_id, station_id, linie_id, quelle, "
        "  epc_gelesen, rssi, bemerkung) VALUES (?,?,?,?,?,?,?,?)",
        (as_ts(zeitpunkt), gestell_id, station_id, st["linie_id"], quelle,
         epc, rssi, bemerkung),
    )
    con.commit()
    return cur.lastrowid


def event_von_epc(con, epc, station_id, rssi=None, ts=None):
    """Rohlesung eines Readers verarbeiten.

    Unbekannte EPCs werden bewusst nicht stillschweigend verschluckt --
    ein gelesener, aber nicht zugeordneter Tag ist ein Hinweis auf ein
    Gestell, das noch nicht in den Stammdaten steht.
    """
    g = gestell_by_epc(con, epc)
    if g is None:
        return None, f"EPC {epc} keinem Gestell zugeordnet"
    eid = event_erfassen(con, g["gestell_id"], station_id, quelle="rfid",
                         epc=epc, rssi=rssi, ts=ts)
    return eid, None


def historie(con, gestell_id, limit=100):
    return con.execute(
        "SELECT e.*, s.name AS station_name FROM event e "
        "JOIN station s ON s.station_id = e.station_id "
        "WHERE e.gestell_id = ? ORDER BY e.ts DESC, e.event_id DESC LIMIT ?",
        (gestell_id, limit),
    ).fetchall()


def letzte_events(con, limit=50):
    return con.execute(
        "SELECT e.*, s.name AS station_name FROM event e "
        "JOIN station s ON s.station_id = e.station_id "
        "ORDER BY e.ts DESC, e.event_id DESC LIMIT ?",
        (limit,),
    ).fetchall()


# --------------------------------------------------------------------------
# Belegung: Auftrag <-> Gestell
# --------------------------------------------------------------------------

def einhaengen(con, gestell_id, auftrag_nr, menge=None, station_id=None,
               quelle="scan", bemerkung=None, ts=None):
    """Auftrag auf ein Gestell haengen.

    Mehrere Auftraege auf einem Gestell sind zulaessig -- derselbe Auftrag
    zweimal offen auf demselben Gestell nicht.
    """
    if gestell(con, gestell_id) is None:
        raise ValueError(f"Unbekanntes Gestell: {gestell_id}")
    if auftrag(con, auftrag_nr) is None:
        raise ValueError(f"Unbekannter Auftrag: {auftrag_nr}")

    doppelt = con.execute(
        "SELECT belegung_id FROM belegung "
        "WHERE gestell_id = ? AND auftrag_nr = ? AND bis IS NULL",
        (gestell_id, auftrag_nr),
    ).fetchone()
    if doppelt:
        raise ValueError(
            f"Auftrag {auftrag_nr} haengt bereits auf Gestell {gestell_id}."
        )

    zeitpunkt = as_ts(ts)
    cur = con.execute(
        "INSERT INTO belegung (gestell_id, auftrag_nr, menge, von, bemerkung) "
        "VALUES (?,?,?,?,?)",
        (gestell_id, auftrag_nr, menge, zeitpunkt, bemerkung),
    )
    con.execute(
        "UPDATE auftrag SET status = 'laufend' WHERE auftrag_nr = ? AND status = 'offen'",
        (auftrag_nr,),
    )
    con.commit()

    if station_id:
        event_erfassen(con, gestell_id, station_id, quelle=quelle,
                       bemerkung=f"Einhaengen {auftrag_nr}", ts=zeitpunkt,
                       entprellen=False)
    return cur.lastrowid


def aushaengen(con, gestell_id, station_id=None, auftrag_nr=None,
               quelle="scan", ts=None, auftrag_fertig=True):
    """Belegung(en) eines Gestells schliessen.

    Ohne auftrag_nr werden alle offenen Belegungen des Gestells geschlossen
    -- das ist der Normalfall am Aushaengeplatz.
    """
    zeitpunkt = as_ts(ts)
    if auftrag_nr:
        offene = con.execute(
            "SELECT * FROM belegung WHERE gestell_id = ? AND auftrag_nr = ? AND bis IS NULL",
            (gestell_id, auftrag_nr),
        ).fetchall()
    else:
        offene = con.execute(
            "SELECT * FROM belegung WHERE gestell_id = ? AND bis IS NULL",
            (gestell_id,),
        ).fetchall()

    for b in offene:
        con.execute("UPDATE belegung SET bis = ? WHERE belegung_id = ?",
                    (zeitpunkt, b["belegung_id"]))
        if auftrag_fertig:
            # Auftrag gilt erst als fertig, wenn er auf keinem Gestell mehr haengt.
            rest = con.execute(
                "SELECT COUNT(*) FROM belegung WHERE auftrag_nr = ? AND bis IS NULL "
                "AND belegung_id <> ?",
                (b["auftrag_nr"], b["belegung_id"]),
            ).fetchone()[0]
            if rest == 0:
                con.execute("UPDATE auftrag SET status = 'fertig' WHERE auftrag_nr = ?",
                            (b["auftrag_nr"],))
    con.commit()

    if station_id and offene:
        namen = ", ".join(b["auftrag_nr"] for b in offene)
        event_erfassen(con, gestell_id, station_id, quelle=quelle,
                       bemerkung=f"Aushaengen {namen}", ts=zeitpunkt,
                       entprellen=False)
    return len(offene)


def offene_belegungen(con, gestell_id=None, auftrag_nr=None):
    sql = ("SELECT b.*, a.kunde, a.artikel, a.farbe, a.liefertermin "
           "FROM belegung b JOIN auftrag a ON a.auftrag_nr = b.auftrag_nr "
           "WHERE b.bis IS NULL")
    args = []
    if gestell_id:
        sql += " AND b.gestell_id = ?"
        args.append(gestell_id)
    if auftrag_nr:
        sql += " AND b.auftrag_nr = ?"
        args.append(auftrag_nr)
    sql += " ORDER BY b.von"
    return con.execute(sql, args).fetchall()
