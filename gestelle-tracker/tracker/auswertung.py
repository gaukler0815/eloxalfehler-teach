"""Auswertungen: aktueller Standort, Liegezeiten, Gestellverfuegbarkeit.

Der aktuelle Standort ist nirgends gespeichert -- er wird immer aus dem
juengsten Event berechnet. Dadurch kann der Zustand nicht mit der Historie
auseinanderlaufen.
"""

from . import config
from .models import as_dt, jetzt, offene_belegungen

# Das juengste Event je Gestell. Basis fuer fast alles hier.
_LETZTE_EVENTS = """
WITH nummeriert AS (
    SELECT e.*,
           ROW_NUMBER() OVER (PARTITION BY e.gestell_id
                              ORDER BY e.ts DESC, e.event_id DESC) AS rn
    FROM event e
)
SELECT n.gestell_id, n.ts, n.station_id, n.linie_id, n.quelle,
       s.name AS station_name, s.art AS station_art, s.soll_dauer_min,
       l.name AS linie_name
FROM nummeriert n
JOIN station s ON s.station_id = n.station_id
JOIN linie   l ON l.linie_id   = n.linie_id
WHERE n.rn = 1
"""


def liegezeit_status(minuten, soll_dauer_min):
    """ok | warnung | alarm | ohne_sollwert"""
    if soll_dauer_min is None:
        return "ohne_sollwert"
    if minuten >= soll_dauer_min * config.ALARM_AB_FAKTOR:
        return "alarm"
    if minuten >= soll_dauer_min * config.WARNUNG_AB_FAKTOR:
        return "warnung"
    return "ok"


def aktueller_standort(con, gestell_id):
    zeile = con.execute(
        _LETZTE_EVENTS + " AND n.gestell_id = ?", (gestell_id,)
    ).fetchone()
    return dict(zeile) if zeile else None


def _standort_index(con):
    return {r["gestell_id"]: dict(r) for r in con.execute(_LETZTE_EVENTS)}


def hallenuebersicht(con, stand=None, nur_belegte=False):
    """Eine Zeile je aktivem Gestell: wo steht es, wie lange schon, was haengt drauf."""
    stand = stand or jetzt()
    standorte = _standort_index(con)

    belegung_je_gestell = {}
    for b in offene_belegungen(con):
        belegung_je_gestell.setdefault(b["gestell_id"], []).append(dict(b))

    zeilen = []
    for g in con.execute(
        "SELECT * FROM gestell WHERE status = 'aktiv' ORDER BY gestell_id"
    ):
        gid = g["gestell_id"]
        belegungen = belegung_je_gestell.get(gid, [])
        if nur_belegte and not belegungen:
            continue

        ort = standorte.get(gid)
        if ort is None:
            minuten, status = None, "unbekannt"
        else:
            minuten = int((stand - as_dt(ort["ts"])).total_seconds() // 60)
            # Liegezeit wird nur bewertet, wenn ein Auftrag drauf haengt.
            # Ein freies Gestell im Lager soll keinen Alarm ausloesen.
            status = (liegezeit_status(minuten, ort["soll_dauer_min"])
                      if belegungen else "frei")

        zeilen.append({
            "gestell_id": gid,
            "typ": g["typ"],
            "getaggt": g["epc"] is not None,
            "linie": ort["linie_name"] if ort else None,
            "station": ort["station_name"] if ort else None,
            "station_id": ort["station_id"] if ort else None,
            "station_art": ort["station_art"] if ort else None,
            "seit": ort["ts"] if ort else None,
            "liegezeit_min": minuten,
            "soll_dauer_min": ort["soll_dauer_min"] if ort else None,
            "status": status,
            "belegt": bool(belegungen),
            "auftraege": [b["auftrag_nr"] for b in belegungen],
            "kunden": sorted({b["kunde"] for b in belegungen if b["kunde"]}),
        })
    return zeilen


def auftrag_verfolgen(con, auftrag_nr):
    """Wo steckt ein Auftrag gerade -- ueber alle Gestelle, die ihn tragen."""
    treffer = []
    for b in offene_belegungen(con, auftrag_nr=auftrag_nr):
        ort = aktueller_standort(con, b["gestell_id"])
        treffer.append({
            "gestell_id": b["gestell_id"],
            "menge": b["menge"],
            "seit_einhaengen": b["von"],
            "linie": ort["linie_name"] if ort else None,
            "station": ort["station_name"] if ort else None,
            "seit": ort["ts"] if ort else None,
        })
    return treffer


def durchlaufzeit_min(con, auftrag_nr):
    """Minuten vom ersten Einhaengen bis zum letzten Aushaengen.

    Gibt None zurueck, solange der Auftrag noch laeuft.
    """
    zeile = con.execute(
        "SELECT MIN(von) AS start, MAX(bis) AS ende, "
        "       SUM(CASE WHEN bis IS NULL THEN 1 ELSE 0 END) AS offen "
        "FROM belegung WHERE auftrag_nr = ?",
        (auftrag_nr,),
    ).fetchone()
    if zeile is None or zeile["start"] is None or zeile["offen"]:
        return None
    return int((as_dt(zeile["ende"]) - as_dt(zeile["start"])).total_seconds() // 60)


def freie_gestelle(con):
    """Gestelle ohne offene Belegung, mit letztem bekannten Standort.

    Bei 400 Gestellen ist das oft genauso wertvoll wie die Auftragsverfolgung:
    freie Gestelle sind da, man findet sie nur nicht.
    """
    standorte = _standort_index(con)
    belegt = {b["gestell_id"] for b in offene_belegungen(con)}
    frei = []
    for g in con.execute(
        "SELECT * FROM gestell WHERE status = 'aktiv' ORDER BY gestell_id"
    ):
        gid = g["gestell_id"]
        if gid in belegt:
            continue
        ort = standorte.get(gid)
        frei.append({
            "gestell_id": gid,
            "typ": g["typ"],
            "plaetze": g["plaetze"],
            "getaggt": g["epc"] is not None,
            "station": ort["station_name"] if ort else None,
            "linie": ort["linie_name"] if ort else None,
            "zuletzt_gesehen": ort["ts"] if ort else None,
        })
    return frei


def rollout_fortschritt(con):
    """Wie viele der Gestelle haben schon einen RFID-Tag."""
    zeile = con.execute(
        "SELECT COUNT(*) AS gesamt, "
        "       SUM(CASE WHEN epc IS NOT NULL THEN 1 ELSE 0 END) AS getaggt "
        "FROM gestell WHERE status = 'aktiv'"
    ).fetchone()
    gesamt = zeile["gesamt"] or 0
    getaggt = zeile["getaggt"] or 0
    return {
        "gesamt": gesamt,
        "getaggt": getaggt,
        "offen": gesamt - getaggt,
        "anteil": (getaggt / gesamt) if gesamt else 0.0,
    }


def kennzahlen(con, stand=None):
    zeilen = hallenuebersicht(con, stand=stand)
    return {
        "gestelle_gesamt": len(zeilen),
        "gestelle_belegt": sum(1 for z in zeilen if z["belegt"]),
        "gestelle_frei": sum(1 for z in zeilen if not z["belegt"]),
        "ohne_standort": sum(1 for z in zeilen if z["status"] == "unbekannt"),
        "warnungen": sum(1 for z in zeilen if z["status"] == "warnung"),
        "alarme": sum(1 for z in zeilen if z["status"] == "alarm"),
        "auftraege_laufend": con.execute(
            "SELECT COUNT(*) FROM auftrag WHERE status = 'laufend'"
        ).fetchone()[0],
        "rollout": rollout_fortschritt(con),
    }


def auslastung_je_station(con, stand=None):
    """Wie viele Gestelle stehen gerade an welcher Station."""
    zeilen = hallenuebersicht(con, stand=stand)
    je_station = {}
    for z in zeilen:
        if not z["station_id"]:
            continue
        eintrag = je_station.setdefault(z["station_id"], {
            "station_id": z["station_id"],
            "station": z["station"],
            "linie": z["linie"],
            "anzahl": 0, "belegt": 0, "warnungen": 0, "alarme": 0,
        })
        eintrag["anzahl"] += 1
        eintrag["belegt"] += 1 if z["belegt"] else 0
        eintrag["warnungen"] += 1 if z["status"] == "warnung" else 0
        eintrag["alarme"] += 1 if z["status"] == "alarm" else 0

    reihenfolge = {s["station_id"]: (s["linie_id"], s["reihenfolge"])
                   for s in config.STATIONEN}
    return sorted(je_station.values(),
                  key=lambda e: reihenfolge.get(e["station_id"], ("ZZ", 999)))
