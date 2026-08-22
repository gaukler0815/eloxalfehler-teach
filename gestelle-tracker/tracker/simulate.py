"""Demo-Betrieb mit simulierten Lesepunkten.

Damit laesst sich das fertige System beurteilen, bevor eine einzige
Antenne bestellt ist. Sobald echte Reader angebunden sind, ruft der
Reader-Dienst statt dieser Datei einfach models.event_von_epc() auf --
alles andere bleibt gleich.
"""

import random
from datetime import timedelta

from . import config, models
from .models import as_dt, jetzt

KUNDEN = [
    "Muster Metallbau GmbH", "Alpha Fenstertechnik", "Berger Maschinenbau",
    "Cordes Fassaden KG", "Delta Profiltechnik", "Eichler Industrie AG",
    "Fabritz Sanitaer", "Grosser Leichtbau", "Hansen Solar", "Iller Apparatebau",
]
ARTIKEL = [
    "Fensterprofil 60mm", "Abdeckblende", "Kuehlkoerper", "Frontplatte",
    "Handlauf 2000mm", "Rahmenprofil eckig", "Lochblech", "Sichtblende",
    "Traegerschiene", "Gehaeusedeckel",
]
FARBEN = config.FARBEN
SCHICHTDICKEN = config.SCHICHTDICKEN

# Prozesskette je Linie ohne Puffer/Lager -- das ist der Weg, den ein
# belegtes Gestell normalerweise nimmt.
def _prozesskette(linie_id):
    return [s["station_id"] for s in config.stationen_nach_linie(linie_id)
            if s["art"] in ("einhaengen", "lesepunkt", "aushaengen")]


def _epc(nummer):
    """Sprechender Platzhalter-EPC. Echte Tags kommen ab Werk vorprogrammiert."""
    return f"E280F3{nummer:06X}"


def gestellpark_anlegen(con, anzahl=400, anteil_getaggt=0.65, seed=42):
    """Stammdaten fuer den Gestellpark.

    anteil_getaggt bildet den rollierenden Rollout ab: mitten im Ausrollen
    hat eben nur ein Teil der Gestelle schon einen Tag. Genau dieser Zustand
    muss im Alltag funktionieren.
    """
    rnd = random.Random(seed)
    typen = config.GESTELLTYPEN
    gewichte = [0.4, 0.3, 0.2, 0.1]
    for i in range(1, anzahl + 1):
        typ = rnd.choices(typen, weights=gewichte, k=1)[0]
        getaggt = rnd.random() < anteil_getaggt
        models.gestell_anlegen(
            con,
            gestell_id=f"G-{i:04d}",
            typ=typ["typ"],
            plaetze=typ["plaetze"],
            epc=_epc(i) if getaggt else None,
        )
    return anzahl


def _auftrag_erzeugen(con, nummer, rnd, ts):
    nr = f"A-{nummer:05d}"
    models.auftrag_anlegen(
        con, nr,
        kunde=rnd.choice(KUNDEN),
        artikel=rnd.choice(ARTIKEL),
        menge=rnd.choice([40, 80, 120, 200, 250, 400, 600]),
        farbe=rnd.choice(FARBEN),
        schichtdicke=rnd.choice(SCHICHTDICKEN),
        liefertermin=(as_dt(models.as_ts(ts)) + timedelta(days=rnd.randint(2, 14)))
            .strftime("%Y-%m-%d"),
        ts=ts,
    )
    return nr


def verlauf_erzeugen(con, tage=7, chargen_pro_tag=70, seed=42):
    """Historie und laufenden Betrieb erzeugen.

    Ein Teil der Chargen ist komplett durchgelaufen (liefert Durchlaufzeiten),
    ein Teil haengt noch in der Anlage (fuellt die Hallenuebersicht), und
    einige stehen absichtlich zu lange im Puffer, damit die Liegezeit-
    Ueberwachung sichtbar anschlaegt.
    """
    rnd = random.Random(seed + 1)
    getaggt = [g["gestell_id"] for g in models.gestelle(con) if g["epc"]]
    if not getaggt:
        raise RuntimeError("Erst gestellpark_anlegen() aufrufen.")

    ketten = {lid: _prozesskette(lid) for lid in (config.AUTOMAT, config.HANDANLAGE)}
    soll = {s["station_id"]: (s["soll_dauer_min"] or 30) for s in config.STATIONEN}

    start = jetzt() - timedelta(days=tage)

    # Startinventur: jedes getaggte Gestell wird einmal im Lager erfasst.
    # Ungetaggte bleiben ohne Standort -- genau diese Luecke schliesst der
    # Rollout Stueck fuer Stueck.
    for gestell_id in getaggt:
        models.event_erfassen(con, gestell_id, "ZZ-LAG", quelle="simulation",
                              ts=start - timedelta(minutes=rnd.randint(1, 600)))

    auftrag_nr = 1
    belegt_bis = {}          # gestell_id -> Zeitpunkt, ab dem es wieder frei ist
    angelegt = 0

    gesamt = tage * chargen_pro_tag
    for n in range(gesamt):
        einhaenge_zeit = start + timedelta(
            minutes=int(n * (tage * 24 * 60) / max(gesamt, 1)) + rnd.randint(0, 25)
        )
        frei = [g for g in getaggt
                if belegt_bis.get(g) is None or belegt_bis[g] <= einhaenge_zeit]
        if not frei:
            continue
        gestell_id = rnd.choice(frei)
        linie_id = config.AUTOMAT if rnd.random() < 0.7 else config.HANDANLAGE
        kette = ketten[linie_id]

        nr = _auftrag_erzeugen(con, auftrag_nr, rnd, einhaenge_zeit)
        auftrag_nr += 1
        angelegt += 1

        menge = models.auftrag(con, nr)["menge"]
        models.einhaengen(con, gestell_id, nr, menge=menge,
                          station_id=kette[0], ts=einhaenge_zeit)

        # Wie weit ist diese Charge gekommen? Aeltere Chargen sind durch,
        # die juengsten stecken noch mittendrin.
        t = einhaenge_zeit
        haenger = rnd.random() < 0.08   # bleibt irgendwo liegen
        bis_schritt = len(kette)
        if (jetzt() - einhaenge_zeit) < timedelta(hours=6):
            bis_schritt = rnd.randint(1, len(kette))

        for station_id in kette[1:bis_schritt]:
            dauer = soll[station_id] * rnd.uniform(0.7, 1.3)
            if haenger and rnd.random() < 0.3:
                dauer *= rnd.uniform(3.0, 5.0)
            t += timedelta(minutes=int(dauer))
            if t > jetzt():
                break
            models.event_erfassen(con, gestell_id, station_id,
                                  quelle="simulation", ts=t)
        else:
            if bis_schritt == len(kette):
                models.aushaengen(con, gestell_id, station_id=kette[-1],
                                  quelle="simulation", ts=t)
                # Gestell danach ins Lager oder in den Puffer
                zurueck = t + timedelta(minutes=rnd.randint(10, 90))
                if zurueck <= jetzt():
                    models.event_erfassen(
                        con, gestell_id,
                        "ZZ-LAG" if rnd.random() < 0.6 else "ZZ-VER",
                        quelle="simulation", ts=zurueck)
                belegt_bis[gestell_id] = zurueck
                continue
        belegt_bis[gestell_id] = t + timedelta(hours=2)

    return angelegt


def demo_aufbauen(con, anzahl_gestelle=400, tage=7, chargen_pro_tag=70, seed=42):
    """Kompletter Demo-Datenbestand von null."""
    from . import db
    db.leeren(con)
    gestellpark_anlegen(con, anzahl=anzahl_gestelle, seed=seed)
    auftraege = verlauf_erzeugen(con, tage=tage, chargen_pro_tag=chargen_pro_tag,
                                 seed=seed)
    return {"gestelle": anzahl_gestelle, "auftraege": auftraege}


def takt(con, anzahl=5, seed=None):
    """Einen Lesezyklus simulieren: ein paar Gestelle ruecken weiter.

    Das ist der Platzhalter fuer den spaeteren Reader-Dienst.
    """
    from . import auswertung
    rnd = random.Random(seed)
    ketten = {lid: _prozesskette(lid) for lid in (config.AUTOMAT, config.HANDANLAGE)}

    kandidaten = [z for z in auswertung.hallenuebersicht(con, nur_belegte=True)
                  if z["station_id"]]
    rnd.shuffle(kandidaten)
    bewegt = []
    for z in kandidaten[:anzahl]:
        kette = None
        for lid, k in ketten.items():
            if z["station_id"] in k:
                kette = k
                break
        if kette is None:
            continue
        pos = kette.index(z["station_id"])
        if pos + 1 >= len(kette):
            models.aushaengen(con, z["gestell_id"], station_id=kette[-1],
                              quelle="simulation")
            models.event_erfassen(con, z["gestell_id"], "ZZ-LAG",
                                  quelle="simulation")
            bewegt.append((z["gestell_id"], "ausgehaengt"))
        else:
            ziel = kette[pos + 1]
            models.event_erfassen(con, z["gestell_id"], ziel, quelle="simulation")
            bewegt.append((z["gestell_id"], config.station(ziel)["name"]))
    return bewegt
