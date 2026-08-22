"""Anlagenkonfiguration.

Hier stehen die Linien und die Stationen (= Lesepunkte), die spaeter durch
echte RFID-Antennen abgebildet werden. Beim Umstellen auf die reale Anlage
wird nur diese Datei angepasst -- die Logik in models.py bleibt unveraendert.

Wichtig: 'reihenfolge' bildet die Prozesskette ab. Weil ein Gestell auf der
Schiene eine feste Reihenfolge abfaehrt, laesst sich zwischen zwei
Lesepunkten interpoliert werden, wo es gerade steckt. Deshalb braucht nicht
jedes Bad einen eigenen Lesepunkt.
"""

AUTOMAT = "AU"
HANDANLAGE = "HA"
UEBERGREIFEND = "ZZ"

LINIEN = [
    {"linie_id": AUTOMAT, "name": "Automat", "art": "automat"},
    {"linie_id": HANDANLAGE, "name": "Handanlage", "art": "hand"},
    {"linie_id": UEBERGREIFEND, "name": "Uebergreifend", "art": "sonstige"},
]

# art:
#   einhaengen  -> hier wird die Belegung Auftrag <-> Gestell geoeffnet
#   aushaengen  -> hier wird die Belegung geschlossen
#   lesepunkt   -> reiner Durchfahrt-Lesepunkt in der Prozesskette
#   puffer      -> Wartebereich, Liegezeit hier ist das teure Problem
#   lager       -> Gestelllager, hier stehen freie Gestelle
#
# soll_dauer_min: erwartete Verweildauer. Wird sie deutlich ueberschritten,
# schlaegt die Hallenuebersicht Alarm. Werte sind Startwerte und muessen
# im Betrieb nachjustiert werden.
STATIONEN = [
    # --- Automat -----------------------------------------------------------
    {"station_id": "AU-PUF", "linie_id": AUTOMAT, "name": "Puffer vor Automat",
     "reihenfolge": 10, "art": "puffer", "soll_dauer_min": 60},
    {"station_id": "AU-EIN", "linie_id": AUTOMAT, "name": "Einhaengeplatz Automat",
     "reihenfolge": 20, "art": "einhaengen", "soll_dauer_min": 30},
    {"station_id": "AU-VB", "linie_id": AUTOMAT, "name": "Vorbehandlung Automat",
     "reihenfolge": 30, "art": "lesepunkt", "soll_dauer_min": 35},
    {"station_id": "AU-ELX", "linie_id": AUTOMAT, "name": "Eloxal Automat",
     "reihenfolge": 40, "art": "lesepunkt", "soll_dauer_min": 60},
    {"station_id": "AU-FRB", "linie_id": AUTOMAT, "name": "Faerberei Automat",
     "reihenfolge": 50, "art": "lesepunkt", "soll_dauer_min": 25},
    {"station_id": "AU-VDT", "linie_id": AUTOMAT, "name": "Verdichtung Automat",
     "reihenfolge": 60, "art": "lesepunkt", "soll_dauer_min": 45},
    {"station_id": "AU-AUS", "linie_id": AUTOMAT, "name": "Aushaengeplatz Automat",
     "reihenfolge": 70, "art": "aushaengen", "soll_dauer_min": 30},

    # --- Handanlage --------------------------------------------------------
    {"station_id": "HA-EIN", "linie_id": HANDANLAGE, "name": "Einhaengeplatz Hand",
     "reihenfolge": 20, "art": "einhaengen", "soll_dauer_min": 40},
    {"station_id": "HA-VB", "linie_id": HANDANLAGE, "name": "Vorbehandlung Hand",
     "reihenfolge": 30, "art": "lesepunkt", "soll_dauer_min": 40},
    {"station_id": "HA-ELX", "linie_id": HANDANLAGE, "name": "Eloxal Hand",
     "reihenfolge": 40, "art": "lesepunkt", "soll_dauer_min": 70},
    {"station_id": "HA-VDT", "linie_id": HANDANLAGE, "name": "Faerben/Verdichten Hand",
     "reihenfolge": 50, "art": "lesepunkt", "soll_dauer_min": 60},
    {"station_id": "HA-AUS", "linie_id": HANDANLAGE, "name": "Aushaengeplatz Hand",
     "reihenfolge": 70, "art": "aushaengen", "soll_dauer_min": 40},

    # --- uebergreifend -----------------------------------------------------
    {"station_id": "ZZ-LAG", "linie_id": UEBERGREIFEND, "name": "Gestelllager",
     "reihenfolge": 90, "art": "lager", "soll_dauer_min": None},
    {"station_id": "ZZ-VER", "linie_id": UEBERGREIFEND, "name": "Verpackung / Versand",
     "reihenfolge": 95, "art": "puffer", "soll_dauer_min": 240},
]

# Quellen eines Standort-Events. 'rfid' und 'scan' kommen spaeter aus der
# Hardware, 'manuell' ist der Notnagel wenn ein Tag abgerissen ist,
# 'simulation' erzeugt der eingebaute Demo-Betrieb.
QUELLEN = ("rfid", "scan", "manuell", "simulation")

# Gestelltypen wie sie im Umlauf sind. Nur Stammdaten, keine Logik.
GESTELLTYPEN = [
    {"typ": "T1", "bezeichnung": "Standard 3000x1500", "plaetze": 24},
    {"typ": "T2", "bezeichnung": "Standard 2000x1200", "plaetze": 16},
    {"typ": "T3", "bezeichnung": "Profilgestell lang", "plaetze": 40},
    {"typ": "T4", "bezeichnung": "Kleinteile-Korbgestell", "plaetze": 8},
]

# Auswahllisten fuer die Erfassungsmasken.
FARBEN = ["natur E6/EV1", "C-33 mittelbronze", "C-34 dunkelbronze",
          "C-35 schwarz", "gold", "blau eingefaerbt"]
SCHICHTDICKEN = ["10 my", "15 my", "20 my", "25 my"]

# Toleranz auf die Soll-Dauer, bevor die Uebersicht warnt bzw. Alarm gibt.
WARNUNG_AB_FAKTOR = 1.5
ALARM_AB_FAKTOR = 2.5


def stationen_nach_linie(linie_id):
    """Stationen einer Linie in Prozessreihenfolge."""
    return sorted(
        (s for s in STATIONEN if s["linie_id"] == linie_id),
        key=lambda s: s["reihenfolge"],
    )


def station(station_id):
    for s in STATIONEN:
        if s["station_id"] == station_id:
            return s
    return None


def linie(linie_id):
    for l in LINIEN:
        if l["linie_id"] == linie_id:
            return l
    return None
