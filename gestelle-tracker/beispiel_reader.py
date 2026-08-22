"""Beispiel: so haengt spaeter ein echter RFID-Reader an der Anwendung.

Dieses Skript ist bewusst kurz -- es zeigt nur die Nahtstelle. Egal ob der
Reader per LLRP, MQTT, HTTP-Callback oder serieller Schnittstelle liefert:
am Ende kommt ein EPC, ein Lesepunkt und ein Zeitstempel heraus, und daraus
wird ein Event. Alles andere -- Entprellung, Zuordnung, Auswertung -- passiert
in tracker/models.py und aendert sich nicht.

Start:  python beispiel_reader.py AU-VB
"""

import sys
import time

from tracker import config, db, models


def lesungen_vom_reader(station_id):
    """Platzhalter fuer die echte Readeranbindung.

    Hier wuerde je nach Geraet stehen:
      - sllurp / LLRP-Client fuer Impinj- und Zebra-Reader
      - ein MQTT-Abonnement, wenn der Reader selbst publiziert
      - ein kleiner HTTP-Endpunkt, den der Reader per Callback anspricht

    Erwartet wird je Lesung: (epc, rssi).
    """
    raise NotImplementedError(
        "Hier die Anbindung des konkreten Readers eintragen.\n"
        "Zum Ausprobieren ohne Hardware: Seite 'Simulator' in der Anwendung."
    )


def hauptschleife(station_id):
    if config.station(station_id) is None:
        raise SystemExit(f"Unbekannter Lesepunkt: {station_id}")

    con = db.connect()
    print(f"Lesepunkt {station_id} ({config.station(station_id)['name']}) aktiv.")

    for epc, rssi in lesungen_vom_reader(station_id):
        eid, fehler = models.event_von_epc(con, epc, station_id, rssi=rssi)
        if fehler:
            # Ein gelesener, aber unbekannter Tag ist ein Hinweis auf ein
            # Gestell, das noch nicht in den Stammdaten steht -- nicht
            # einfach wegwerfen.
            print("HINWEIS:", fehler)
        elif eid is None:
            pass  # Doppellesung, still verworfen
        else:
            g = models.gestell_by_epc(con, epc)
            print(f"{time.strftime('%H:%M:%S')}  {g['gestell_id']} erfasst")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Aufruf: python beispiel_reader.py <STATION-ID>\n"
                         "Bekannte Lesepunkte: "
                         + ", ".join(s["station_id"] for s in config.STATIONEN))
    hauptschleife(sys.argv[1])
