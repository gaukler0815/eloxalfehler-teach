"""Gemeinsame Helfer fuer die Streamlit-Oberflaeche."""

import pandas as pd
import streamlit as st

from . import auswertung, db

STATUS_SYMBOL = {
    "ok": "🟢 laeuft",
    "warnung": "🟡 ueberfaellig",
    "alarm": "🔴 steht zu lange",
    "frei": "⚪ frei",
    "unbekannt": "❔ kein Standort",
    "ohne_sollwert": "🟢 laeuft",
}


@st.cache_resource
def _verbindung(pfad):
    return db.connect(pfad)


def verbindung():
    """Eine Verbindung je Datenbankdatei, ueber die Session hinweg gehalten.

    Der Pfad ist Teil des Cache-Schluessels -- sonst haengt die Anwendung
    nach einem Wechsel der Datenbank an der alten Verbindung fest.
    """
    return _verbindung(db.db_pfad())


def kopf(titel, untertitel=None):
    st.set_page_config(page_title=f"{titel} – Gestellverfolgung",
                       page_icon="🏭", layout="wide")
    st.title(titel)
    if untertitel:
        st.caption(untertitel)


def leerer_bestand_hinweis(con):
    """Zeigt einen Hinweis, wenn noch keine Daten da sind."""
    anzahl = con.execute("SELECT COUNT(*) FROM gestell").fetchone()[0]
    if anzahl == 0:
        st.info(
            "Noch keine Daten. Auf der Seite **Simulator** laesst sich mit einem "
            "Klick ein Demo-Bestand mit 400 Gestellen erzeugen."
        )
        return True
    return False


def kennzahlen_zeile(con):
    k = auswertung.kennzahlen(con)
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Gestelle im Umlauf", k["gestelle_gesamt"])
    s2.metric("davon belegt", k["gestelle_belegt"])
    s3.metric("Auftraege laufend", k["auftraege_laufend"])
    s4.metric("Ueberfaellig", k["warnungen"],
              delta=None if not k["warnungen"] else "pruefen",
              delta_color="off")
    s5.metric("Alarme", k["alarme"],
              delta=None if not k["alarme"] else "sofort pruefen",
              delta_color="inverse")
    return k


def rollout_balken(con):
    r = auswertung.rollout_fortschritt(con)
    st.progress(r["anteil"] if r["gesamt"] else 0.0,
                text=f"RFID-Rollout: {r['getaggt']} von {r['gesamt']} Gestellen "
                     f"getaggt ({r['anteil']:.0%}) – {r['offen']} offen")
    return r


def tabelle(zeilen, spalten=None, hoehe=None):
    """Liste von dicts als Tabelle. Leere Liste wird sauber abgefangen."""
    if not zeilen:
        st.caption("– keine Eintraege –")
        return None
    rahmen = pd.DataFrame(zeilen)
    if spalten:
        vorhanden = [s for s in spalten if s in rahmen.columns]
        rahmen = rahmen[vorhanden]
    # height darf nicht None sein, deshalb nur setzen wenn angegeben.
    zusatz = {"height": hoehe} if hoehe else {}
    st.dataframe(rahmen, width="stretch", hide_index=True, **zusatz)
    return rahmen


def uebersicht_aufbereiten(zeilen):
    """Hallenuebersicht fuer die Anzeige lesbar machen."""
    aufbereitet = []
    for z in zeilen:
        aufbereitet.append({
            "Gestell": z["gestell_id"],
            "Typ": z["typ"] or "–",
            "Tag": "✓" if z["getaggt"] else "–",
            "Linie": z["linie"] or "–",
            "Station": z["station"] or "–",
            "seit": (z["seit"] or "–").replace("T", " ") if z["seit"] else "–",
            "Liegezeit (min)": z["liegezeit_min"] if z["liegezeit_min"] is not None else "–",
            "Soll (min)": z["soll_dauer_min"] or "–",
            "Zustand": STATUS_SYMBOL.get(z["status"], z["status"]),
            "Auftraege": ", ".join(z["auftraege"]) or "–",
            "Kunde": ", ".join(z["kunden"]) or "–",
        })
    return aufbereitet
