"""Live-Uebersicht: wo steht gerade welches Gestell, seit wann, mit welchem Auftrag."""

import streamlit as st

from tracker import auswertung, config, ui

ui.kopf("📍 Hallenuebersicht", "Stand jetzt – jede Zeile ein Gestell.")

con = ui.verbindung()
if ui.leerer_bestand_hinweis(con):
    st.stop()

ui.kennzahlen_zeile(con)
st.divider()

f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
linien = ["alle"] + [l["name"] for l in config.LINIEN]
linie_filter = f1.selectbox("Linie", linien)
zustand_filter = f2.multiselect(
    "Zustand", ["ok", "warnung", "alarm", "frei", "unbekannt"],
    default=["ok", "warnung", "alarm"])
suche = f3.text_input("Suche", placeholder="Gestell, Auftrag oder Kunde")
nur_belegt = f4.toggle("nur belegte", value=True)

zeilen = auswertung.hallenuebersicht(con, nur_belegte=nur_belegt)

if linie_filter != "alle":
    zeilen = [z for z in zeilen if z["linie"] == linie_filter]
if zustand_filter:
    zeilen = [z for z in zeilen if z["status"] in zustand_filter
              or (z["status"] == "ohne_sollwert" and "ok" in zustand_filter)]
if suche.strip():
    begriff = suche.strip().lower()
    zeilen = [z for z in zeilen if begriff in z["gestell_id"].lower()
              or any(begriff in a.lower() for a in z["auftraege"])
              or any(begriff in k.lower() for k in z["kunden"])]

# Auffaelliges nach oben.
rang = {"alarm": 0, "warnung": 1, "ok": 2, "ohne_sollwert": 2,
        "unbekannt": 3, "frei": 4}
zeilen.sort(key=lambda z: (rang.get(z["status"], 9), -(z["liegezeit_min"] or 0)))

st.caption(f"{len(zeilen)} Gestell(e)")
ui.tabelle(ui.uebersicht_aufbereiten(zeilen), hoehe=520)

st.divider()
st.subheader("Verteilung ueber die Prozesskette")
st.caption("Wo staut es sich gerade.")
verteilung = []
for e in auswertung.auslastung_je_station(con):
    if linie_filter != "alle" and e["linie"] != linie_filter:
        continue
    verteilung.append({
        "Station": f"{e['linie']} · {e['station']}",
        "Gestelle": e["anzahl"],
        "belegt": e["belegt"],
    })
if verteilung:
    import pandas as pd
    rahmen = pd.DataFrame(verteilung).set_index("Station")
    st.bar_chart(rahmen, height=320)
else:
    st.caption("– keine Daten –")
