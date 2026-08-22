"""Gestell- und Auftragsverfolgung – Startseite.

Start:  streamlit run app.py
"""

import streamlit as st

from tracker import auswertung, models, ui

ui.kopf("🏭 Gestell- und Auftragsverfolgung",
        "Wo steckt welcher Auftrag – Automat und Handanlage")

con = ui.verbindung()

if ui.leerer_bestand_hinweis(con):
    st.stop()

ui.kennzahlen_zeile(con)
st.divider()

links, rechts = st.columns([3, 2])

with links:
    st.subheader("Belegung der Anlagen")
    st.caption("Wie viele Gestelle stehen gerade an welcher Station.")
    zeilen = []
    for e in auswertung.auslastung_je_station(con):
        zeilen.append({
            "Linie": e["linie"],
            "Station": e["station"],
            "Gestelle": e["anzahl"],
            "davon belegt": e["belegt"],
            "🟡": e["warnungen"],
            "🔴": e["alarme"],
        })
    ui.tabelle(zeilen)

with rechts:
    st.subheader("Braucht Aufmerksamkeit")
    st.caption("Belegte Gestelle, die deutlich laenger stehen als vorgesehen.")
    kritisch = [z for z in auswertung.hallenuebersicht(con, nur_belegte=True)
                if z["status"] in ("alarm", "warnung")]
    kritisch.sort(key=lambda z: -(z["liegezeit_min"] or 0))
    if not kritisch:
        st.success("Nichts Auffaelliges – alle belegten Gestelle sind im Takt.")
    else:
        for z in kritisch[:12]:
            symbol = "🔴" if z["status"] == "alarm" else "🟡"
            auftraege = ", ".join(z["auftraege"]) or "ohne Auftrag"
            st.write(
                f"{symbol} **{z['gestell_id']}** – {z['station']} · "
                f"{z['liegezeit_min']} min (Soll {z['soll_dauer_min']}) · {auftraege}"
            )
        if len(kritisch) > 12:
            st.caption(f"… und {len(kritisch) - 12} weitere")

st.divider()
ui.rollout_balken(con)
st.caption(
    "Gestelle ohne Tag lassen sich weiterhin von Hand erfassen – sie tauchen "
    "in der Uebersicht nur ohne automatischen Standort auf."
)

st.divider()
st.subheader("Letzte Lesungen")
letzte = []
for e in models.letzte_events(con, limit=25):
    letzte.append({
        "Zeit": e["ts"].replace("T", " "),
        "Gestell": e["gestell_id"],
        "Station": e["station_name"],
        "Quelle": e["quelle"],
        "Bemerkung": e["bemerkung"] or "",
    })
ui.tabelle(letzte)
