"""Simulierte Lesepunkte.

Diese Seite ersetzt die Hardware, solange keine Reader haengen. Sobald
echte Reader da sind, faellt sie ersatzlos weg -- der Reader-Dienst ruft
dann models.event_von_epc() auf, sonst aendert sich nichts.
"""

import streamlit as st

from tracker import auswertung, config, models, simulate, ui

ui.kopf("🧪 Simulator", "Demo-Betrieb, solange noch keine Antenne haengt.")

con = ui.verbindung()

st.warning(
    "Diese Seite gibt es nur fuer die Demo. Im Echtbetrieb liefert der "
    "Reader-Dienst die Lesungen – die Fachlogik dahinter ist dieselbe.",
    icon="⚠️")

aufbau_tab, takt_tab, roh_tab = st.tabs(
    ["Demo-Bestand erzeugen", "Anlage weiterlaufen lassen", "Einzelne Rohlesung"])


with aufbau_tab:
    st.caption("Erzeugt Gestellpark, Auftraege und eine Betriebshistorie. "
               "Der vorhandene Bestand wird dabei geloescht.")
    s1, s2, s3, s4 = st.columns(4)
    anzahl = s1.number_input("Gestelle", min_value=10, max_value=2000,
                             value=400, step=10)
    tage = s2.number_input("Historie (Tage)", min_value=1, max_value=30, value=7)
    chargen = s3.number_input("Chargen pro Tag", min_value=1, max_value=300,
                              value=70, step=5)
    seed = s4.number_input("Zufallsbasis", min_value=0, value=42,
                           help="Gleiche Zahl = gleicher Datenbestand.")

    if st.button("Demo-Bestand erzeugen", type="primary"):
        with st.spinner("Erzeuge Daten …"):
            ergebnis = simulate.demo_aufbauen(
                con, anzahl_gestelle=int(anzahl), tage=int(tage),
                chargen_pro_tag=int(chargen), seed=int(seed))
        st.success(f"{ergebnis['gestelle']} Gestelle und "
                   f"{ergebnis['auftraege']} Auftraege erzeugt.")
        st.rerun()

    if con.execute("SELECT COUNT(*) FROM gestell").fetchone()[0]:
        ui.kennzahlen_zeile(con)


with takt_tab:
    if ui.leerer_bestand_hinweis(con):
        st.stop()
    st.caption("Ruecken einige belegte Gestelle eine Station weiter – so, wie "
               "es die Lesepunkte spaeter melden wuerden.")
    anzahl_takt = st.slider("Wie viele Gestelle bewegen", 1, 20, 5)
    if st.button("Takt ausloesen", type="primary"):
        bewegt = simulate.takt(con, anzahl=int(anzahl_takt))
        if not bewegt:
            st.info("Kein belegtes Gestell zum Bewegen da.")
        else:
            for gid, ziel in bewegt:
                st.write(f"➡️ **{gid}** → {ziel}")
    st.divider()
    ui.tabelle([{
        "Zeit": e["ts"].replace("T", " "),
        "Gestell": e["gestell_id"],
        "Station": e["station_name"],
        "Quelle": e["quelle"],
    } for e in models.letzte_events(con, limit=20)])


with roh_tab:
    if ui.leerer_bestand_hinweis(con):
        st.stop()
    st.caption("So kommt eine Lesung vom Reader an: ein EPC, eine Antenne, "
               "ein Zeitstempel. Unbekannte EPCs werden gemeldet, nicht verschluckt.")
    with st.form("rohlesung"):
        s1, s2, s3 = st.columns(3)
        epc = s1.text_input("EPC", placeholder="E280F3000042")
        station_id = s2.selectbox(
            "Lesepunkt", [s["station_id"] for s in config.STATIONEN],
            format_func=lambda sid: config.station(sid)["name"])
        rssi = s3.number_input("RSSI (dBm)", min_value=-90, max_value=-20, value=-55)
        senden = st.form_submit_button("Lesung senden", type="primary")

    if senden:
        eid, fehler = models.event_von_epc(con, epc.strip(), station_id, rssi=rssi)
        if fehler:
            st.error(fehler)
        elif eid is None:
            st.info("Als Doppellesung verworfen – das Gestell wurde eben schon "
                    "an diesem Punkt erfasst.")
        else:
            g = models.gestell_by_epc(con, epc.strip())
            ort = auswertung.aktueller_standort(con, g["gestell_id"])
            st.success(f"Gestell **{g['gestell_id']}** steht jetzt an "
                       f"**{ort['station_name']}**.")

    st.divider()
    st.write("**Verfuegbare EPCs (Auszug):**")
    ui.tabelle([{"Gestell": g["gestell_id"], "EPC": g["epc"]}
                for g in models.gestelle(con)[:15] if g["epc"]])
