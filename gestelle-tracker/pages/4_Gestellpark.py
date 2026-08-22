"""Gestellpark: Rollout der Tags, freie Gestelle, Historie einzelner Gestelle."""

import streamlit as st

from tracker import auswertung, models, ui

ui.kopf("🧰 Gestellpark", "400 Gestelle im Umlauf – Ausruestung, Verfuegbarkeit, Historie.")

con = ui.verbindung()
if ui.leerer_bestand_hinweis(con):
    st.stop()

rollout_tab, frei_tab, historie_tab = st.tabs(
    ["RFID-Rollout", "Freie Gestelle", "Gestell-Historie"])


with rollout_tab:
    r = ui.rollout_balken(con)
    st.caption(
        "Die Gestelle werden rollierend ausgeruestet – am besten beim "
        "Aushaengen, wenn das Gestell ohnehin leer dasteht. Bis dahin "
        "funktioniert alles auch ohne Tag, nur eben ohne automatischen Standort."
    )

    st.subheader("Tag zuordnen")
    st.caption("Gestell leer? Tag anbringen, hier zuordnen, fertig.")
    with st.form("taggen", clear_on_submit=True):
        s1, s2 = st.columns(2)
        offene = [g["gestell_id"] for g in models.gestelle_ohne_tag(con)]
        if offene:
            gestell_id = s1.selectbox(f"Gestell ohne Tag ({len(offene)} offen)", offene)
        else:
            gestell_id = None
            s1.success("Alle Gestelle sind ausgeruestet.")
        epc = s2.text_input("Tag scannen (EPC)", placeholder="E280F3000042")
        zuordnen = st.form_submit_button("Zuordnen", type="primary",
                                         disabled=gestell_id is None)

    if zuordnen and gestell_id:
        if not epc.strip():
            st.error("Kein Tag gescannt.")
        else:
            try:
                models.gestell_taggen(con, gestell_id, epc.strip())
            except ValueError as e:
                st.error(str(e))
            else:
                st.success(f"Gestell {gestell_id} traegt jetzt Tag {epc.strip()}.")
                st.rerun()

    st.divider()
    st.subheader("Noch nicht ausgeruestet")
    ui.tabelle([{
        "Gestell": g["gestell_id"],
        "Typ": g["typ"] or "–",
        "Plaetze": g["plaetze"] or "–",
    } for g in models.gestelle_ohne_tag(con)], hoehe=320)


with frei_tab:
    frei = auswertung.freie_gestelle(con)
    s1, s2 = st.columns(2)
    s1.metric("Freie Gestelle", len(frei))
    s2.metric("Davon mit bekanntem Standort",
              sum(1 for f in frei if f["station"]))
    st.caption(
        "Bei 400 Gestellen ist das oft genauso wertvoll wie die "
        "Auftragsverfolgung: freie Gestelle sind da, man findet sie nur nicht."
    )

    typen = sorted({f["typ"] for f in frei if f["typ"]})
    gewaehlt = st.multiselect("Typ", typen, default=typen)
    gefiltert = [f for f in frei if f["typ"] in gewaehlt] if gewaehlt else frei

    ui.tabelle([{
        "Gestell": f["gestell_id"],
        "Typ": f["typ"] or "–",
        "Plaetze": f["plaetze"] or "–",
        "Tag": "✓" if f["getaggt"] else "–",
        "Linie": f["linie"] or "–",
        "zuletzt gesehen": (f["zuletzt_gesehen"] or "nie").replace("T", " "),
        "Standort": f["station"] or "unbekannt",
    } for f in gefiltert], hoehe=480)


with historie_tab:
    alle = [g["gestell_id"] for g in models.gestelle(con)]
    gewaehltes = st.selectbox("Gestell", alle)
    if gewaehltes:
        g = models.gestell(con, gewaehltes)
        ort = auswertung.aktueller_standort(con, gewaehltes)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Typ", g["typ"] or "–")
        s2.metric("Plaetze", g["plaetze"] or "–")
        s3.metric("RFID-Tag", g["epc"] or "keiner")
        s4.metric("Standort", ort["station_name"] if ort else "unbekannt")

        offen = models.offene_belegungen(con, gestell_id=gewaehltes)
        if offen:
            st.write("**Haengt aktuell:**")
            ui.tabelle([{
                "Auftrag": b["auftrag_nr"],
                "Kunde": b["kunde"] or "–",
                "Menge": b["menge"] or "–",
                "seit": b["von"].replace("T", " "),
            } for b in offen])
        else:
            st.info("Gestell ist frei.")

        st.write("**Letzte Bewegungen:**")
        ui.tabelle([{
            "Zeit": e["ts"].replace("T", " "),
            "Station": e["station_name"],
            "Quelle": e["quelle"],
            "RSSI": e["rssi"] if e["rssi"] is not None else "–",
            "Bemerkung": e["bemerkung"] or "",
        } for e in models.historie(con, gewaehltes, limit=80)], hoehe=400)
