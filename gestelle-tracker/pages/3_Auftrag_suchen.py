"""Die Frage, um die es eigentlich geht: wo ist mein Auftrag gerade?"""

import streamlit as st

from tracker import auswertung, models, ui

ui.kopf("🔎 Auftrag suchen", "Auftragsnummer eingeben – oder Kundennamen.")

con = ui.verbindung()
if ui.leerer_bestand_hinweis(con):
    st.stop()

suche = st.text_input("Auftrag oder Kunde", placeholder="A-00123 / Muster Metallbau")

if not suche.strip():
    st.subheader("Laufende Auftraege")
    zeilen = []
    for a in models.auftraege(con, status="laufend"):
        orte = auswertung.auftrag_verfolgen(con, a["auftrag_nr"])
        zeilen.append({
            "Auftrag": a["auftrag_nr"],
            "Kunde": a["kunde"] or "–",
            "Artikel": a["artikel"] or "–",
            "Farbe": a["farbe"] or "–",
            "Gestelle": len(orte),
            "Steht gerade": ", ".join(
                sorted({o["station"] for o in orte if o["station"]})) or "–",
            "Liefertermin": a["liefertermin"] or "–",
        })
    ui.tabelle(zeilen, hoehe=520)
    st.stop()

begriff = suche.strip().lower()
treffer = [a for a in models.auftraege(con)
           if begriff in a["auftrag_nr"].lower()
           or (a["kunde"] and begriff in a["kunde"].lower())]

if not treffer:
    st.warning("Kein Auftrag gefunden.")
    st.stop()

st.caption(f"{len(treffer)} Treffer")

for a in treffer[:20]:
    kopfzeile = (f"{a['auftrag_nr']} · {a['kunde'] or '–'} · "
                 f"{a['artikel'] or '–'} · {a['status']}")
    with st.expander(kopfzeile, expanded=len(treffer) == 1):
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Menge", a["menge"] or "–")
        s2.metric("Farbe", a["farbe"] or "–")
        s3.metric("Schichtdicke", a["schichtdicke"] or "–")
        dauer = auswertung.durchlaufzeit_min(con, a["auftrag_nr"])
        s4.metric("Durchlaufzeit",
                  f"{dauer // 60} h {dauer % 60} min" if dauer is not None else "laeuft")

        orte = auswertung.auftrag_verfolgen(con, a["auftrag_nr"])
        if orte:
            st.write("**Haengt gerade auf:**")
            ui.tabelle([{
                "Gestell": o["gestell_id"],
                "Menge": o["menge"] or "–",
                "Linie": o["linie"] or "–",
                "Station": o["station"] or "kein Standort",
                "dort seit": (o["seit"] or "–").replace("T", " "),
                "eingehaengt": o["seit_einhaengen"].replace("T", " "),
            } for o in orte])
        else:
            st.info("Haengt auf keinem Gestell mehr – Auftrag ist durch.")

        # Kompletter Weg durch die Anlage: die Grundlage fuer die
        # Fehlerrueckverfolgung.
        st.write("**Weg durch die Anlage:**")
        gestelle = {b["gestell_id"] for b in con.execute(
            "SELECT gestell_id FROM belegung WHERE auftrag_nr = ?",
            (a["auftrag_nr"],))}
        weg = []
        for gid in sorted(gestelle):
            for e in reversed(models.historie(con, gid, limit=200)):
                weg.append({
                    "Zeit": e["ts"].replace("T", " "),
                    "Gestell": gid,
                    "Station": e["station_name"],
                    "Quelle": e["quelle"],
                    "Bemerkung": e["bemerkung"] or "",
                })
        weg.sort(key=lambda z: z["Zeit"])
        ui.tabelle(weg)
