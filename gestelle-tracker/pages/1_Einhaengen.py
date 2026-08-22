"""Erfassungsmaske fuer den Ein- und Aushaengeplatz.

Gedacht fuer ein Tablet oder Industrie-Panel direkt am Platz. Ein
Handscanner verhaelt sich wie eine Tastatur und schickt nach dem Scan
Enter -- deshalb sitzen die Scanfelder in einem Formular, das mit Enter
abgeschickt wird. Der Werker muss nichts anklicken.
"""

import streamlit as st

from tracker import auswertung, config, models, ui

ui.kopf("📥 Ein- und Aushaengeplatz",
        "Hier wird der Auftrag mit dem Gestell verknuepft – der wichtigste Schritt.")

con = ui.verbindung()

# Ohne Gestellstammdaten laesst sich nichts einhaengen. Die Seite bleibt
# trotzdem bedienbar, damit sich Auftraege schon anlegen lassen.
ui.leerer_bestand_hinweis(con)

EINHAENGE_STATIONEN = [s for s in config.STATIONEN if s["art"] == "einhaengen"]
AUSHAENGE_STATIONEN = [s for s in config.STATIONEN if s["art"] == "aushaengen"]


def gestell_aufloesen(eingabe):
    """Akzeptiert Gestellnummer, EPC oder DataMatrix-Inhalt.

    Der Werker soll nicht wissen muessen, was er da gerade gescannt hat.
    """
    eingabe = (eingabe or "").strip()
    if not eingabe:
        return None, "Bitte Gestell scannen oder eingeben."
    treffer = models.gestell(con, eingabe)
    if treffer:
        return treffer, None
    treffer = models.gestell_by_epc(con, eingabe)
    if treffer:
        return treffer, None
    # Haeufiger Fall: Nummer ohne Praefix eingetippt ("42" statt "G-0042").
    if eingabe.isdigit():
        treffer = models.gestell(con, f"G-{int(eingabe):04d}")
        if treffer:
            return treffer, None
    return None, f"Gestell '{eingabe}' nicht gefunden."


einhaengen_tab, aushaengen_tab, auftrag_tab = st.tabs(
    ["Einhaengen", "Aushaengen", "Auftrag anlegen"])


with einhaengen_tab:
    with st.form("einhaengen", clear_on_submit=True):
        s1, s2 = st.columns(2)
        gestell_eingabe = s1.text_input(
            "Gestell scannen", placeholder="G-0042 oder EPC",
            help="RFID-Tag, DataMatrix oder Nummer von Hand.")
        auftrag_eingabe = s2.text_input(
            "Auftrag scannen", placeholder="A-00123")
        s3, s4 = st.columns(2)
        menge = s3.number_input("Menge auf diesem Gestell", min_value=0,
                                step=1, value=0)
        station_id = s4.selectbox(
            "Platz", [s["station_id"] for s in EINHAENGE_STATIONEN],
            format_func=lambda sid: config.station(sid)["name"])
        abgeschickt = st.form_submit_button("Einhaengen", type="primary",
                                            width="stretch")

    if abgeschickt:
        g, fehler = gestell_aufloesen(gestell_eingabe)
        auftrag_nr = (auftrag_eingabe or "").strip()
        if fehler:
            st.error(fehler)
        elif models.auftrag(con, auftrag_nr) is None:
            st.error(f"Auftrag '{auftrag_nr}' ist nicht angelegt. "
                     "Im Reiter *Auftrag anlegen* anlegen.")
        else:
            try:
                models.einhaengen(con, g["gestell_id"], auftrag_nr,
                                  menge=menge or None, station_id=station_id)
            except ValueError as e:
                st.error(str(e))
            else:
                st.success(
                    f"Auftrag **{auftrag_nr}** haengt jetzt auf Gestell "
                    f"**{g['gestell_id']}**.")
                if g["epc"] is None:
                    st.warning(
                        f"Gestell {g['gestell_id']} hat noch keinen RFID-Tag. "
                        "Der weitere Weg wird nicht automatisch erfasst.")

    st.divider()
    st.subheader("Gerade eingehaengt")
    zeilen = [{
        "Gestell": b["gestell_id"],
        "Auftrag": b["auftrag_nr"],
        "Kunde": b["kunde"] or "–",
        "Menge": b["menge"] or "–",
        "seit": b["von"].replace("T", " "),
    } for b in models.offene_belegungen(con)]
    zeilen.sort(key=lambda z: z["seit"], reverse=True)
    ui.tabelle(zeilen[:20])


with aushaengen_tab:
    with st.form("aushaengen", clear_on_submit=True):
        s1, s2 = st.columns(2)
        gestell_aus = s1.text_input("Gestell scannen", placeholder="G-0042 oder EPC")
        station_aus = s2.selectbox(
            "Platz", [s["station_id"] for s in AUSHAENGE_STATIONEN],
            format_func=lambda sid: config.station(sid)["name"])
        aus_abgeschickt = st.form_submit_button("Aushaengen", type="primary",
                                                width="stretch")

    if aus_abgeschickt:
        g, fehler = gestell_aufloesen(gestell_aus)
        if fehler:
            st.error(fehler)
        else:
            offen = models.offene_belegungen(con, gestell_id=g["gestell_id"])
            if not offen:
                st.warning(f"Auf Gestell {g['gestell_id']} haengt kein Auftrag.")
            else:
                anzahl = models.aushaengen(con, g["gestell_id"],
                                           station_id=station_aus)
                namen = ", ".join(b["auftrag_nr"] for b in offen)
                st.success(f"{anzahl} Auftrag/Auftraege ausgehaengt: {namen}")
                for b in offen:
                    dauer = auswertung.durchlaufzeit_min(con, b["auftrag_nr"])
                    if dauer is not None:
                        st.caption(f"{b['auftrag_nr']}: Durchlaufzeit "
                                   f"{dauer // 60} h {dauer % 60} min")


with auftrag_tab:
    st.caption("Solange die Schaltzentrale noch nicht angebunden ist, "
               "werden Auftraege hier von Hand angelegt.")
    with st.form("auftrag_anlegen", clear_on_submit=True):
        s1, s2, s3 = st.columns(3)
        nr = s1.text_input("Auftragsnummer", placeholder="A-00123")
        kunde = s2.text_input("Kunde")
        artikel = s3.text_input("Artikel")
        s4, s5, s6, s7 = st.columns(4)
        menge_neu = s4.number_input("Menge", min_value=0, step=1, value=0)
        farbe = s5.selectbox("Farbe", [""] + config.FARBEN)
        dicke = s6.selectbox("Schichtdicke", [""] + config.SCHICHTDICKEN)
        termin = s7.date_input("Liefertermin", value=None)
        anlegen = st.form_submit_button("Auftrag anlegen", type="primary")

    if anlegen:
        if not nr.strip():
            st.error("Auftragsnummer fehlt.")
        else:
            models.auftrag_anlegen(
                con, nr.strip(), kunde=kunde or None, artikel=artikel or None,
                menge=menge_neu or None, farbe=farbe or None,
                schichtdicke=dicke or None,
                liefertermin=termin.isoformat() if termin else None)
            st.success(f"Auftrag {nr.strip()} angelegt.")
