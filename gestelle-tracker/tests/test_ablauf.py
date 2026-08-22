"""Der Einhaenge-/Aushaenge-Ablauf durch die echte Oberflaeche.

Das ist der Schritt, an dem im Betrieb alles haengt: wird hier nicht sauber
verknuepft, nuetzt der beste Lesepunkt nichts. Deshalb wird er nicht nur
auf Logikebene, sondern durch die Maske getestet.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WURZEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEITE = os.path.join(WURZEL, "pages", "1_Einhaengen.py")

# Feldpositionen auf der Seite (Reihenfolge der Reiter).
EIN_GESTELL, EIN_AUFTRAG, AUS_GESTELL = 0, 1, 2
BTN_EIN, BTN_AUS = 0, 1


class TestEinhaengeAblauf(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["GESTELLE_DB"] = os.path.join(self.tmp.name, "t.db")

        import streamlit as st
        st.cache_resource.clear()

        from tracker import db, models
        self.con = db.connect(os.environ["GESTELLE_DB"])
        models.gestell_anlegen(self.con, "G-0042", typ="T1", plaetze=24,
                               epc="E280F3000042")
        models.gestell_anlegen(self.con, "G-0099", typ="T2", plaetze=16)  # ohne Tag
        models.auftrag_anlegen(self.con, "A-00123", kunde="Muster GmbH", menge=100)
        models.auftrag_anlegen(self.con, "A-00124", kunde="Muster GmbH", menge=50)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()
        os.environ.pop("GESTELLE_DB", None)
        import streamlit as st
        st.cache_resource.clear()

    def seite(self):
        from streamlit.testing.v1 import AppTest
        test = AppTest.from_file(SEITE, default_timeout=60)
        test.run()
        return test

    def einhaengen(self, gestell, auftrag):
        test = self.seite()
        test.text_input[EIN_GESTELL].set_value(gestell)
        test.text_input[EIN_AUFTRAG].set_value(auftrag)
        test.button[BTN_EIN].click()
        test.run()
        self.assertFalse(test.exception)
        return test

    def offene(self, gestell_id=None):
        from tracker import models
        return models.offene_belegungen(self.con, gestell_id=gestell_id)

    # -- Gutfall ----------------------------------------------------------

    def test_einhaengen_verknuepft_auftrag_mit_gestell(self):
        test = self.einhaengen("G-0042", "A-00123")
        self.assertTrue(any("A-00123" in s.value for s in test.success))
        offen = self.offene("G-0042")
        self.assertEqual([b["auftrag_nr"] for b in offen], ["A-00123"])

    def test_einhaengen_erzeugt_standort_event(self):
        from tracker import auswertung
        self.einhaengen("G-0042", "A-00123")
        ort = auswertung.aktueller_standort(self.con, "G-0042")
        self.assertEqual(ort["station_id"], "AU-EIN")

    # -- Eingabewege, die der Werker tatsaechlich benutzt ------------------

    def test_gestell_per_epc_scannen(self):
        self.einhaengen("E280F3000042", "A-00123")
        self.assertEqual(len(self.offene("G-0042")), 1)

    def test_gestell_als_blosse_nummer_eintippen(self):
        self.einhaengen("42", "A-00123")
        self.assertEqual(len(self.offene("G-0042")), 1)

    def test_umgebende_leerzeichen_stoeren_nicht(self):
        self.einhaengen("  G-0042 ", " A-00123 ")
        self.assertEqual(len(self.offene("G-0042")), 1)

    # -- Fehlbedienung ----------------------------------------------------

    def test_unbekanntes_gestell_wird_abgewiesen(self):
        test = self.einhaengen("G-9999", "A-00123")
        self.assertTrue(any("nicht gefunden" in e.value for e in test.error))
        self.assertEqual(self.offene(), [])

    def test_unbekannter_auftrag_wird_abgewiesen(self):
        test = self.einhaengen("G-0042", "A-99999")
        self.assertTrue(any("nicht angelegt" in e.value for e in test.error))
        self.assertEqual(self.offene(), [])

    def test_leere_eingabe_wird_abgewiesen(self):
        test = self.einhaengen("", "A-00123")
        self.assertTrue(any("Bitte Gestell" in e.value for e in test.error))

    def test_gestell_ohne_tag_wird_gewarnt_aber_erfasst(self):
        test = self.einhaengen("G-0099", "A-00123")
        self.assertTrue(any("noch keinen RFID-Tag" in w.value for w in test.warning))
        self.assertEqual(len(self.offene("G-0099")), 1)

    def test_derselbe_auftrag_nicht_zweimal_auf_dasselbe_gestell(self):
        self.einhaengen("G-0042", "A-00123")
        test = self.einhaengen("G-0042", "A-00123")
        self.assertTrue(any("haengt bereits" in e.value for e in test.error))
        self.assertEqual(len(self.offene("G-0042")), 1)

    # -- Aushaengen -------------------------------------------------------

    def test_aushaengen_schliesst_belegung(self):
        self.einhaengen("G-0042", "A-00123")
        test = self.seite()
        test.text_input[AUS_GESTELL].set_value("G-0042")
        test.button[BTN_AUS].click()
        test.run()
        self.assertFalse(test.exception)
        self.assertTrue(any("A-00123" in s.value for s in test.success))
        self.assertEqual(self.offene("G-0042"), [])

    def test_aushaengen_ohne_auftrag_warnt(self):
        test = self.seite()
        test.text_input[AUS_GESTELL].set_value("G-0042")
        test.button[BTN_AUS].click()
        test.run()
        self.assertTrue(any("kein Auftrag" in w.value for w in test.warning))

    def test_zwei_auftraege_auf_einem_gestell_gemeinsam_aushaengen(self):
        self.einhaengen("G-0042", "A-00123")
        self.einhaengen("G-0042", "A-00124")
        self.assertEqual(len(self.offene("G-0042")), 2)
        test = self.seite()
        test.text_input[AUS_GESTELL].set_value("G-0042")
        test.button[BTN_AUS].click()
        test.run()
        self.assertEqual(self.offene("G-0042"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
