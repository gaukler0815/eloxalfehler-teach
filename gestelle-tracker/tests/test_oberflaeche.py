"""Rauchtest der Streamlit-Seiten.

Fuehrt jede Seite tatsaechlich aus und prueft, dass keine Ausnahme fliegt.
Aufruf: python -m unittest discover -s tests
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WURZEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEITEN = [
    "app.py",
    "pages/1_Einhaengen.py",
    "pages/2_Hallenuebersicht.py",
    "pages/3_Auftrag_suchen.py",
    "pages/4_Gestellpark.py",
    "pages/5_Simulator.py",
]


class TestSeiten(unittest.TestCase):
    """Jede Seite muss mit gefuellter und mit leerer Datenbank durchlaufen."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db_pfad = os.path.join(cls.tmp.name, "test.db")
        os.environ["GESTELLE_DB"] = cls.db_pfad

        from tracker import db, simulate
        con = db.connect(cls.db_pfad)
        simulate.demo_aufbauen(con, anzahl_gestelle=60, tage=3, chargen_pro_tag=15)
        con.close()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        os.environ.pop("GESTELLE_DB", None)

    def _ausfuehren(self, seite):
        from streamlit.testing.v1 import AppTest
        test = AppTest.from_file(os.path.join(WURZEL, seite), default_timeout=60)
        test.run()
        self.assertFalse(
            test.exception,
            f"{seite} wirft: "
            + "; ".join(str(e.message) for e in test.exception),
        )
        return test

    def test_seiten_laufen_mit_daten(self):
        for seite in SEITEN:
            with self.subTest(seite=seite):
                self._ausfuehren(seite)

    def test_seiten_laufen_mit_leerer_datenbank(self):
        from tracker import db
        leer = os.path.join(self.tmp.name, "leer.db")
        os.environ["GESTELLE_DB"] = leer
        db.connect(leer).close()
        try:
            for seite in SEITEN:
                with self.subTest(seite=seite):
                    test = self._ausfuehren(seite)
                    # Beweis, dass wirklich die leere Datenbank benutzt wurde:
                    # jede Seite weist auf den fehlenden Bestand hin.
                    hinweise = [i.value for i in test.info]
                    self.assertTrue(
                        any("Noch keine Daten" in h for h in hinweise),
                        f"{seite} zeigt keinen Hinweis auf den leeren Bestand: {hinweise}")
        finally:
            os.environ["GESTELLE_DB"] = self.db_pfad


if __name__ == "__main__":
    unittest.main(verbosity=2)
