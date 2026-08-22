"""Tests der Fachlogik. Aufruf: python -m unittest discover -s tests"""

import os
import sys
import unittest
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracker import auswertung, config, db, models, simulate  # noqa: E402


class Basis(unittest.TestCase):
    def setUp(self):
        self.con = db.connect(":memory:")
        self.t0 = models.jetzt() - timedelta(hours=4)

    def tearDown(self):
        self.con.close()

    def gestell(self, gid="G-0001", epc="E280F3000001"):
        models.gestell_anlegen(self.con, gid, typ="T1", plaetze=24, epc=epc)
        return gid

    def auftrag(self, nr="A-00001"):
        models.auftrag_anlegen(self.con, nr, kunde="Testkunde", menge=100)
        return nr


class TestStammdaten(Basis):
    def test_stationen_aus_config_in_db(self):
        anzahl = self.con.execute("SELECT COUNT(*) FROM station").fetchone()[0]
        self.assertEqual(anzahl, len(config.STATIONEN))

    def test_gestell_ohne_tag_ist_zulaessig(self):
        models.gestell_anlegen(self.con, "G-0002", typ="T2", plaetze=16, epc=None)
        offen = models.gestelle_ohne_tag(self.con)
        self.assertEqual([g["gestell_id"] for g in offen], ["G-0002"])

    def test_epc_nur_einmal_vergeben(self):
        self.gestell("G-0001", "E280F3000001")
        models.gestell_anlegen(self.con, "G-0002", typ="T2")
        with self.assertRaises(ValueError):
            models.gestell_taggen(self.con, "G-0002", "E280F3000001")

    def test_taggen_schliesst_rollout_luecke(self):
        models.gestell_anlegen(self.con, "G-0009", typ="T1")
        self.assertEqual(auswertung.rollout_fortschritt(self.con)["offen"], 1)
        models.gestell_taggen(self.con, "G-0009", "E280F3000009")
        fortschritt = auswertung.rollout_fortschritt(self.con)
        self.assertEqual(fortschritt["offen"], 0)
        self.assertEqual(fortschritt["anteil"], 1.0)


class TestEvents(Basis):
    def test_doppellesung_wird_verworfen(self):
        gid = self.gestell()
        erste = models.event_erfassen(self.con, gid, "AU-VB", ts=self.t0)
        zweite = models.event_erfassen(
            self.con, gid, "AU-VB", ts=self.t0 + timedelta(seconds=5))
        self.assertIsNotNone(erste)
        self.assertIsNone(zweite)

    def test_erneute_lesung_nach_entprellfenster_zaehlt(self):
        gid = self.gestell()
        models.event_erfassen(self.con, gid, "AU-VB", ts=self.t0)
        spaeter = self.t0 + timedelta(seconds=models.ENTPRELL_SEKUNDEN + 1)
        self.assertIsNotNone(
            models.event_erfassen(self.con, gid, "AU-VB", ts=spaeter))

    def test_andere_station_wird_nie_entprellt(self):
        gid = self.gestell()
        models.event_erfassen(self.con, gid, "AU-VB", ts=self.t0)
        self.assertIsNotNone(models.event_erfassen(
            self.con, gid, "AU-ELX", ts=self.t0 + timedelta(seconds=5)))

    def test_unbekannter_epc_meldet_fehler_statt_zu_schlucken(self):
        self.gestell()
        eid, fehler = models.event_von_epc(self.con, "E280F3999999", "AU-VB")
        self.assertIsNone(eid)
        self.assertIn("E280F3999999", fehler)

    def test_bekannter_epc_wird_auf_gestell_aufgeloest(self):
        gid = self.gestell("G-0007", "E280F3000007")
        eid, fehler = models.event_von_epc(self.con, "E280F3000007", "AU-ELX")
        self.assertIsNone(fehler)
        self.assertIsNotNone(eid)
        self.assertEqual(
            auswertung.aktueller_standort(self.con, gid)["station_id"], "AU-ELX")

    def test_unbekannte_station_wird_abgelehnt(self):
        gid = self.gestell()
        with self.assertRaises(ValueError):
            models.event_erfassen(self.con, gid, "XX-XXX")

    def test_standort_ist_immer_das_juengste_event(self):
        gid = self.gestell()
        models.event_erfassen(self.con, gid, "AU-VB", ts=self.t0)
        models.event_erfassen(self.con, gid, "AU-ELX",
                              ts=self.t0 + timedelta(minutes=30))
        self.assertEqual(
            auswertung.aktueller_standort(self.con, gid)["station_id"], "AU-ELX")


class TestBelegung(Basis):
    def test_mehrere_auftraege_auf_einem_gestell(self):
        gid = self.gestell()
        self.auftrag("A-00001")
        self.auftrag("A-00002")
        models.einhaengen(self.con, gid, "A-00001", station_id="AU-EIN")
        models.einhaengen(self.con, gid, "A-00002", station_id="AU-EIN")
        offen = models.offene_belegungen(self.con, gestell_id=gid)
        self.assertEqual({b["auftrag_nr"] for b in offen}, {"A-00001", "A-00002"})

    def test_gleicher_auftrag_nicht_zweimal_offen(self):
        gid = self.gestell()
        self.auftrag("A-00001")
        models.einhaengen(self.con, gid, "A-00001")
        with self.assertRaises(ValueError):
            models.einhaengen(self.con, gid, "A-00001")

    def test_aushaengen_schliesst_alle_offenen_belegungen(self):
        gid = self.gestell()
        self.auftrag("A-00001")
        self.auftrag("A-00002")
        models.einhaengen(self.con, gid, "A-00001")
        models.einhaengen(self.con, gid, "A-00002")
        geschlossen = models.aushaengen(self.con, gid, station_id="AU-AUS")
        self.assertEqual(geschlossen, 2)
        self.assertEqual(models.offene_belegungen(self.con, gestell_id=gid), [])

    def test_auftrag_erst_fertig_wenn_er_auf_keinem_gestell_mehr_haengt(self):
        self.gestell("G-0001", "E280F3000001")
        self.gestell("G-0002", "E280F3000002")
        nr = self.auftrag("A-00001")
        models.einhaengen(self.con, "G-0001", nr)
        models.einhaengen(self.con, "G-0002", nr)

        models.aushaengen(self.con, "G-0001", station_id="AU-AUS")
        self.assertEqual(models.auftrag(self.con, nr)["status"], "laufend")

        models.aushaengen(self.con, "G-0002", station_id="AU-AUS")
        self.assertEqual(models.auftrag(self.con, nr)["status"], "fertig")

    def test_gestell_nach_aushaengen_wieder_frei(self):
        gid = self.gestell()
        nr = self.auftrag()
        models.einhaengen(self.con, gid, nr)
        self.assertNotIn(gid, [f["gestell_id"] for f in auswertung.freie_gestelle(self.con)])
        models.aushaengen(self.con, gid)
        self.assertIn(gid, [f["gestell_id"] for f in auswertung.freie_gestelle(self.con)])

    def test_unbekanntes_gestell_oder_auftrag_wird_abgelehnt(self):
        self.gestell()
        self.auftrag()
        with self.assertRaises(ValueError):
            models.einhaengen(self.con, "G-9999", "A-00001")
        with self.assertRaises(ValueError):
            models.einhaengen(self.con, "G-0001", "A-99999")


class TestLiegezeit(Basis):
    def test_schwellwerte(self):
        soll = 60
        self.assertEqual(auswertung.liegezeit_status(30, soll), "ok")
        self.assertEqual(auswertung.liegezeit_status(90, soll), "warnung")
        self.assertEqual(auswertung.liegezeit_status(150, soll), "alarm")
        self.assertEqual(auswertung.liegezeit_status(9999, None), "ohne_sollwert")

    def test_freies_gestell_loest_keinen_alarm_aus(self):
        gid = self.gestell()
        # Steht seit vier Stunden im Verpackungsbereich, aber ohne Auftrag.
        models.event_erfassen(self.con, gid, "ZZ-VER", ts=self.t0)
        zeile = auswertung.hallenuebersicht(self.con)[0]
        self.assertEqual(zeile["status"], "frei")
        self.assertEqual(auswertung.kennzahlen(self.con)["alarme"], 0)

    def test_belegtes_gestell_schlaegt_an(self):
        gid = self.gestell()
        nr = self.auftrag()
        models.einhaengen(self.con, gid, nr, ts=self.t0)
        models.event_erfassen(self.con, gid, "AU-ELX", ts=self.t0)  # soll 60 min
        zeile = auswertung.hallenuebersicht(self.con)[0]
        self.assertEqual(zeile["status"], "alarm")
        self.assertGreaterEqual(zeile["liegezeit_min"], 239)

    def test_gestell_ohne_event_hat_unbekannten_standort(self):
        self.gestell()
        self.assertEqual(auswertung.hallenuebersicht(self.con)[0]["status"], "unbekannt")


class TestAuswertung(Basis):
    def test_auftrag_ueber_mehrere_gestelle_verfolgen(self):
        self.gestell("G-0001", "E280F3000001")
        self.gestell("G-0002", "E280F3000002")
        nr = self.auftrag()
        models.einhaengen(self.con, "G-0001", nr, station_id="AU-EIN", ts=self.t0)
        models.einhaengen(self.con, "G-0002", nr, station_id="HA-EIN", ts=self.t0)
        models.event_erfassen(self.con, "G-0001", "AU-ELX",
                              ts=self.t0 + timedelta(minutes=40))
        treffer = auswertung.auftrag_verfolgen(self.con, nr)
        self.assertEqual(len(treffer), 2)
        orte = {t["gestell_id"]: t["station"] for t in treffer}
        self.assertEqual(orte["G-0001"], "Eloxal Automat")
        self.assertEqual(orte["G-0002"], "Einhaengeplatz Hand")

    def test_durchlaufzeit_erst_nach_abschluss(self):
        gid = self.gestell()
        nr = self.auftrag()
        models.einhaengen(self.con, gid, nr, ts=self.t0)
        self.assertIsNone(auswertung.durchlaufzeit_min(self.con, nr))
        models.aushaengen(self.con, gid, ts=self.t0 + timedelta(minutes=180))
        self.assertEqual(auswertung.durchlaufzeit_min(self.con, nr), 180)

    def test_linienwechsel_wird_mitgeschrieben(self):
        gid = self.gestell()
        models.event_erfassen(self.con, gid, "AU-ELX", ts=self.t0)
        models.event_erfassen(self.con, gid, "HA-ELX",
                              ts=self.t0 + timedelta(minutes=90))
        self.assertEqual(
            auswertung.aktueller_standort(self.con, gid)["linie_id"],
            config.HANDANLAGE)


class TestSimulation(Basis):
    def test_demo_liefert_konsistenten_bestand(self):
        simulate.demo_aufbauen(self.con, anzahl_gestelle=40, tage=2,
                               chargen_pro_tag=10)
        kpi = auswertung.kennzahlen(self.con)
        self.assertEqual(kpi["gestelle_gesamt"], 40)
        self.assertEqual(kpi["gestelle_belegt"] + kpi["gestelle_frei"], 40)
        self.assertEqual(kpi["rollout"]["getaggt"] + kpi["rollout"]["offen"], 40)

    def test_demo_ist_reproduzierbar(self):
        simulate.demo_aufbauen(self.con, anzahl_gestelle=30, tage=2,
                               chargen_pro_tag=8, seed=7)
        erste = auswertung.kennzahlen(self.con)
        simulate.demo_aufbauen(self.con, anzahl_gestelle=30, tage=2,
                               chargen_pro_tag=8, seed=7)
        self.assertEqual(erste["gestelle_belegt"],
                         auswertung.kennzahlen(self.con)["gestelle_belegt"])

    def test_takt_bewegt_gestelle_weiter(self):
        simulate.demo_aufbauen(self.con, anzahl_gestelle=40, tage=2,
                               chargen_pro_tag=10)
        vorher = self.con.execute("SELECT COUNT(*) FROM event").fetchone()[0]
        simulate.takt(self.con, anzahl=3, seed=1)
        self.assertGreater(
            self.con.execute("SELECT COUNT(*) FROM event").fetchone()[0], vorher)


if __name__ == "__main__":
    unittest.main(verbosity=2)
