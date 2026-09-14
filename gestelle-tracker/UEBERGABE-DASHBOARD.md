# Übergabe: Gestellverfolgung per RFID → Dashboard-Kachel

**Stand:** 14.09.2026 · Jacobi Eloxal · Hardware ist beschafft und eingetroffen

Dieses Dokument fasst eine komplette Konzeptphase zusammen, damit im
Dashboard-Projekt ohne Rückfragen weitergearbeitet werden kann. Alle
Entscheidungen unten sind getroffen, nicht mehr offen — ausgenommen der
Abschnitt **Offene Punkte**.

---

## 1. Worum es geht

In der Eloxal-Fertigung laufen rund **400 Gestelle** (Warenträger) um.
Heute weiß niemand zuverlässig, wo ein Auftrag gerade steckt und welche
Gestelle frei sind. Gesucht war eine kostengünstige automatische Erfassung.

**Bewusst NICHT das Ziel:** Verfolgung auf Badebene („in welchem Becken
hängt das Gestell gerade"). Das wurde verworfen — zu teuer, zu aufwendig,
und im Betrieb gar nicht gebraucht.

**Das Ziel:** eine grobe Zustandseinteilung, automatisch erfasst beim
Vorbeifahren, ohne dass jemand scannen muss.

---

## 2. Betriebliche Randbedingungen

- **Zwei Linien:** ein Automat und eine Handanlage. Dieselben Gestelle
  laufen über beide — die Linie muss mitgeschrieben werden.
- **Die Krane lassen sich nicht auslesen.** Eine Anbindung an die
  Anlagensteuerung ist ausgeschlossen; die Erfassung muss eigenständig sein.
- **Die Tags sitzen trocken.** Oben an der Schiene bzw. auf der Traverse,
  nie im Tauchbereich. Belastung: Spritzwasser und Wasserdampf, keine
  Badchemie. Das war die entscheidende Randbedingung für die Technikwahl.
- **Abstand Gestell ↔ Lesegerät am Tor: maximal 1,5 m.**

---

## 3. Zustandsmodell

Vier Zustände, **zwei physische Erfassungspunkte**, ein manueller Abschluss.

```
  frei / am Lagerplatz
        │  (Auftrag wird kontaktiert, Gestell fährt los)
        ▼
  ┌─────────────────┐
  │  STATION A      │  Durchgang zum Eloxieren
  └─────────────────┘
        ▼
   in der Anlage
        │  (Eloxalprozess)
        ▼
  ┌─────────────────┐
  │  STATION B      │  vom Eloxieren Richtung Dekontaktieren
  └─────────────────┘
        ▼
   fertig eloxiert
        ▼
   beim Dekontaktieren
        │  (Mitarbeiter bucht von Hand)
        ▼
   verpackt  →  Gestell wieder frei, Zyklus geschlossen
```

**Wichtig:** Station A und B werden im Umlauf mehrfach passiert. Welcher
Zustandswechsel gemeint ist, ergibt sich **nicht aus der Lesung allein**,
sondern aus dem vorherigen Zustand (siehe Erfassungslogik).

---

## 4. Beschaffte Hardware

### Am Gestell — Transponder

| | |
|---|---|
| Produkt | **Confidex Ironside** (On-Metal-Hardtag) |
| Bezug | barcode-shop.com, Art. CON-Confidex-Ironside |
| Preis | **5,42 €** netto/Stück (Mindestbestellwert 50 €) |
| Maße | 51,5 × 47,5 × 10 mm, zwei Schraublöcher |
| Chip | Monza4QT · EPC 128 Bit · TID 96 Bit · User 512 Bit |
| Standard | EPC1 Gen2 / ISO18000-6C |
| Montage | angeschraubt oder genietet auf der Traverse, **nicht geklebt** |

Passiv, ohne Batterie — wartungsfrei, unempfindlich gegen Dampf und
Kondensat. Bei 400 Gestellen war das der entscheidende Grund gegen BLE.

**Montageregel:** immer dieselbe Position und dieselbe Ausrichtung an
jedem Gestell, sonst schwankt die Leserate.

### Am Tor — Lesegerät

| | |
|---|---|
| Produkt | **Yanzeo SR682** — Reader und Antenne in einem Gehäuse |
| Preis | **259 €** brutto (Amazon) |
| Schutzart | **IP67**, ABS+PC-Gehäuse |
| Antenne | 8 dBi, zirkular, Öffnungswinkel ca. 60–90° |
| Reichweite | 3–8 m (Herstellerangabe) |
| Sendeleistung | **einstellbar 0–30 dBm** |
| Protokolle | ISO18000-6C / EPC Gen2, ISO18000-6B |
| Anschlüsse | **RJ45 (Netzwerk)**, USB, RS485, Wiegand26/34 |
| Besonderheit | **liefert RSSI pro Lesung**, Python-Demo-SDK verfügbar |
| Versorgung | 12 V (**kein PoE** — RJ45 ist reine Datenleitung) |
| SDK | yanzeo.com/download |

**Warum dieses Gerät:** IP67 löst die Dampffrage; die einstellbare
Sendeleistung erlaubt, das Lesefeld von 8 m auf die benötigten ~2 m zu
verkleinern; RSSI und Python-SDK sind für die Auswertung entscheidend.
Ein Markengerät (Kathrein ARU/RRU, Impinj R700, FEIG) hätte 1.500–2.500 €
gekostet — bei 259 € ist ein Reservegerät im Schrank die bessere
Ausfallsicherung als ein teureres Einzelgerät.

**Kein Raspberry Pi nötig** — der SR682 hängt über RJ45 direkt am Netz.
Der Aufnahmedienst läuft auf dem Rechner, auf dem auch die Schaltzentrale
läuft.

---

## 5. Erfassungslogik — vier Regeln

Das ist der wichtigste Abschnitt. Die Regeln decken **verschiedene**
Fehlerarten ab; keine ersetzt eine andere.

### Regel 1 — Verweildauer-Filter (gegen parkende Gestelle)

Ein Gestell, das neben dem Lesepunkt abgestellt wird, liegt dauerhaft im
Feld und würde sich sonst von allein durch die Zustandskette arbeiten.
Eine reine Zeitsperre hilft nicht — sie streckt den Fehler nur.

Der Unterschied zwischen „fährt durch" und „steht daneben" ist die
**Dauer der Sichtung**:

```
Tag wird erstmals gelesen      → Sichtung beginnt, t_start merken
Tag wird weiter gelesen        → t_last aktualisieren, max. RSSI mitschreiben
ca. 3 s keine Lesung mehr      → Sichtung beendet

Sichtungsdauer  <  ~30 s       → Durchfahrt  → Zustandswechsel
Sichtungsdauer  >  ~30 s       → Parker      → KEIN Zustandswechsel
```

Geschaltet wird also auf der **fallenden Flanke** (Tag verlässt das Feld),
nicht beim Auftauchen.

### Regel 2 — Reihenfolgeregel (gegen Sprünge)

Ein Zustandswechsel ist nur gültig, wenn der **vorherige Zustand passt**.
Ohne Durchfahrt an Station A kein Wechsel an Station B.

Der eigentliche Gewinn dieser Regel: sie verwandelt Fehler von *falsch*
in *hängengeblieben*. Ein hängengebliebenes Gestell fällt auf und wird
korrigiert; ein falsch gebuchtes sieht korrekt aus und wird geglaubt.

**Rückwärtsschritt** (zurück auf „frei / am Lagerplatz") ist nur erlaubt,
nachdem der Zyklus durch die manuelle Verpackt-Buchung geschlossen wurde.

### Regel 3 — Mindestverweildauer je Zustand (gegen Lesefehler)

Eloxieren dauert seine Zeit. Taucht ein Gestell zwei Minuten nach
Station A schon bei Station B auf, ist das keine Charge, sondern ein
Lesefehler. Der Mindestwert ist betrieblich bekannt und wird als
Plausibilitätsprüfung gesetzt.

### Regel 4 — Der Mensch gewinnt, aber es wird protokolliert

Die manuelle Buchung beim Dekontaktieren überschreibt jeden
Automatikzustand. Sie ist die höchste Vertrauensstufe.

**Aber:** jede Korrektur wird als Korrektur mitgeschrieben, inklusive der
Station, an der die Automatik versagt hat.

Grund: die **Korrekturquote ist die einzige Qualitätskennzahl** des
Systems. Bei 2–3 % läuft alles. Springt sie auf 20 %, ist etwas passiert —
Antenne verschoben, Tag abgerissen, jemand stellt vor dem Tor ab. Ohne
dieses Mitschreiben merkt man die schleichende Verschlechterung erst,
wenn niemand dem System mehr glaubt.

### Zusätzliche Hebel (Hardware/Organisation)

- **Sendeleistung herunterdrehen** — bei 1,5 m wird nur ein Bruchteil der
  Geräteleistung gebraucht. Kleines Feld = wenig Störlesungen.
- **RSSI-Schwellwert** — nur Sichtungen über einem Grenzwert zählen.
  Nachjustierbar im Betrieb, ohne an die Hardware zu müssen.
- **Zwei Lichtschranken je Station** — aus der Auslösereihenfolge ergibt
  sich die Fahrtrichtung. Ein parkendes Gestell unterbricht sie nie.
- **Freihaltezone vor dem Tor markieren** (ca. 3 m). Kostet einen Eimer
  Farbe am Anfang und ist später kaum noch durchzusetzen.

---

## 6. Datenmodell

Bereits als Prototyp umgesetzt (siehe Abschnitt 11). Drei Tabellen tragen
alles.

### `gestell` — Stammdaten
```
gestell_id   TEXT PK      z. B. "G-0042"  — bleibt für immer
epc          TEXT UNIQUE  RFID-Nummer, DARF NULL SEIN
typ, plaetze, status, notiz, angelegt_am
```
`epc = NULL` ist der Normalzustand während des Rollouts, nicht die
Ausnahme. 400 Gestelle werden rollierend ausgerüstet — am besten beim
Aushängen, wenn das Gestell ohnehin leer dasteht. Gestelle ohne Tag
bleiben von Hand erfassbar, nur ohne automatischen Standort.

### `belegung` — Auftrag ↔ Gestell, zeitlich begrenzt
```
belegung_id PK, gestell_id, auftrag_nr, menge, von, bis, bemerkung
```
Bewusst n:m mit `von`/`bis`, keine Spalte am Gestell — ein Gestell wird
immer wieder neu belegt und kann mehrere Aufträge tragen. `bis IS NULL`
heißt „hängt gerade". Ein Auftrag gilt erst als fertig, wenn er auf
**keinem** Gestell mehr hängt.

### `event` — Standortmeldungen, nur angehängt
```
event_id PK, ts, gestell_id, station_id, linie_id,
quelle (rfid|scan|manuell), epc_gelesen, rssi, bemerkung
```
Der aktuelle Zustand wird **immer aus dem jüngsten Event berechnet** und
nirgends gespeichert. Damit können Zustand und Historie nicht
auseinanderlaufen. Nebeneffekt: vollständige Rückverfolgung, welches
Gestell wann wo war — die Datenbasis, die der KI-Fehleranalyse
(`main.py` im Repo-Stamm) bisher fehlt.

---

## 7. Tag-Zuordnung

Jeder Tag hat eine weltweit eindeutige Nummer ab Werk (TID fest, EPC
nutzbar). Der Reader unterscheidet auch bei mehreren Tags im Feld immer
sauber, welches Gestell er sieht — Antikollision ist in EPC Gen2
eingebaut. Die Unsicherheit ist nie „welches Gestell", sondern nur
„fährt es oder steht es".

**Zuordnung über eine Tabelle**, nicht über Tag-Programmierung:

```
EPC E280F3A419…  →  Gestell G-0042
```

Ablauf beim Antaggen (am Tablet, ca. 20 s je Gestell): Tag scannen,
Gestellnummer wählen, speichern.

**Vorteil:** Geht ein Tag kaputt, wird ein neuer mit derselben
Gestellnummer gepaart. Die gesamte Historie hängt an der Gestellnummer,
nicht am Chip.

**Zusätzlich:** Gestellnummer sichtbar aufs Gestell (graviertes
Edelstahlschild), damit ein Mensch es ohne Lesegerät identifizieren kann.

---

## 8. Architektur der Anbindung

```
SR682 am Tor ──RJ45──► Netzwerk ──► Aufnahmedienst ──► Datenbank ──► Dashboard
  (2 Stück)                          (~50 Zeilen)
```

Der Aufnahmedienst läuft auf dem Rechner, auf dem die Schaltzentrale
liegt. Kein eigenes Gerät am Tor, kein Pi, kein Schaltschrank.

Die Naht zur Fachlogik ist eine einzige Funktion:

```python
models.event_von_epc(con, epc, station_id, rssi=rssi)
```

Egal ob der Reader per TCP, UDP, HTTP-Callback oder seriell liefert —
am Ende steht dieser Aufruf. Alles dahinter bleibt unverändert.

---

## 9. Offene Punkte

Diese sind **noch nicht geklärt** und blockieren teilweise:

1. **ETSI-Einstellung am Reader.** Das Gerät ist als Breitband
   860–960 MHz beworben. In Deutschland gilt **865–868 MHz nach
   ETSI EN 302 208**. In der Konfigurationssoftware muss die Region auf
   ETSI/EU stellbar sein — das ist beim ersten Einschalten zu prüfen.
   CE-Konformitätserklärung beim Händler anfragen.
2. **Digitale Eingänge.** Im mitgelieferten Handbuch ist ein Anschlussplan
   mit Trigger-Eingang abgebildet. Klemmenbelegung im Handbuch
   verifizieren — davon hängt ab, ob die Lichtschranken direkt ans Gerät
   können oder ein kleines IO-Modul (50–80 €) nötig wird.
3. **Länge des angegossenen Anschlusskabels** — bestimmt, ob verlängert
   werden muss.
4. **Schaltzentrale: Technik unbekannt.** Access, SQL Server, Excel,
   Weblösung? Und besteht direkter Datenbankzugriff oder nur Export?
   Solange das offen ist, werden Aufträge im Prototyp von Hand angelegt.
5. **Betriebstemperaturbereich** des SR682 — in der Anzeige nirgends
   genannt, im Datenblatt nachsehen.

---

## 10. Inbetriebnahme — Testplan

Nicht sofort ans Tor schrauben. In dieser Reihenfolge:

1. **Am Schreibtisch.** Gerät ans Netz, Demo-Software, ein Ironside
   danebenlegen. Wird gelesen? Welche Region ist eingestellt, steht ETSI
   zur Auswahl?
2. **Region auf ETSI, Leistung auf Minimum.** Mit Zollstock messen, ab
   welcher Entfernung gelesen wird. Leistung schrittweise erhöhen bis
   ca. 2 m. **Diesen Wert notieren** — das ist die Arbeitseinstellung.
3. **RSSI aufnehmen.** Tag auf 1,5 m und auf 3 m. Der Abstand zwischen
   den Werten ist der spätere Schwellwert.
4. **Ans Tor.** Ironside an eine Traverse, dann **20–30 echte
   Durchfahrten zählen**.
   **Maßstab: 100 %, nicht 95 %.** Bei 95 % fehlt jede zwanzigste
   Durchfahrt — und weil die Zustandskette darauf aufbaut, steht danach
   jedes betroffene Gestell falsch. Eine Leserate, die für ein Lager
   reicht, reicht hier nicht.
5. **Gegenprobe.** Zweites Gestell 3 m neben der Durchfahrt abstellen.
   Wird es mitgelesen? Wenn ja: Leistung runter oder RSSI-Schwelle
   anheben, dann Schritt 4 wiederholen.

---

## 11. Was im Repo schon existiert

Branch `claude/gestelle-auftraege-tracker-xl6utl`, Ordner
`gestelle-tracker/` — ein lauffähiger Prototyp mit simulierten
Lesepunkten, 42 Tests (13 davon durch die Oberfläche).

```
app.py                  Startseite mit Kennzahlen
pages/                  Einhängen · Hallenübersicht · Auftragssuche ·
                        Gestellpark · Simulator
beispiel_reader.py      Gerüst für die echte Reader-Anbindung
tracker/
  config.py             Linien, Stationen, Soll-Dauern  ← hier anpassen
  db.py                 SQLite-Schema
  models.py             Gestelle, Aufträge, Belegung, Events
  auswertung.py         Standort, Liegezeiten, Verfügbarkeit
  simulate.py           Demo-Betrieb ohne Hardware
```

Start: `pip install -r requirements.txt` · `streamlit run app.py`,
dann Seite *Simulator → Demo-Bestand erzeugen* (400 Gestelle, eine Woche
Historie).

**Achtung:** Der Prototyp bildet noch das **alte, feingliedrige**
Stationsmodell ab (Vorbehandlung, Eloxal, Färberei, Verdichtung). Das
vereinfachte Vier-Zustands-Modell aus Abschnitt 3 ist dort **noch nicht**
umgesetzt — `tracker/config.py` muss darauf umgestellt werden. Datenmodell
und Auswertungslogik bleiben unverändert gültig.

---

## 12. Was die Dashboard-Kachel zeigen sollte

Vorschlag als Ausgangspunkt — die endgültige Gestaltung richtet sich nach
dem bestehenden Dashboard.

**Kernzahlen (immer sichtbar):**

| Kennzahl | Quelle |
|---|---|
| Gestelle in der Anlage | Anzahl im Zustand *in der Anlage* |
| Fertig eloxiert, wartet | Anzahl im Zustand *fertig eloxiert* |
| Beim Dekontaktieren | Anzahl im Zustand *beim Dekontaktieren* |
| Freie Gestelle | ohne offene Belegung |
| Hängengeblieben | Zustand länger als Sollzeit unverändert |
| Korrekturquote (7 Tage) | manuelle Korrekturen ÷ Durchfahrten |

**Warum „Hängengeblieben" und „Korrekturquote" mit auf die Kachel
gehören:** Regel 2 macht aus Erfassungsfehlern hängengebliebene Gestelle —
die müssen sichtbar sein, sonst nützt die Regel nichts. Und die
Korrekturquote ist das Frühwarnsignal für die Qualität der ganzen
Erfassung.

**Benötigte Datenstruktur je Gestell:**

```json
{
  "gestell_id": "G-0042",
  "epc": "E280F3A419…",
  "zustand": "in_der_anlage",
  "seit": "2026-09-14T08:31:00",
  "linie": "Automat",
  "auftraege": ["A-01234"],
  "kunde": "Muster Metallbau GmbH",
  "liegezeit_min": 74,
  "soll_dauer_min": 60,
  "status": "warnung",
  "letzte_station": "A",
  "letzte_quelle": "rfid"
}
```

**Zustandswerte:** `frei` · `in_der_anlage` · `fertig_eloxiert` ·
`beim_dekontaktieren` · `unbekannt`
**Statuswerte:** `ok` · `warnung` · `alarm` · `haengengeblieben` · `frei`

---

## 13. Kostenübersicht

| Position | Betrag |
|---|---|
| 400 × Confidex Ironside à 5,42 € | ~2.200 € |
| 3 × Yanzeo SR682 (2 im Betrieb, 1 Reserve) | ~780 € |
| 2 × 12-V-Versorgung bzw. PoE-Splitter | ~60 € |
| 4 × Lichtschranke (2 je Station, Richtungserkennung) | ~400 € |
| Bedienteil am Dekontaktierplatz | ~500 € |
| **Summe** | **~3.940 €** |

Zum Vergleich: die erste Schätzung für eine Verfolgung auf Badebene lag
bei rund 15.000 €. Den größten Teil der Einsparung brachte die
Vereinfachung auf zwei Erfassungspunkte, nicht die Hardwareauswahl.
