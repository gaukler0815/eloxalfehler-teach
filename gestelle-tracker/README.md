# Gestell- und Auftragsverfolgung

Wo steckt gerade welcher Auftrag – über Automat und Handanlage hinweg,
bei rund 400 Gestellen im Umlauf.

Der Prototyp läuft komplett ohne Hardware. Die Lesepunkte sind simuliert,
damit sich das fertige System beurteilen lässt, **bevor eine einzige
Antenne bestellt ist**. Wird später ein echter Reader angeschlossen,
ändert sich an der Fachlogik nichts – nur die Datenquelle.

---

## Start

```
pip install -r requirements.txt
streamlit run app.py
```

Beim ersten Start ist die Datenbank leer. Auf der Seite **Simulator →
Demo-Bestand erzeugen** entsteht mit einem Klick ein vollständiger
Betrieb: 400 Gestelle, mehrere hundert Aufträge und eine Woche Historie.

Tests:

```
python -m unittest discover -s tests
```

---

## Die Seiten

| Seite | Wofür |
|---|---|
| **Start** | Kennzahlen, Belegung der Anlagen, was gerade zu lange steht |
| **Einhängen** | Erfassungsmaske für Tablet/Panel am Ein- und Aushängeplatz |
| **Hallenübersicht** | Jede Zeile ein Gestell: wo, seit wann, welcher Auftrag |
| **Auftrag suchen** | „Wo ist A-00123?" – inklusive komplettem Weg durch die Anlage |
| **Gestellpark** | RFID-Rollout, freie Gestelle, Historie einzelner Gestelle |
| **Simulator** | Ersatz für die Hardware, solange keine Reader hängen |

---

## Warum das Datenmodell so aussieht

Drei Tabellen tragen alles. Die Feinheiten daran sind kein Selbstzweck –
jede einzelne stammt aus einem Problem, das im Betrieb sonst auftritt.

**`gestell` – Stammdaten.**
`epc` darf `NULL` sein. Bei 400 Gestellen werden die Tags rollierend
angebracht, am besten beim Aushängen, wenn das Gestell ohnehin leer
dasteht. Wochenlang ist also nur ein Teil ausgerüstet. Ein System, das
diesen Zustand nicht aushält, ist am ersten Tag unbrauchbar. Gestelle ohne
Tag lassen sich weiterhin von Hand erfassen – sie erscheinen lediglich
ohne automatischen Standort.

**`belegung` – Auftrag ↔ Gestell, mit `von`/`bis`.**
Bewusst keine Spalte `auftrag` am Gestell: ein Gestell wird immer wieder
neu belegt und kann mehrere Aufträge gleichzeitig tragen. `bis IS NULL`
heißt „hängt gerade". Ein Auftrag gilt erst als fertig, wenn er auf
*keinem* Gestell mehr hängt – sonst ist ein über zwei Gestelle verteilter
Auftrag zu früh abgehakt.

**`event` – Standortmeldungen, nur angehängt, nie überschrieben.**
Der aktuelle Standort wird immer aus dem jüngsten Event berechnet und
nirgends gespeichert. Damit kann der Zustand nicht mit der Historie
auseinanderlaufen. Nebeneffekt: die vollständige Rückverfolgung, welches
Gestell wann in welchem Bad war, fällt gratis an.

---

## Zwei Details, an denen RFID-Projekte scheitern

**Entprellung.** Ein UHF-Reader liefert mehrere Lesungen pro Sekunde. Ohne
Filter stehen nach einer Schicht Zehntausende identischer Zeilen in der
Datenbank. `models.ENTPRELL_SEKUNDEN` fasst wiederholte Lesungen desselben
Gestells am selben Punkt zu einer Vorbeifahrt zusammen.

In der Anlage gehört dazu die passende Hardware: eine **Lichtschranke am
Lesepunkt**, die das Lesefenster auslöst. Bei 400 Gestellen sind die Puffer
voll, und eine dauernd lesende Antenne sieht fünf Gestelle gleichzeitig,
ohne sagen zu können, welches gerade durchgefahren ist. Der Trigger kostet
50–150 € pro Punkt und ist der Unterschied zwischen belastbaren Daten und
Datenmüll.

**Unbekannte EPCs.** Ein gelesener, aber keinem Gestell zugeordneter Tag
wird gemeldet, nicht stillschweigend verworfen – er ist der Hinweis auf ein
Gestell, das noch nicht in den Stammdaten steht.

---

## Liegezeiten

Jede Station trägt in `tracker/config.py` eine Soll-Verweildauer. Wird sie
um das 1,5-fache überschritten, gibt es eine Warnung, ab dem 2,5-fachen
Alarm.

Bewertet werden **nur belegte Gestelle**. Ein freies Gestell, das drei Tage
im Lager steht, ist kein Alarm, sondern der Normalzustand – sonst ersäuft
die Übersicht in Fehlalarmen und wird nach einer Woche ignoriert.

---

## Der Nebeneffekt bei 400 Gestellen

Weil die Belegung beim Einhängen geöffnet und beim Aushängen geschlossen
wird, weiß das System jederzeit, welche Gestelle **frei** sind und wo sie
zuletzt gesehen wurden (Seite *Gestellpark → Freie Gestelle*).

Bei dieser Stückzahl ist das erfahrungsgemäß fast so wertvoll wie die
Auftragsverfolgung selbst: freie Gestelle sind da, man findet sie nur nicht.

---

## Anpassen an die reale Anlage

Linien, Stationen und Soll-Dauern stehen vollständig in
**`tracker/config.py`**. Die Startwerte sind eine plausible Annahme, keine
Vermessung eurer Anlage – sie müssen im Betrieb nachjustiert werden. Die
Logik in `models.py` und `auswertung.py` bleibt davon unberührt.

---

## Wenn echte Reader kommen

`beispiel_reader.py` zeigt die Nahtstelle. Egal ob der Reader per LLRP,
MQTT, HTTP-Callback oder seriell liefert – am Ende steht ein Aufruf:

```python
models.event_von_epc(con, epc, station_id, rssi=rssi)
```

Alles dahinter bleibt unverändert. Die Seite *Simulator* fällt dann
ersatzlos weg.

---

## Anbindung an die Schaltzentrale

Aktuell werden Aufträge in der Maske von Hand angelegt. Das ist der
Platzhalter für die Anbindung. Sobald klar ist, worauf die Schaltzentrale
technisch läuft (Access, SQL Server, Excel, Weblösung), ersetzt ein
Importer `models.auftrag_anlegen()` – entweder als regelmäßiger Abgleich
oder als direkter Zugriff auf die vorhandene Datenbank.

SQLite ist bewusst gewählt: läuft ohne Server, die Datei lässt sich
kopieren und sichern. Das Schema in `tracker/db.py` ist so gehalten, dass
es sich unverändert auf SQL Server oder MySQL übertragen lässt, wenn die
Anwendung später neben der Schaltzentrale liegen soll.

---

## Aufbau

```
app.py                  Startseite
pages/                  Die weiteren Seiten
beispiel_reader.py      Nahtstelle zur echten Hardware
tracker/
  config.py             Linien, Stationen, Soll-Dauern  <- hier anpassen
  db.py                 Schema und Verbindung
  models.py             Fachlogik: Gestelle, Aufträge, Belegung, Events
  auswertung.py         Standort, Liegezeiten, Verfügbarkeit
  simulate.py           Demo-Betrieb ohne Hardware
  ui.py                 Gemeinsame Bausteine der Oberfläche
tests/                  42 Tests, davon 13 durch die Oberfläche
```

`tracker/` kennt weder Oberfläche noch Hardware – deshalb ist die Logik
sowohl von der Streamlit-App als auch von einem Reader-Dienst nutzbar,
und sie lässt sich testen.
