# Eloxal Rebels

Physik-Puzzle im Browser nach dem Vorbild von Angry Birds, thematisch im
Eloxalbetrieb (Jacobi Eloxal GmbH). Der Spieler schießt Aluminiumteile mit einer
Schleuder auf die Festungen der „Korrosionsbande“. Kernmechanik ist die
**Leitfähigkeit**: Trifft ein Geschoss eine Stromschiene, läuft ein Lichtbogen
durch alle sich berührenden leitenden Bauteile und erledigt jeden Gegner, der
darauf sitzt.

Das vollständige Konzept steht in [`docs/game-design-bible.html`](docs/game-design-bible.html);
die verbindlichen Projektregeln in [`CLAUDE.md`](CLAUDE.md).

## Starten

Kein Build, keine Installation. **`index.html` per Doppelklick im Browser
öffnen.** Fertig.

> Hinweis zum Aufbau: Damit der Doppelklick-Start (`file://`) in jedem Browser
> funktioniert, werden die Skripte als klassische `<script>`-Dateien in fester
> Reihenfolge geladen und hängen an einem gemeinsamen Namespace `window.ER`.
> ES-Module (`import`/`export`) werden von Browsern unter `file://` blockiert –
> darum dieser Weg. Der Code bleibt trotzdem in kleine, klar getrennte Dateien
> aufgeteilt. Über einen kleinen Webserver (`python3 -m http.server`) lässt sich
> das Spiel ebenfalls ausliefern; dann lädt der Levellader zusätzlich die
> JSON-Dateien direkt.

## Steuern

- **Ziehen und Loslassen** an der Schleuder (Maus oder Touch) – Zug bestimmt
  Richtung und Kraft, die gepunktete Linie zeigt die Flugbahn.
- **Klick/Tap im Flug** löst die Fähigkeit des Geschosses aus (Lasar 1× pro
  Level). `Leertaste` geht auch.
- `R` = Level neu starten, `Esc` = Weltmenü.

## Was ist umgesetzt

- **Schleuder & Physik** (matter.js) in fester Weltgröße 1920×1080, an das
  Fenster skaliert. Physik rechnet in Weltkoordinaten.
- **Fünf Materialien + Säurefass + Stromschiene** mit eigenen Werten für Dichte,
  Elastizität, Bruchschwelle und Leitfähigkeit (`src/config.js`).
- **Acht Geschosse mit je einer Fähigkeit**: Ali (Referenz), Bolle (Sturzflug/
  Durchschlag), Rippi (Streuung), Titania (Zug), Bubbles (Auftrieb), Säuri
  (Fläche), Bürsti (Schrot), Lasar (Schnitt, 1×).
- **Leitfähigkeit** als eigenes, getestetes Modul (`src/conductivity.js`), sauber
  von der Rendering-Logik getrennt.
- **Wertung in Mikrometern** (6/12/20 µm, max. 20 pro Level), Levelende-Bildschirm
  mit Prüfprotokoll.
- **Bestenliste** in `localStorage` mit Pflicht-Namensfilter (max. 12 Zeichen,
  Großbuchstaben, Sperrliste, Doppelnamen mit laufender Nummer), Top 10 plus
  eigener Eintrag mit echtem Rang, Tabs für Gesamt/Heute.
- **Spielstände**: Fortschritt (beste µm je Level) und Bestenliste werden
  automatisch im Browser gespeichert (`localStorage`) und überleben Schließen
  und Neustart. Über das Weltmenü lässt sich der Spielstand zusätzlich als
  JSON-Datei **sichern und laden** – z. B. um ihn vom Büro-PC auf den
  Messestand mitzunehmen. Beim Laden werden alle Einträge erneut durch den
  Namensfilter geprüft.
- **12 Level in drei Welten** mit ansteigender Schwierigkeit: vom
  Kartonage-Tutorial über Bankschüsse, Leitfähigkeits-Puzzles und
  Kettenreaktionen bis zum Bosskampf gegen Baron Korrosius (drei
  Trefferphasen). Spätere Level nutzen breitere Welten; die Kamera zoomt beim
  Zielen heraus und folgt dem Schuss. Jedes gewonnene Level schaltet das
  nächste frei. Level liegen als JSON unter `levels/` (generiert aus
  `src/levels.js`, dem Laufzeit-Katalog).

- **Design & „Juice“**: Figuren mit Gesichtern (blinzeln, schauen in
  Flugrichtung, erschrecken vor nahenden Geschossen), Squash & Stretch,
  Partikel, Kameraführung mit Intro-Schwenk und Zeitlupe beim letzten Treffer,
  Parallax-Halle mit Jacobi-Eloxal-Beschilderung, synthetisierter Sound.

Status der Meilensteine: **M1–M3 im Kern umgesetzt** (der Leveleditor wurde
bewusst wieder entfernt), vom **M4** stehen Bosskampf und Ton. Offen bleiben
u. a. Färben und das Prüfprotokoll je Welt.

Ein Hinweis zum Logo: Die Firmen-Beschilderung im Spiel ist als Vektor-Grafik
nachgebaut (Sechseck-Marke + Schriftzug). Liegt die echte Logodatei vor, kann
sie das Nachbau-Logo in `index.html` (SVG im Menü) und `src/render.js`
(`drawLogoSign`) ersetzen.

## Ordnerstruktur

```
index.html            Spiel-Einstieg
src/
  config.js           Palette, Materialien, Geschosse, Physik-/Balancing-Werte
  conductivity.js     Kernmechanik Lichtbogen (rein, testbar)
  levels.js           Level-Katalog + Loader (eingebettet, mit fetch-Fallback)
  leaderboard.js      Bestenliste inkl. Namensfilter (localStorage)
  abilities.js        die eine Fähigkeit je Geschoss
  render.js           Canvas-2D-Renderer (einfache Formen in der Eloxal-Palette)
  game.js             Physikwelt, Schleuder, Kollisionen, Wertung, Sieg/Niederlage
  main.js             Bootstrap, Eingabe, HUD, Menü, Endbildschirm, Bestenliste
  characters.js       Figuren-Art (Gesichter, Squash & Stretch)
  particles.js        Partikel (Staub, Splitter, Funken, Konfetti)
  sound.js            WebAudio-Soundeffekte (ohne Audiodateien)
levels/               ein JSON pro Level (l01–l12)
assets/               SVG-Quellen der Figuren (folgen)
vendor/matter.min.js  Physik-Engine, lokal (läuft offline)
docs/                 Game Design Bible
tests/                Node-Tests (ohne Framework)
```

## Tests

Zwei Test-Suiten laufen ohne Browser und ohne zusätzliche Abhängigkeiten mit
Node:

```bash
node tests/conductivity.test.js   # die Kernmechanik isoliert
node tests/sim.test.js            # ganze Runde headless: Schuss, Bruch, Lichtbogen, Wertung
node tests/levels.test.js         # alle 12 Level: Struktur, JSON-Sync, statische Stabilität
```

Der Sim-Test lädt dieselbe matter.js-Datei wie das Spiel und beweist u. a., dass
Welt 1 Level 1 mit den vorgesehenen Schüssen lösbar ist und dass der Lichtbogen
im „Kontaktstelle“-Puzzle erst zündet, wenn die leitende Kette geschlossen ist.

## Sprache

Code, Kommentare und Commit-Messages: Englisch. Alle sichtbaren Texte im Spiel:
Deutsch. (Regel aus `CLAUDE.md`.)
