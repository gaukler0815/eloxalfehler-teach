# Eloxal Rebels

Physik-Puzzle im Browser nach dem Vorbild von Angry Birds, thematisch im Eloxalbetrieb
(Jacobi Eloxal GmbH). Der Spieler schießt Aluminiumteile mit einer Schleuder auf
Festungen der "Korrosionsbande".

**Das vollständige Konzept steht in `docs/game-design-bible.html`. Diese Datei ist die
verbindliche Quelle für Story, Figuren, Fähigkeiten, Materialien, Welten, Wertung und
Grafikstil. Vor größeren Aufgaben immer dort nachlesen.**

## Sprache

Code, Kommentare, Commit-Messages: Englisch.
Alle sichtbaren Texte im Spiel und in der UI: Deutsch.

## Stack

- Reines HTML/CSS/JavaScript, ES-Module, kein Build-Schritt, kein Framework.
- Physik: `matter.js`, per CDN oder als lokale Datei unter `vendor/`.
- Rendering: Canvas 2D auf einer festen Weltgröße von 1920×1080, per Skalierung
  an das Fenster angepasst. Physik rechnet immer in Weltkoordinaten, nie in Pixeln
  des Bildschirms.
- Grafik: SVG-Quellen unter `assets/`, zur Laufzeit in Sprites gerendert.
- Kein Backend für die Einzelspieler-Version. Fortschritt und Bestenliste in
  `localStorage`. Ein optionaler kleiner Node-Server für die geteilte Messe-Bestenliste
  kommt später und ist in `server/` sauber vom Spiel getrennt.

Neue Abhängigkeiten nur nach Rückfrage. Das Projekt soll durch Öffnen einer HTML-Datei
lauffähig bleiben.

## Ordnerstruktur

```
index.html            Einstieg, lädt src/main.js
src/                  Spielcode (ES-Module)
levels/               ein JSON pro Level, z. B. w1-l03.json
assets/               SVG-Quellen der Figuren und Bauteile
docs/                 Game Design Bible
editor.html           Leveleditor (ab M3)
```

## Feste Regeln aus dem Konzept

- **Kein Harteloxal.** Nur Standard-Eloxal. Die Wertung läuft über 6, 12 oder 20 µm
  Schichtdicke pro Level, maximal 20 µm.
- Acht Geschosse: Ali, Bolle, Rippi, Titania, Bubbles, Säuri, Bürsti, Lasar.
  Jedes hat genau **eine** Fähigkeit, ausgelöst durch einen Klick oder Tap im Flug.
- Fünf Baumaterialien: Schutzfolie/Kartonage, Kunststoff-Distanzstück, Rohaluminium,
  Stahlgestell, Fehlcharge-Eloxal. Jedes Material hat feste Werte für Dichte,
  Elastizität, Bruchschwelle und **Leitfähigkeit**.
- **Leitfähigkeit ist die Kernmechanik**: Trifft ein Geschoss eine Stromschiene, läuft
  ein Lichtbogen durch alle sich berührenden leitenden Bauteile. Aluminium und Stahl
  leiten, Kunststoff und Kartonage isolieren. Gegner auf leitenden Bauteilen sind
  sofort erledigt. Diese Mechanik muss in einer eigenen, testbaren Datei liegen und
  darf nicht in die Rendering-Logik gemischt werden.
- Bestenliste: Namenseingabe nach Levelabschluss, maximal 12 Zeichen, Großbuchstaben,
  kein Konto und keine Mailadresse. Top 10 sichtbar, der eigene Eintrag immer
  zusätzlich mit echtem Rang. Namensfilter ist Pflicht.

## Grafikstil

- Konturlinie überall `#17131F`, gleiche Stärke bei allen Figuren.
- Palette ist die Eloxal-Farbkarte: Alu `#C9D2DC`, Blau `#1E74DC`, Gold `#F5A81C`,
  Rot `#E33A2C`, Titan `#B4A996`, Schwarz `#17131F`.
- Figuren rund und kompakt, Festungen kantig und industriell.
- Animation über Squash & Stretch, keine Einzelbildsequenzen.
- Kein Airbrush, Volumen über zwei bis drei Farbstufen.

## Arbeitsweise

- Kleine, abgeschlossene Schritte. Nach jedem Schritt muss das Spiel im Browser
  startbar sein.
- Levelinhalte gehören in JSON, niemals hart in den Code.
- Zahlenwerte für Physik und Balancing in eine zentrale Datei `src/config.js`,
  damit sie ohne Codeänderung angepasst werden können.
- Keine großen Umbauten ohne Rückfrage.

## Meilensteine

- **M1** Schleuder, Physik, Materialverhalten, Welt 1 mit drei Leveln, Ali und Bolle.
- **M2** Alle acht Geschosse mit Fähigkeiten, Wertung in µm, Bestenliste.
- **M3** Leitfähigkeit, Leveleditor, Welten 1–4.
- **M4** Färben, Bosskampf, Ton, Prüfprotokoll am Weltende.
