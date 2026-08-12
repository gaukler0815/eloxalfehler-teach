# Eloxal Strike

3D-Ego-Shooter im Browser, thematisch im Eloxalbetrieb (Jacobi Eloxal GmbH).
Die Korrosionsbande ist in die Halle eingebrochen — der Spieler verteidigt sie
Welle für Welle in der First-Person-Perspektive.

**Spielen:** https://gaukler0815.github.io/eloxalfehler-teach/strike/
(oder lokal einfach `index.html` doppelklicken — kein Build, kein Server nötig.)

## Features

- **Eine große 3D-Welt:** die Eloxal-Halle mit glühenden Elektrolyt-Becken,
  Warengestellen voller Aluminiumteile, Kränen, Kisten und
  JACOBI-ELOXAL-Beschilderung (JX-Logo als Vektor-/Canvas-Nachbau, wie in
  Eloxal Rebels).
- **4 Schwierigkeitsgrade:** Azubi, Geselle, Schichtleiter, Korrosius-Modus
  (härtere Gegner, mehr Punkte, im Albtraum-Modus keine Regeneration).
- **3 Waffen:** Anodisierer MK-1 (Pistole, unendliche Reserve), Säure-Streuer
  (Schrotflinte), Lichtbogen-LMG (Vollautomatik) — mit Nachladen, Rückstoß,
  Mündungsfeuer, Tracern und Schadensabfall über Distanz.
- **4 Gegnertypen:** Rostling (schnell), Blatterbrocken (Tank), Säure-Sprüher
  (Fernkampf mit Säuregeschossen) und alle 5 Wellen **Baron Korrosius** als Boss.
- **Endloses Wellen-System** mit Verschnaufpausen, Gesundheits- und
  Munitions-Pickups, Kopftreffer-Bonus, Punktewertung und Rekord je
  Schwierigkeitsgrad (localStorage).
- **Sound komplett synthetisiert** (WebAudio, keine Audiodateien), abschaltbar.
- **AAA-Grafikkette** (im Menü abschaltbar für schwache Rechner): Bloom,
  SSAO-Umgebungsverdeckung, spiegelnder Industrieboden (Realtime-Reflector),
  Farb-Grading mit Filmkorn/Vignette/Chromatic Aberration, FXAA,
  ACES-Tone-Mapping, Environment-Reflexionen, Lichtschächte, Funkenflug,
  Zerplatz-Effekte mit Physik-Brocken und bleibenden Rostflecken.

- **Multiplayer (Deathmatch):** Welt aufmachen, Kollegen joinen über die
  Raumliste, bis 8 Spieler, erster mit 15 Abschüssen gewinnt — mit
  Spielerfiguren, Namensschildern, Killfeed, Scoreboard und Respawn.
  Benötigt den kleinen Server aus `server/` (siehe `DEPLOY.md`);
  der Einzelspieler läuft weiterhin komplett ohne Server.

## Steuerung

| Taste | Aktion |
| --- | --- |
| W/A/S/D | Bewegen |
| Maus | Zielen & Schießen |
| Shift | Sprinten |
| Leertaste | Springen |
| R | Nachladen |
| 1/2/3 oder Mausrad | Waffe wechseln |
| Esc | Pause |

## Technik

- Reines HTML/CSS/JS ohne Build-Schritt; alle Module hängen an `window.ES`
  und werden in `index.html` in fester Reihenfolge geladen (läuft unter
  `file://`, gleiche Regel wie bei Eloxal Rebels).
- three.js r134 (letzte UMD-Version) lokal unter `vendor/` — läuft offline.
- Balancing zentral in `src/config.js` (reine Daten + reine Funktionen,
  direkt per Node testbar).
- Sichtbare Spieltexte Deutsch, Code/Kommentare Englisch.

## Dateien

```
index.html          Menü, HUD, Overlays, lädt alle Skripte
src/config.js       Balancing: Schwierigkeitsgrade, Waffen, Gegner, Wellen
src/sound.js        synthetisierter WebAudio-Sound
src/world.js        Aufbau der Halle + Kollisionslogik
src/enemies.js      Gegner-Meshes, KI, Säuregeschosse
src/weapons.js      Waffenmodelle, Munition, Rückstoß, Mündungsfeuer
src/main.js         Spielschleife, Spieler, Wellen, Treffer, HUD
tests/config.test.js  Node-Tests fürs Balancing (laufen im Deploy-Workflow)
vendor/three.min.js three.js r134 (UMD)
```

## Tests

```bash
node tests/config.test.js
```
