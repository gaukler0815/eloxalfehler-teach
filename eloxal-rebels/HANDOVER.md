# Eloxal Rebels — Übergabe / Projektstand

Stand: 12.08.2026. Dieses Dokument ist die Übergabe an die nächste
Claude-Code-Sitzung (oder einen menschlichen Entwickler). Es beschreibt, was
fertig ist, wie deployt wird und welche Entscheidungen offen sind.
**Zuerst lesen: `CLAUDE.md` (verbindliche Projektregeln) und
`docs/game-design-bible.html` (das Konzept).**

## Wo das Spiel lebt

- **Live (öffentlich):** https://gaukler0815.github.io/eloxalfehler-teach/
- **GitHub-Repo:** `gaukler0815/eloxalfehler-teach`, Spiel im Ordner `eloxal-rebels/`
- **Deployment:** Push auf `main` (Pfad `eloxal-rebels/**`) → GitHub-Action
  `.github/workflows/deploy-pages.yml` → führt alle `tests/*.test.js` aus →
  veröffentlicht den Ordner auf den Branch `gh-pages` → GitHub Pages liefert
  ihn aus (Pages-Quelle ist der `gh-pages`-Branch, bereits aktiviert).
  Wichtig: Der Workflow-Token darf die Pages-Site nicht per API anlegen —
  deshalb der gh-pages-Weg, nicht der Artifact-Deploy.
- **Arbeitsweise bisher:** Feature-Branch `claude/game-from-zip-3krfej` →
  PR → Merge in `main` → Auto-Deploy. PRs #1–#5 sind gemerged.

## Was fertig ist

- **12 Level in 3 Welten** (`src/levels.js` = Laufzeit-Katalog; `levels/*.json`
  daraus generiert und per Test synchron gehalten): Tutorial → Bolle → Fässer →
  Bankschuss → Titania → Bubbles → Säuri → Kontaktstelle (Lichtbogen-Puzzle) →
  Kettenreaktion → Doppelfestung → Lasar-Bunker → **Bosskampf Baron Korrosius**
  (3 Trefferphasen, isolierender Kunststoff-Thron). Spätere Level bis 3000
  Einheiten breit, bis 7 Gegner. Jeder Sieg schaltet das nächste Level frei.
- **Alle 8 Geschosse** mit je einer Tap-Fähigkeit (`src/abilities.js`).
- **Leitfähigkeit** als reines, getestetes Modul (`src/conductivity.js`) —
  nie mit Rendering mischen (Regel aus CLAUDE.md).
- **Angry-Birds-Kamera** (`src/render.js`): Intro-Schwenk von der Festung,
  beim Zielen ganzes Level sichtbar (Zoom raus), Flug-Verfolgung (Zoom rein),
  Shake bei Treffern, Zeitlupe beim letzten Kill (`game.slowmo`).
- **Design/„Juice“:** Figuren mit Gesichtern/Blinzeln/Angst (`src/characters.js`),
  Squash & Stretch, Partikel (`src/particles.js`), WebAudio-Sound ohne Dateien
  (`src/sound.js`), Parallax-Halle mit Kran, Becken, Lichtschächten, Vignette.
- **Wertung** 6/12/20 µm, **Bestenliste** mit Pflicht-Namensfilter
  (`src/leaderboard.js`), **Spielstand-Export/-Import** als JSON-Datei (Menü).
- **Logo:** Runde **JX-Marke** + „JACOBI ELOXAL“-Schriftzug als Vektor-Nachbau.
  Drei Stellen: `index.html` (Menü-SVG + Prüfprotokoll-SVG) und
  `src/render.js` → `drawLogoSign()` (Hallenschilder). Die echte Logodatei
  lag nicht vor (Firmen-Website aus der Sandbox nicht erreichbar).
- **Leveleditor:** auf Wunsch des Nutzers **entfernt** (war editor.html) —
  nicht wieder einbauen, außer er verlangt es. Die Ordnerstruktur-Angabe in
  CLAUDE.md ist insofern überholt.

## Technik-Entscheidungen (nicht ohne Grund ändern)

- **Kein Build, keine ES-Module:** Browser blockieren `import` unter `file://`.
  Alle Dateien hängen an `window.ER` und werden in `index.html` in fester
  Reihenfolge geladen. Das Spiel MUSS per Doppelklick auf `index.html` laufen.
- **matter.js liegt lokal** unter `vendor/` (läuft offline; CDN war in der
  Entwicklungs-Sandbox blockiert).
- **Level doppelt:** eingebettet in `src/levels.js` (Laufzeit, wegen file://)
  und als `levels/*.json`. Nach Level-Änderungen JSON regenerieren:
  `node -e "const fs=require('fs');require('./src/levels.js').all().forEach(l=>fs.writeFileSync('levels/'+l.id+'.json',JSON.stringify(l,null,2)+'\n'))"`
  — der Test `levels.test.js` schlägt sonst fehl.
- **Balancing zentral** in `src/config.js` (Physik, Materialien, Geschosse,
  Gegner, Wertung).

## Tests (müssen grün bleiben, laufen auch im Deploy-Workflow)

```bash
node tests/conductivity.test.js   # Kernmechanik isoliert (7 Tests)
node tests/sim.test.js            # headless: Schuss, Bruch, Lichtbogen, Wertung (5)
node tests/levels.test.js         # 12 Level: Struktur, JSON-Sync, 300-Frames-Stabilität (5)
```

## Offene Punkte / nächste Schritte

1. **Schaltzentrale + PIN (Entscheidung des Nutzers steht aus):** Er will das
   Spiel als Kachel in seine interne „Schaltzentrale“ hängen, evtl. mit
   PIN-Schutz. Besprochene Varianten: (a) hinter bestehende Anmeldung der
   Schaltzentrale, (b) Server-Verzeichnisschutz, (c) PIN-Bildschirm im Spiel
   (Hürde, keine echte Sicherheit — PIN steht im JS). Empfehlung war (c) für
   den Zweck. Dazu gehört: **Favicon/App-Icon + Web-App-Manifest** ergänzen
   (JX-Logo oder Ali, 512×512), damit die Kachel/der Homescreen-Eintrag ein
   Icon hat.
2. **Echtes JX-Logo einbauen**, sobald der Nutzer die Datei hochlädt
   (Stellen siehe oben).
3. **M4-Reste** aus der Design Bible: Färben (Welt 6-Mechanik), Prüfprotokoll
   je Welt, ggf. echte Hallen-Sounds statt Synthese, SVG-Assets unter `assets/`.
4. **Balancing-Feedback** des Nutzers zum Schwierigkeitsgrad steht noch aus.
5. Optionaler kleiner Node-Server für die geteilte Messe-Bestenliste
   (laut CLAUDE.md sauber getrennt in `server/`).

## Sprache & Stil (verbindlich, aus CLAUDE.md)

Code/Kommentare/Commits Englisch; alle sichtbaren Spieltexte Deutsch.
Kontur `#17131F` überall, Eloxal-Farbkarte, Squash & Stretch statt
Einzelbilder, kein Airbrush. Kein Harteloxal — max. 20 µm pro Level.
