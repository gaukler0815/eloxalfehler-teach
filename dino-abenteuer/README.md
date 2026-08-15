# 🦖 Linneas Dino-Abenteuer

Ein 2D-Jump-and-Run als **Progressive Web App** — mit 30 Leveln, 30 Dinos und
nach jedem Level einer Lerneinheit (Lesen bzw. Mathe) für Erstleser.

Gebaut mit **Vite** + **Phaser 3** (Arcade Physics). Alle Grafiken werden zur
Laufzeit auf Canvas erzeugt — es gibt **keine externen Bilddateien**, das Spiel
ist nach `npm install` sofort startklar.

---

## Schnellstart

```bash
cd dino-abenteuer
npm install
npm run dev      # Entwicklungsserver auf http://localhost:5173
```

Weitere Skripte:

```bash
npm run build    # Produktions-Build nach dist/ (inkl. Service Worker + Manifest)
npm run preview  # Build lokal testen (PWA/Offline funktioniert nur hier, nicht im dev-Modus)
npm run icons    # PWA-Icons neu erzeugen (public/icons/, reines Node-Skript ohne Abhängigkeiten)
```

### Auf dem Tablet/Handy installieren

1. `npm run build && npm run preview -- --host` (oder `dist/` auf einen HTTPS-Server legen)
2. Seite im Browser öffnen
3. **iOS/Safari:** Teilen → „Zum Home-Bildschirm" · **Android/Chrome:** Menü → „App installieren"

Danach startet das Spiel im Vollbild (`display: standalone`) und läuft dank
Service Worker auch **offline**.

---

## Steuerung

| Aktion | Tastatur | Touch |
|---|---|---|
| Laufen | ← → / A D | Joystick links |
| Springen | Leertaste / ↑ / W | Knopf „Springen" |
| Klettern & Schwimmen | ↑ ↓ / W S | Joystick hoch/runter |
| Spezialfähigkeit | Umschalt / E | Knopf „Kraft" |
| Pause | P / Esc | ⏸ oben rechts |

Die Touch-Steuerung erscheint automatisch auf Touch-Geräten und lässt sich in
den Einstellungen fest ein- oder ausschalten.

Kinderfreundliche Extras: **Coyote-Time** (kurz nach der Kante darf man noch
springen), **Sprung-Puffer** (zu früh gedrückt zählt trotzdem), 3 Herzen und
Checkpoints statt „Game Over".

---

## Spielaufbau

### 30 Level in 4 Biomen

| Level | Biom | Besonderheiten |
|---|---|---|
| 1–8 | Urzeit-Dschungel | Plattformen, Lianen, Kletterwände |
| 9–15 | Sumpf & Mangroven | Wasserbereiche zum Schwimmen, bewegliche Treibhölzer |
| 16–22 | Kristallhöhlen & Felsen | Sprungfedern, bröckelnde Plattformen, Felsblöcke |
| 23–30 | Vulkanland | Feuerbälle, Rauch-Partikel, Hebebühnen |

In jedem Level: **Eier** und **Früchte** als Punktesammler, freundliche
Hindernis-Gegner (von oben plätten, seitlich kostet ein Herz), **Checkpoints**
und ein Zielnest.

**Sterne:** 1 ⭐ fürs Durchkommen · 2 ⭐ wenn alle Eier gefunden sind ·
3 ⭐ wenn du in der Zielzeit bleibst.

### 30 Dinos

Start mit Baby-T-Rex **Rexi**. Nach jedem Level mit *bestandener* Lerneinheit
kommt genau ein neuer Dino dazu. Jeder Dino hat eigene Werte
(Geschwindigkeit, Sprungkraft, Schwimmtempo) und eine Spezialfähigkeit:

| Fähigkeit | Dinos (Beispiele) | Wirkung |
|---|---|---|
| 💥 Felsen-Schmettern | T-Rex, Allosaurus, Giganotosaurus | zertrümmert Felsblöcke |
| 🪂 Gleitflug | Pteranodon, Archaeopteryx, Microraptor | Doppelsprung + langsames Gleiten |
| 🐗 Ramm-Sprint | Triceratops, Pachycephalosaurus, Carnotaurus | Stoß nach vorn, bricht Felsen, schubst Gegner weg |
| 🌊 Schwimm-Turbo | Spinosaurus, Plesiosaurus, Mosasaurus | sehr schnell unter Wasser |
| 🛡️ Panzerschild | Ankylosaurus, Stegosaurus, Kentrosaurus | 5 Sekunden unverwundbar |
| ⚡ Flitzer / 🦘 Superhüpfer | Velociraptor, Brachiosaurus, Parasaurolophus | Turbo-Spurt bzw. extra hoher Sprung |

> **Hinweis zum Roster:** Die Liste enthält 30 Dinos plus einen versteckten
> Bonus-Dino („Aurora"), damit wirklich **jedes** der 30 Level mit genau einem
> neuen Dino belohnt wird — auch das letzte.

### Minispiele

* **🦴 Fossilien-Ausgrabung** – Sand wegpinseln, danach die Knochen per
  Drag & Drop auf ihre Schatten ziehen (3 Skelette).
* **🎨 Dino-Malbuch** – 4 Motive aus einfachen Farbflächen, per Klick/Touch
  ausmalen; die Bilder bleiben gespeichert.

---

## Lernsystem (30 Einheiten)

Nach jedem Level erscheint ein Pop-up mit persönlicher Ansprache an Linnea.
**Das nächste Level wird erst freigeschaltet, wenn die Aufgabe bestanden ist** —
Wiederholen ist beliebig oft möglich, ohne Strafe.

### Ungerade Level (1, 3, … 29) → 15 Lese-Einheiten

* Große Erstleser-Schrift (25 px, Zeilenabstand 1,95 – per Knopf noch größer)
* 10 Texte über Dinosaurier, 5 über Tiere & Natur, je ca. 5–8 Minuten Lesezeit
* Danach 5 Multiple-Choice-Fragen mit je 4 Antworten und Tipp bei Fehlern
* **Bestanden ab 4 von 5 richtigen Antworten (80 %)**

Themen: T-Rex · Triceratops · Honigbiene · Brachiosaurus · Stegosaurus · Igel ·
Velociraptor · Flugsaurier · Der Wald · Spinosaurus · Ankylosaurus · Wale ·
Dino-Eier & Nester · Zugvögel · Wie ein Fossil entsteht

### Gerade Level (2, 4, … 30) → 15 Mathe-Einheiten

Zahlenraum bis 20, je 6–10 Aufgaben mit großem Ziffernfeld (auch per Tastatur):
Plus/Minus bis 10 und bis 20, Verdoppeln, Halbieren, Lückenaufgaben (Lücke vorn
oder hinten), Zehnerübergang, drei Zahlen, Sachaufgaben im Dino-Gewand.
**Bestanden ab 80 % richtig.** Die Aufgaben werden bei jedem Versuch neu
gewürfelt — auswendig lernen geht also nicht.

---

## Projektstruktur

```
dino-abenteuer/
├── index.html               HTML-Gerüst mit Overlay-Ebenen (HUD, Touch, Dialoge)
├── vite.config.js           Vite + vite-plugin-pwa (Manifest, Service Worker)
├── scripts/generate-icons.mjs  erzeugt die PNG-Icons ohne Fremdpakete
├── public/icons/            fertige PWA-Icons (PNG + SVG)
└── src/
    ├── main.js              Phaser-Konfiguration, Scene-Liste, SW-Registrierung
    ├── audio/sfx.js         Töne per Web Audio API (keine Sounddateien)
    ├── data/
    │   ├── dinos.js         30 Dinos + Bonus-Dino, Werte & Fähigkeiten
    │   └── levels.js        30 Level-Konfigurationen, 4 Biome mit Farbpaletten
    ├── entities/
    │   ├── Player.js        Steuerung, Fähigkeiten, Klettern, Schwimmen, Herzen
    │   ├── Enemy.js         patrouillierende Gegner
    │   └── Collectible.js   Eier & Früchte
    ├── game/levelGenerator.js  Seed-basierter Aufbau der Level-Geometrie
    ├── gfx/
    │   ├── dinoArt.js       zeichnet alle Dinos (4 Frames) auf Canvas
    │   ├── textures.js      alle übrigen Texturen (Boden, Gegner, Items …)
    │   └── kulisse.js       Parallax-Hintergrund
    ├── learning/
    │   ├── readingData.js   15 Lesetexte mit je 5 Fragen
    │   ├── mathData.js      15 Mathe-Einheiten (Aufgaben-Generatoren)
    │   └── QuizController.js  Ablauf & 80-%-Regel
    ├── scenes/
    │   ├── BootScene.js     erzeugt Texturen und Animationen
    │   ├── MenuScene.js     Hauptmenü, Dino-Sammlung, Einstellungen
    │   ├── LevelSelectScene.js  30 Level-Kacheln nach Biom
    │   ├── DinoSelectScene.js   Dino-Auswahl vor dem Level
    │   ├── GameScene.js     das Jump-and-Run
    │   └── MinigameScene.js Fossilien-Ausgrabung
    ├── state/storage.js     Spielstand in localStorage
    ├── styles/main.css      große Schrift, dicke Buttons, dunkles/helles Overlay
    └── ui/
        ├── Dialog.js        DOM-Helfer und Kartensystem
        ├── Hud.js           Punkte, Eier, Herzen, Zeit
        ├── TouchControls.js virtueller Joystick + Knöpfe
        └── ColoringBook.js  Dino-Malbuch
```

---

## Spielstand

Alles liegt unter dem localStorage-Schlüssel `linnea-dino-abenteuer-v1`:
freigeschaltete Level und Dinos, Highscores und Sterne pro Level, bestandene
Lerneinheiten samt Statistik, ausgemalte Bilder, gefundene Fossilien und die
Einstellungen. Löschen geht über *Fortschritt & Einstellungen → Spielstand löschen*.

Ist localStorage nicht verfügbar (privater Modus), läuft das Spiel trotzdem —
der Fortschritt gilt dann nur für die aktuelle Sitzung.
