# 🧠 Linneas Quiz-Abenteuer

Eine Quiz-App als **Progressive Web App**: **100 Level mit je 5 Multiple-Choice-Fragen**
(insgesamt **500 Fragen**) aus vier Wissensgebieten — für Erstleser ab ca. 7 Jahren.

Gleicher Stil wie *Linneas Dino-Abenteuer*, aber ohne Spiel-Engine: reines
HTML/CSS/JavaScript, dadurch winzig (33 kB gepackt) und blitzschnell.

---

## Wo man es spielt

* **Im Netz:** <https://gaukler0815.github.io/eloxalfehler-teach/quiz-abenteuer/>
* **Auf dem Handy/Tablet installieren:** Seite öffnen →
  *iOS/Safari:* Teilen → „Zum Home-Bildschirm" ·
  *Android/Chrome:* Menü → „App installieren".
  Danach läuft sie im Vollbild und **offline**.

## Schnellstart

```bash
cd quiz-abenteuer
npm install
npm run dev      # http://localhost:5174
npm test         # prüft alle 500 Fragen auf Vollständigkeit
npm run build    # Produktions-Build nach dist/
npm run icons    # App-Icons neu erzeugen
```

---

## Spielprinzip

* **100 Level**, jedes mit **5 Fragen** und **4 Antwortmöglichkeiten**.
* **Punkte:** 10 pro richtiger Antwort, **+20 Bonus** wenn alle fünf stimmen (max. 70).
* **Bestanden ab 4 von 5 richtigen Antworten (80 %)** — dann wird das nächste Level frei.
* **Sterne:** ⭐⭐ für 4 von 5, ⭐⭐⭐ für alle fünf. (Bei weniger als vier gibt es keinen
  Stern, weil das Level dann nicht bestanden ist.)
* **Wiederholen geht beliebig oft.** Fragen *und* Antworten werden jedes Mal neu
  gemischt — auswendig lernen funktioniert also nicht, verstehen schon.
* Nach jeder Antwort erscheint ein **Erklärsatz** („Wusstest du schon?"), auch wenn
  die Antwort richtig war.
* **9 Abzeichen** als Fernziel: je eines pro fertiger Welt, dazu 10/25/50/100 Level
  und 100 gesammelte Sterne.

## Die vier Welten

Die Level wechseln sich ab — Level 1 Dino, 2 Tiere, 3 Natur, 4 Weltraum, 5 Dino …
So kommt jede Welt 25-mal vor und es wird nie eintönig.

| Welt | Themen (25 Level je Welt) |
|---|---|
| 🦕 **Dino-Welt** | T-Rex, Pflanzen- und Fleischfresser, Panzer und Stacheln, Langhälse, Flugsaurier, Meeressaurier, Eier und Babys, Rekorde, Zähne, Fossilien, Forscher, Zeitalter, Raptoren, Hörner, Namen, Federn, Fußspuren, Fundorte, das Ende der Dinos, kleine Dinos, berühmte Funde, Sinne, Irrtümer, Finale |
| 🦊 **Tier-Welt** | Haustiere, Bauernhof, heimischer Wald, Vögel, Insekten, Bienen und Ameisen, Fluss und See, Meer, Wale, Afrika, Dschungel, Wüste, Eis und Schnee, Australien, Tierkinder, Tiergruppen, Winter und Zugvögel, Tarnung, schnell und langsam, Riesen und Zwerge, Reptilien, Amphibien, Spinnen, Sinne, Finale |
| 🌳 **Natur-Welt** | Jahreszeiten, Wetter, Wolken, Bäume, Blumen, Pilze, Wald, Wiese und Garten, Wasser, Flüsse und Meere, Berge, Vulkane, Steine, Boden, Obst und Gemüse, vom Korn zum Brot, Blätter, Umweltschutz, Recycling, Energie, Tag und Nacht, Licht und Regenbogen, Luft, Lebensräume, Finale |
| 🚀 **Weltraum** | Sonne, Mond, Erde, Merkur und Venus, Mars, Jupiter, Saturn, Uranus und Neptun, Sterne, Sternbilder, Milchstraße, Sonnensystem, Raketen, Astronauten, ISS, Mondlandung, Satelliten, Kometen, Asteroiden, Teleskope, Schwerkraft, Tag und Jahr, Mondphasen, Roboter im All, Finale |

---

## Projektstruktur

```
quiz-abenteuer/
├── index.html                 schlankes Gerüst, alles Weitere baut JavaScript
├── vite.config.js             Vite + vite-plugin-pwa (Manifest, Service Worker)
├── scripts/generate-icons.mjs erzeugt die PNG-Icons ohne Fremdpakete
├── tests/fragen.test.mjs      prüft alle 500 Fragen (npm test)
└── src/
    ├── main.js                Start, Service Worker, Debug-Zugriff
    ├── audio/sfx.js           Töne per Web Audio API, keine Sounddateien
    ├── data/
    │   ├── frage.js           Frage-Baustein (erste Antwort ist die richtige)
    │   ├── dinos.js           125 Fragen
    │   ├── tiere.js           125 Fragen
    │   ├── natur.js           125 Fragen
    │   ├── weltraum.js        125 Fragen
    │   └── welten.js          baut daraus die 100 Level
    ├── state/storage.js       Spielstand in localStorage
    ├── styles/main.css        große Schrift, dicke Flächen, Welt-Farben
    └── ui/
        ├── dom.js             Mini-Helfer statt Framework
        └── screens.js         Start, Welten, Level, Quiz, Ergebnis, Abzeichen
```

### Wie neue Fragen dazukommen

In der passenden Datei unter `src/data/` einen Eintrag ergänzen:

```js
f('Wie heißt der größte Planet?', ['Jupiter', 'Mars', 'Merkur', 'Venus'],
  'In den Jupiter passen alle anderen Planeten hinein.'),
```

**Die erste Antwort ist immer die richtige** — die App mischt beim Anzeigen. Dadurch
kann beim Schreiben kein falscher Index passieren. Danach `npm test` laufen lassen:
Der Test prüft Anzahl, doppelte Antworten, fehlende Erklärungen und Fragezeichen.

## Spielstand

Alles liegt unter dem localStorage-Schlüssel `linnea-quiz-abenteuer-v1`:
freigeschaltete Level, Sterne, Punkte, Versuche und die Ton-Einstellung.
Löschen geht über *Einstellungen → Fortschritt löschen*.
