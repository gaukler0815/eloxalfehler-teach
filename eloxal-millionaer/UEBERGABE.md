# Übergabe-Info · „Wer wird Eloxal-Millionär"

Diese Datei fasst alles zusammen, damit ein neuer Chat das Spiel sofort
weiterentwickeln kann. Einfach diese ZIP in den neuen Chat laden und sagen:
*„Hier ist mein Eloxal-Quiz, bitte weiterentwickeln – lies UEBERGABE.md."*

## Was ist das?
Ein reines Browser-Spiel im Stil von „Wer wird Millionär", thematisch rund um
**Eloxieren / Anodisation / Oberflächentechnik**. Läuft komplett offline durch
Doppelklick auf `index.html` – **kein Server, kein Build, keine Abhängigkeiten**.

## Dateien in dieser ZIP
- `index.html` – das komplette Spiel (HTML + CSS + JS in einer Datei, ~546 Zeilen)
- `fragen.js` – der Fragenkatalog als globale Variable `FRAGENKATALOG`
- `README.md` – Nutzer-/Feature-Doku
- `UEBERGABE.md` – diese Datei

## Aktueller Funktionsumfang
- Jacobi-Eloxal-Design: Logo (Inline-SVG „JX / JACOBI ELOXAL"), Markenfarben,
  rotierender Spotlight-Hintergrund
- 15-stufige Gewinnleiter (50 € … 1.000.000 €), Sicherheitsstufen bei
  Stufe 5 (500 €) und Stufe 10 (16.000 €)
- 3 Joker: 50:50, Publikum (mit Balken), Telefon
- Sound per Web Audio (prozedural, keine Audiodateien): richtig/falsch/Auswahl/
  Joker/Gewinn/verloren + dezente Spannungsschleife; Ton-Schalter oben rechts
- Konfetti beim Millionengewinn
- Rangliste mit Namenseingabe am Ende jeder Partie (Sieg ODER Niederlage),
  Top 10 mit Medaillen, gespeichert in `localStorage`
- Tastatur: A–D bzw. 1–4
- Responsiv (Desktop & Handy)

## Fragenkatalog
- Aktuell **91 Fragen**: 27 leicht (s:1), 32 mittel (s:2), 32 schwer (s:3)
- Pro Runde werden 15 gezogen (5 je Stufe), lange nicht gesehene bevorzugt
  (Merker in `localStorage`), Antwortreihenfolge wird jedes Mal gemischt.
- Format je Frage:
  ```js
  {
    f: "Frage?",
    a: ["A", "B", "C", "D"],   // genau 4 Antworten
    r: 0,                       // Index der richtigen Antwort (0–3)
    e: "Erklärung nach dem Antworten.",
    k: "Kategorie",            // z.B. Grundlagen, Anodisation, Färben, Fehler …
    s: 1                        // 1 leicht | 2 mittel | 3 schwer
  }
  ```
- Neue Fragen einfach vor dem schließenden `];` in `fragen.js` anhängen.

## Wichtige Stellen im Code (index.html)
- `FRAGENKATALOG` kommt aus `fragen.js` (per `<script src="fragen.js">`)
- `GELDLEITER`, `SICHER` – Gewinnleiter & Sicherheitsstufen
- `baueFragenset()` – zieht/mischt die 15 Fragen
- `Sound` – Web-Audio-Soundmotor (Töne prozedural erzeugt)
- Rangliste: `LB_KEY`, `addScore()`, `ranglisteHTML()`, `entryBlock()`,
  `eintragen()`, `zeigeRangliste()`
- Spielablauf: `spielStart()` → `renderFrage()` → `waehle()` →
  `weiter()` / `verloren()` / `gewonnen()`

## localStorage-Schlüssel
- `eloxal_millionaer_gesehen` – wie oft jede Frage schon dran war
- `eloxal_millionaer_rangliste` – die Bestenliste (Array)
- `eloxal_millionaer_name` – zuletzt eingegebener Name
- `eloxal_sound` – "on"/"off" Tonzustand

## Git-Kontext (Ursprung)
- Repo: `gaukler0815/eloxalfehler-teach`
- Branch: `claude/eloxal-quiz-game-5gfe19`
- Ordner im Repo: `eloxal-millionaer/`

## Mögliche nächste Schritte (Ideen)
- Gemeinsame/geräteübergreifende Online-Rangliste (bräuchte kleinen Server/Cloud)
- Echte Fehlerbild-Fotos aus dem Betrieb in Fragen einbauen
- Zeitlimit-Modus, mehr Fragen, Kategorienauswahl, Druck-/Präsentationsmodus
