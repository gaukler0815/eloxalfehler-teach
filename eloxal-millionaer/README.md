# 💡 Wer wird Eloxal-Millionär

Ein Browser-Quiz im Stil von „Wer wird Millionär" – nur mit Fragen rund um
**Eloxieren, Anodisation, Färben, Verdichten und typische Oberflächenfehler**.

## Spielen

Einfach `index.html` im Browser öffnen (Doppelklick genügt, kein Server nötig).

## Features

- **15-stufige Geldleiter** von 50 € bis 1.000.000 €
- **Sicherheitsstufen** bei 500 € und 16.000 €
- **Drei Joker:** 50:50, Publikumsjoker (👥), Telefonjoker (📞)
- **Steigende Schwierigkeit:** Fragen 1–5 leicht, 6–10 mittel, 11–15 schwer
- **Erklärung nach jeder Antwort** – man lernt beim Spielen
- Responsiv für Desktop & Handy

## Fragenkatalog – keine Wiederholungen

Alle Fragen stehen in [`fragen.js`](fragen.js). Pro Runde werden **15 Fragen
zufällig** gezogen (5 je Schwierigkeitsstufe). Damit nicht immer dieselben
Fragen einer Kategorie kommen, merkt sich das Spiel im Browser (`localStorage`),
welche Fragen schon gezeigt wurden, und bevorzugt lange nicht gesehene Fragen.
Zusätzlich wird die Reihenfolge der vier Antworten jedes Mal neu gemischt.

## Fragen ergänzen

Neue Fragen einfach in `fragen.js` anhängen:

```js
{
  f: "Deine Frage?",
  a: ["Antwort A", "Antwort B", "Antwort C", "Antwort D"],
  r: 0,                       // Index der richtigen Antwort (0–3)
  e: "Erklärung, die nach dem Antworten erscheint.",
  k: "Kategorie",            // z.B. Grundlagen, Anodisation, Färben, Fehler …
  s: 1                        // Schwierigkeit: 1 leicht, 2 mittel, 3 schwer
}
```

Je mehr Fragen pro Stufe vorhanden sind, desto größer die Abwechslung.
Aktuell umfasst der Katalog Fragen zu Grundlagen, Vorbehandlung, Anodisation,
Färben, Verdichten, Werkstoff, Chemie, Qualität und Fehlerbildern.
