# Eloxal Kart 🏎️

Ein 3D-Funracer im Browser nach dem Vorbild von Mario Kart – mit den Figuren
aus Eloxal Rebels. Wüsten-Canyon-Strecke, 3 Runden, 8 Fahrer, Driften mit
Mini-Turbo, Item-Boxen, Abschießen, Bots und **Multiplayer über einen
zentralen Node.js-Server**.

- **Solo-Modus:** läuft komplett im Browser gegen 7 Bots – auch statisch
  gehostet (GitHub Pages) unter `…/kart/`.
- **Online-Modus:** benötigt den mitgelieferten Node-Server (ein Prozess
  liefert Client **und** Spielserver aus).

## Schnellstart

```bash
cd eloxal-kart
npm install        # einmalig (ws für den Server)
npm start          # Server auf http://localhost:8420
```

Dann `http://localhost:8420` im Browser öffnen. Im LAN können andere über
`http://<deine-IP>:8420` mitfahren („Online fahren“). Für Rennen übers
Internet den Ordner auf einen Node-Host deployen (Render, Railway, Fly.io,
eigener VPS – Startbefehl `npm start`, Port kommt aus `PORT`).

## Steuerung

| Aktion | Tasten |
| --- | --- |
| Gas / Bremse-Rückwärts | `W`/`↑` bzw. `S`/`↓` |
| Lenken | `A`/`D` bzw. `←`/`→` |
| Driften (halten, Mini-Turbo beim Loslassen) | `Leertaste` oder `Shift` |
| Item abfeuern | `Enter`, `Strg` oder `E` |

Auf Touch-Geräten erscheinen Bildschirm-Buttons.

## Items

| Item | Wirkung |
| --- | --- |
| ⚡ Blitzbolzen | Geradeaus-Geschoss, prallt bis zu 3× von der Bande ab |
| 🎯 Zielsucher | verfolgt den Fahrer direkt vor dir entlang der Strecke |
| 🚀 Turbo | kurzer Geschwindigkeitsschub |
| 🛢️ Ölfass | wird hinter dem Kart abgelegt, dreht Verfolger |
| 🛡️ Schutzschild | blockt genau einen Treffer |

Wer hinten liegt, zieht bessere Items (Aufhol-Logik wie beim Vorbild).

## Architektur

```
eloxal-kart/
├─ server/server.js      Node: HTTP-Static + WebSocket-Spielserver
├─ public/               kompletter Client (statisch auslieferbar)
│  ├─ index.html         Menü, Lobby, HUD, Ergebnis (UI deutsch)
│  ├─ vendor/three.module.min.js   three.js lokal (läuft offline)
│  ├─ shared/            von Server UND Client genutzte Spiellogik
│  │  ├─ config.js       alle Tuning-Konstanten, Items, Figuren
│  │  ├─ track.js        Strecken-Spline, Bogenlängen, Startaufstellung
│  │  ├─ kartphysics.js  Arcade-Fahrphysik (pure Funktion)
│  │  └─ racesim.js      komplette Rennsimulation: Bots, Items,
│  │                     Geschosse, Runden, Ränge, Events
│  └─ src/               Rendering & Client-Logik (three.js)
│     ├─ game.js         Szene, Kamera, Spielschleife, Solo/Online
│     ├─ world.js        Canyon-Welt: Straße, Curbs, Mesas, Banner …
│     ├─ kartview.js     prozedurale Karts + Fahrer + Animation
│     ├─ effects.js      Partikel (Drift-Funken, Explosionen, Konfetti)
│     ├─ hud.js, input.js, audio.js, net.js, main.js
└─ tests/                Logik-Tests (node --test), laufen im CI
```

**Autoritätsmodell (Prototyp-Niveau, bewusst gewählt):** Jeder Client
simuliert sein eigenes Kart (dadurch null Eingabe-Latenz) und meldet
Position/Tempo ~15×/s. Der Server ist die alleinige Autorität für alles
Wettbewerbsrelevante: Rennphasen, Runden/Ränge, Item-Boxen, Item-Vergabe,
Geschosse, Treffer und Bots. Grobe Positionssprünge verwirft er. Für
Freundesrunden und Firmen-Events ist das genau richtig; ein vollständig
server-autoritatives Bewegungsmodell mit Client-Prediction/Rollback steht
auf der Roadmap (die Physik ist dafür schon als pure Funktion in
`shared/` isoliert).

## Tests

```bash
npm test   # Strecke, Physik, komplettes Bot-Rennen, Items, Treffer/Schild
```

## Roadmap-Ideen

- Weitere Strecken (die Strecke ist nur eine Kontrollpunktliste in
  `shared/track.js`), Streckenwahl in der Lobby.
- Höhenprofil + Sprungschanzen, Unterbodén-Boost-Felder.
- Server-autoritative Bewegung mit Client-Prediction (Anti-Cheat).
- Mehrere Räume/private Lobbys mit Beitritts-Code.
- Eigene Charakter-Modelle (GLTF) statt der prozeduralen Karts.

## Warum kein Unreal Engine?

Der Wunsch „Grafik wie Mario Kart 8, gerne Unreal“ ist verständlich – die
ehrliche Einordnung: Ein UE-Projekt kann nicht im Browser/auf GitHub Pages
laufen, braucht Gigabyte-Downloads pro Spieler, dedizierte Game-Server und
vor allem ein Team aus 3D-Artists (Modelle, Texturen, Animationen), sonst
sieht es schlechter aus als eine gut gemachte stilisierte Web-Version.
Nintendo-Figuren und -Assets sind zudem geschützt und dürfen nicht
nachgebaut werden – deshalb fährt hier der eigene Eloxal-Rebels-Cast.
Dieses Projekt liefert das gleiche Spielgefühl (Driften, Items, Positionskampf)
sofort spielbar für jeden mit einem Link.
