# Eloxal Strike — Multiplayer auf dem Firmenserver einrichten

Der Multiplayer braucht ein kleines Server-Programm. Es ist **eine einzige
Datei ohne Abhängigkeiten** (`server/server.js`) und liefert gleichzeitig das
ganze Spiel aus — es braucht nur Node.js (Version 14 oder neuer).

## Start

```bash
cd eloxal-strike/server
node server.js              # Standard-Port 8081
node server.js --port 9000  # anderer Port
```

Danach ist alles unter **http://SERVER-IP:8081/** erreichbar:
das Spiel selbst UND der Multiplayer (WebSocket auf demselben Port —
die Mehrspieler-Seite trägt die Adresse automatisch richtig ein).

## Firewall

Den gewählten Port (Standard 8081) im Firmennetz freigeben — genauso wie
beim Dashboard. Mehr ist nicht nötig.

## Dauerhaft laufen lassen

Am einfachsten wie das Dashboard starten (z. B. als geplanter Task beim
Hochfahren, oder unter Linux):

```bash
nohup node server.js > eloxal-strike-server.log 2>&1 &
```

## So spielt ihr

1. Alle öffnen `http://SERVER-IP:8081/` im Browser.
2. „🌐 Mehrspieler" → Name eintippen → Verbinden.
3. Einer klickt „⚒ Welt aufmachen" — alle anderen sehen die Welt in der
   Liste und klicken „▶ Beitreten" (bis 8 Spieler).
4. Der Host startet die Runde: **Deathmatch, wer zuerst 15 Abschüsse hat,
   gewinnt.** Danach landen alle wieder in der Raum-Lobby für die nächste
   Runde.

## Hinweise

- Der Server simuliert kein Spiel, er vermittelt nur (Lobby + Nachrichten)
  und zählt die Abschüsse — er braucht praktisch keine Leistung.
- Wer das Spiel nur allein spielen will, braucht den Server nicht:
  Einzelspieler läuft weiter komplett ohne (GitHub Pages / Doppelklick).
- Kachel mit PIN fürs Dashboard: die Spiel-URL einfach als Kachel
  hinterlegen; ein PIN-Bildschirm kann direkt ins Spiel eingebaut werden,
  sobald gewünscht.
