# 📅 Familienkalender

Ein gemeinsamer, für alle Familienmitglieder **synchronisierter** Kalender –
als installierbare **App (PWA)** für Handy, Tablet und PC. Mit Terminsuche,
Datei- und Bild-Anhängen, Wiederholungen, Geburtstagen, mehreren Ansichten,
Personen und **Push-Erinnerungen** an die betroffenen Personen.

Alle Geräte greifen auf **denselben** Kalender zu: Trägt jemand einen Termin
ein, sehen ihn sofort alle anderen.

---

## ✨ Funktionen (dein Wunschzettel – alles umgesetzt)

| Wunsch | Umgesetzt als |
|--------|---------------|
| Für die ganze Familie synchronisiert | Gemeinsamer Server – jedes Gerät zeigt denselben Kalender |
| Jeder mit der App kann Termine suchen | 🔍 Volltextsuche (Titel, Notiz, Ort) |
| Dokumente & Bilder hochladen | 📎 Anhänge pro Termin (Bilder mit Vorschau) |
| Wiederholungen einstellen | 🔁 Täglich / wöchentlich / monatlich / jährlich, Intervall + Enddatum |
| Geburtstage eintragen | 🎂 Bei einer Person hinterlegt → automatischer jährlicher Termin |
| Ansicht auswählen | Monat · Woche · Tag · Liste (umschaltbar, Standard einstellbar) |
| Push-Erinnerungen | 🔔 Web-Push-Benachrichtigungen aufs Handy |
| Vorlaufzeit einstellbar (Stunden/Tage) | Von „zum Zeitpunkt“ bis „2 Wochen vorher“ |
| Mehrere Erinnerungen pro Termin | Beliebig viele, z. B. *2 Tage* **und** *2 Stunden* vorher |
| Personen anlegen & bei Termin auswählen | 👥 Personenverwaltung, Mehrfachauswahl pro Termin |
| Push nur an ausgewählte Personen (sofern App genutzt) | Erinnerung geht an die verknüpften Konten der betroffenen Personen |

---

## 🧱 Technik (kurz)

- **Backend:** Python + FastAPI, SQLite-Datenbank, Hintergrund-Zeitplaner für
  Erinnerungen, Web-Push über VAPID.
- **Frontend:** Installierbare PWA (HTML/CSS/JS) mit Service Worker für
  Offline-Betrieb und Push-Nachrichten.
- **Keine Cloud-Abhängigkeit:** Läuft auf einem eigenen Server, einem
  kleinen VPS oder z. B. einem Raspberry Pi zu Hause.

```
familienkalender/
├── backend/      FastAPI-Server (API + Zeitplaner + Push)
├── frontend/     PWA (die eigentliche App)
├── Dockerfile    Container-Build
├── docker-compose.yml
└── start.sh      Lokaler Start ohne Docker
```

---

## 🚀 Schnellstart (lokal testen)

```bash
cd familienkalender
./start.sh
```

Dann im Browser **http://localhost:8000** öffnen, ein Konto registrieren und
loslegen. (Lokal funktioniert alles außer echten Push-Nachrichten – die
brauchen HTTPS, siehe unten.)

### Alternativ mit Docker

```bash
cd familienkalender
docker compose up --build
```

---

## 🌍 Damit die ganze Familie es nutzen kann (Produktivbetrieb)

Damit alle von überall darauf zugreifen und **Push-Nachrichten** ankommen,
muss die App unter einer **öffentlichen Adresse mit HTTPS** laufen
(Web-Push funktioniert nur über `https://`).

**Empfohlener Weg:**

1. Kleinen Server / VPS mieten (oder Dienste wie Railway, Render, Fly.io).
2. Eigene (Sub-)Domain darauf zeigen lassen, z. B. `kalender.meinefamilie.de`.
3. Mit Docker starten:
   ```bash
   docker compose up -d --build
   ```
4. Einen Reverse-Proxy mit automatischem HTTPS davorsetzen (z. B. **Caddy** –
   holt Let's-Encrypt-Zertifikate von selbst):
   ```
   kalender.meinefamilie.de {
       reverse_proxy localhost:8000
   }
   ```
5. Wichtige Umgebungsvariablen setzen (siehe unten), vor allem `FK_SECRET_KEY`.

### 📱 App aufs Handy holen

- **Android (Chrome):** Seite öffnen → Menü → „App installieren“ /
  „Zum Startbildschirm hinzufügen“.
- **iPhone/iPad (Safari):** Teilen-Symbol → „Zum Home-Bildschirm“.
  > Hinweis: Auf dem iPhone kommen Push-Nachrichten **nur**, wenn die App so
  > installiert und **von dort** (nicht aus Safari) geöffnet wurde
  > (ab iOS 16.4).
- Danach in der App unter **⚙️ Einstellungen → Benachrichtigungen aktivieren**.

---

## ⚙️ Konfiguration (Umgebungsvariablen)

| Variable | Bedeutung | Standard |
|----------|-----------|----------|
| `FK_SECRET_KEY` | Schlüssel zum Signieren der Logins – **unbedingt setzen!** | (Platzhalter) |
| `FK_FAMILY_CODE` | Codewort, das bei der Registrierung Pflicht ist (nur eure Familie kann beitreten). Leer = offene Registrierung. | (leer) |
| `FK_TIMEZONE` | Zeitzone der Familie (für Erinnerungen) | `Europe/Berlin` |
| `FK_VAPID_CONTACT` | Kontakt-E-Mail für Push-Dienste | `mailto:familie@example.com` |
| `FK_DATA_DIR` | Ablage für Datenbank, Uploads, Schlüssel | `./data` |
| `FK_MAX_UPLOAD_MB` | Maximale Dateigröße für Anhänge | `25` |
| `FK_TOKEN_TTL` | Gültigkeit eines Logins in Sekunden | `2592000` (30 Tage) |

Zufälligen Schlüssel erzeugen:
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

---

## 👨‍👩‍👧‍👦 So funktioniert's im Alltag

1. **Registrieren:** Jedes Familienmitglied, das die App nutzt, legt ein Konto
   an (mit dem **Familien-Code**, falls gesetzt). Alle teilen automatisch
   denselben Familienkalender; wer den Code nicht kennt, kann nicht beitreten.
2. **Personen anlegen** (👥): auch für Kinder ohne eigenes Handy. Wer ein
   eigenes Konto hat, kann mit seiner Person **verknüpft** werden – nur dann
   bekommt diese Person Push-Nachrichten.
3. **Termin erstellen** (＋): Titel, Zeit, Ort, Notiz, Farbe, optional
   Wiederholung, **betroffene Personen** und beliebig viele **Erinnerungen**.
   Dokumente/Bilder direkt anhängen.
4. **Erinnerung:** Zur eingestellten Vorlaufzeit bekommen genau die
   ausgewählten Personen (sofern sie die App nutzen und Push aktiviert haben)
   eine Benachrichtigung. Betrifft ein Termin niemanden konkret, wird die
   ganze Familie erinnert.

---

## 🔒 Datenschutz

Alle Daten liegen ausschließlich auf **eurem** Server (`FK_DATA_DIR`).
Es gibt keine Weitergabe an Dritte. Für die Zustellung der Push-Nachrichten
werden lediglich die (anonymen) Push-Endpunkte der Browser genutzt – das ist
technisch für Web-Push notwendig und enthält keine Kalenderinhalte im Klartext
über Dritte hinaus (die Nachricht selbst ist Ende-zu-Ende über VAPID
verschlüsselt).

---

## 🛠️ Wartung

- **Backup:** Sichere regelmäßig das Verzeichnis `FK_DATA_DIR` (enthält
  Datenbank, Uploads und VAPID-Schlüssel).
- **VAPID-Schlüssel** werden beim ersten Start automatisch erzeugt und liegen
  unter `FK_DATA_DIR`. Nicht löschen – sonst müssen alle Push-Abos neu
  eingerichtet werden.
