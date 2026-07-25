# 🚀 Familienkalender auf Render veröffentlichen

Diese Anleitung ist für **ohne Vorkenntnisse** gedacht. Am Ende hast du eine
feste Internet-Adresse (mit `https://`), die die ganze Familie aufs Handy
holen kann – ähnlich wie deine Hochsitz-Seite bei Netlify, nur dass hier ein
richtiger Server dahinter läuft (nötig für Sync + Push-Erinnerungen).

Zeitaufwand: ca. 10 Minuten.

---

## Schritt 1 – Render-Konto anlegen

1. Gehe auf **https://render.com** und klicke **Get Started**.
2. Melde dich mit **GitHub** an (dasselbe Konto, in dem dieses Projekt liegt).
   So sieht Render dein Repository automatisch.

## Schritt 2 – Projekt als „Blueprint" starten

1. Oben rechts **New +** → **Blueprint** anklicken.
2. In der Liste das Repository **`eloxalfehler-teach`** auswählen und
   **Connect** klicken.
3. Render findet automatisch die Datei `render.yaml` und zeigt einen Dienst
   namens **familienkalender** an.
4. Render fragt nach einem Wert für **`FK_FAMILY_CODE`** – das ist euer
   **Familien-Codewort** (z. B. `Jacobi2026`). Trage hier etwas ein, das nur
   deine Familie kennt. Nur wer diesen Code kennt, kann sich später
   registrieren. (Der Code wird sicher bei Render gespeichert, **nicht** im
   öffentlichen Code.)
5. **Apply** (bzw. **Create**) klicken.

> Render baut jetzt die App. Das dauert beim ersten Mal ein paar Minuten.
> Wenn oben **„Live"** in Grün steht, ist alles fertig.

## Schritt 3 – Adresse öffnen

- Render zeigt dir eine Adresse wie
  **`https://familienkalender.onrender.com`** (deiner Name kann leicht
  abweichen).
- Öffne sie im Browser → du siehst die Anmeldeseite → **Registrieren**.

## Schritt 4 – App aufs Handy holen

- **Android (Chrome):** Adresse öffnen → Menü (⋮) → **App installieren**.
- **iPhone (Safari):** Adresse öffnen → Teilen-Symbol → **Zum Home-Bildschirm**.
  > Wichtig: Auf dem iPhone kommen Push-Nachrichten **nur**, wenn du die App
  > so installierst und **über das neue Symbol** (nicht über Safari) öffnest.
- In der App: **⚙️ → Benachrichtigungen aktivieren** und einmal
  **Test-Benachrichtigung** drücken.

## Schritt 5 – Familie einladen

Schick den anderen die Render-Adresse **und den Familien-Code**. Jeder
registriert sich einmal (Name, E-Mail, Passwort **und Code**) – alle landen
automatisch im **gleichen** Familienkalender. Wer den Code nicht hat, kommt
nicht hinein.

> **Code später ändern?** Im Render-Dashboard beim Dienst →
> **Environment → `FK_FAMILY_CODE`** anpassen und speichern. Bereits
> angemeldete Personen bleiben angemeldet; nur **neue** Registrierungen
> brauchen dann den neuen Code.

---

## 💶 Wichtig zum Tarif (bitte lesen)

In der Datei `render.yaml` ist der **Starter-Tarif** eingestellt (aktuell
ca. **7 USD/Monat**). Das hat zwei Gründe, die für einen Familienkalender
wichtig sind:

1. **Erinnerungen kommen pünktlich.** Der kostenlose Render-Tarif „schläft"
   nach 15 Minuten ohne Besuch ein – dann verschickt der Kalender in dieser
   Zeit **keine** Push-Erinnerungen. Der Starter-Tarif läuft rund um die Uhr.
2. **Daten bleiben erhalten.** Der Starter-Tarif hat einen dauerhaften
   Speicher (die 1-GB-„Disk" in der Datei). Im kostenlosen Tarif gehen
   Termine und Dateien bei jedem Neustart verloren.

👉 **Empfehlung:** Beim Starter-Tarif bleiben – für einen echten
Familienkalender ist das die zuverlässige Variante.

> Möchtest du es trotzdem zuerst **kostenlos** ausprobieren? Sag mir Bescheid,
> dann stelle ich `render.yaml` auf den Free-Tarif um. Du musst dann nur wissen:
> Erinnerungen können sich verzögern und die Daten sind nicht dauerhaft.

---

## 🌐 Optional: eigene Adresse (z. B. kalender.jacobi-eloxal.de)

Im Render-Dashboard beim Dienst → **Settings → Custom Domains** deine
Wunsch-Adresse eintragen. Render nennt dir dann einen kleinen Eintrag, den du
bei deinem Domain-Anbieter hinterlegst. HTTPS richtet Render automatisch ein.

---

## 🔄 Änderungen später

Wenn ich am Kalender etwas verbessere und es ins Repository kommt,
**aktualisiert Render die App automatisch** (`autoDeploy` ist an). Du musst
nichts weiter tun.

## 💾 Sicherung

Deine Daten liegen im dauerhaften Speicher (`/data`). Über
**Render-Dashboard → Disks** kannst du bei Bedarf Sicherungen verwalten.
