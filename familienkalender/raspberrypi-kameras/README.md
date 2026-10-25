# eufy-Kameras auf dem TV (Raspberry Pi) – 4er-Ansicht

Ziel: Die 4 eufy-Kameras (Garten, Eingang + 2 weitere) **auf Abruf** als
**2×2-Ansicht** auf dem Fernseher zeigen, über einen Raspberry Pi.

> **Wichtig zu deinen Kameras (eufy SoloCam S340 = Akku/Solar):**
> Akku-Kameras senden **nicht dauerhaft** live – sie „schlafen" und wachen
> nur bei Bewegung oder auf Abruf auf. „Live beim Aufrufen" ist deshalb
> genau der richtige Modus. Ein 24/7-Dauer-Livebild aller 4 Kameras ist mit
> Akku-Kameras **nicht** sinnvoll (Akku).

Es gibt zwei Wege. **Weg A** ist in 15 Minuten fertig, **Weg B** ist die
voll integrierte Lösung (Streams auch in der Familien-App), aber ein
kleines Bastelprojekt.

---

## Weg A – Schnell: eufy Web-Portal auf dem Pi (empfohlen zum Start)

eufy hat eine eigene **Web-Ansicht**, die alle Kameras in einem Raster zeigt
(genau die 2×2-Ansicht, die du im Screenshot gesehen hast). Der Pi zeigt
einfach diese Webseite im Vollbild auf dem TV. **Keine Zusatz-Software,
kein Docker.**

**So geht's:**

1. **Web-Portal in der eufy-App freischalten**
   eufy Security App → *Ich* (unten rechts) → *Web-Portal-Zugang*
   (*Web Portal Access*) → **einschalten**.
   - Gültigkeitsdauer wählen (z. B. **24 Stunden**).
   - **Safety-PIN erzeugen** und notieren.

2. **Auf dem Pi einmalig Chromium installieren** (Terminal am Pi):
   ```bash
   sudo apt update && sudo apt install -y chromium-browser unclutter
   ```

3. **Kamera-Wand öffnen** (Datei `tv-kiosk/kamera-wand.sh` aus diesem Ordner
   auf den Pi kopieren, dann):
   ```bash
   chmod +x kamera-wand.sh
   ./kamera-wand.sh "https://security.eufylife.com/"
   ```
   Im Vollbild einmal mit **eufy-Konto + Safety-PIN** anmelden → die
   Kameras erscheinen als Raster.
   Beenden am Pi mit **Alt + F4**.

**Vorteile:** sofort fertig, offiziell von eufy, nichts zu warten.
**Nachteile:** läuft über die eufy-**Cloud**, die Anmeldung gilt nur für die
gewählte Dauer (max. 24 Std.) – danach neuer PIN / neue Anmeldung nötig.
Nicht in die Familien-App einbettbar.

---

## Weg B – Voll integriert: Home Assistant auf dem Pi

Hier laufen die Kameras **lokal** über den Pi. Vorteil: kein ständiges
Neu-Anmelden, und die Streams lassen sich **auch in der Familien-App**
(Reiter 📹 Kameras) einbinden. Nachteil: Einrichtung ist ein Nachmittag
Arbeit, und eufy-Updates können die Verbindung gelegentlich stören.

**Bausteine (alle als Home-Assistant-Add-ons / Integration):**

1. **Home Assistant OS** auf den Pi (mit *Raspberry Pi Imager* → „Home
   Assistant" auswählen, auf SD-Karte/SSD schreiben, Pi starten).
2. **Add-on „eufy Security WS"** installieren – stellt die Verbindung zum
   eufy-Konto her (meldet sich einmal mit E-Mail + Passwort + 2FA-Code an).
3. **Add-on „go2rtc"** installieren – wandelt die Kamera-Streams in ein
   Format, das der Browser/TV zeigen kann.
4. **Integration „Eufy Security" (fuatakgun)** über **HACS** installieren –
   bringt die Kameras als Geräte in Home Assistant. Für Akku-Kameras den
   Stream-Typ **P2P / „nur bei Bedarf"** verwenden.
5. **Dashboard** mit 2×2-Raster anlegen (siehe
   `home-assistant/kamerawand-dashboard.yaml`), für flüssiges Live-Bild die
   Karte **„WebRTC Camera" (AlexxIT)** aus HACS.
6. **TV-Kiosk** wie in Weg A, nur mit der Dashboard-Adresse:
   ```bash
   ./kamera-wand.sh "http://homeassistant.local:8123/lovelace/kamerawand?kiosk"
   ```

> Die genauen Namen der Add-ons/Karten können sich ändern. Aktuelle
> Schritt-für-Schritt-Hilfe: die Projektseite der Integration
> „fuatakgun/eufy_security".

**In der Familien-App:** Sobald die Streams in go2rtc laufen, liefert go2rtc
pro Kamera eine lokale Adresse (z. B. `http://<Pi-IP>:1984/...`). Diese
Adresse trägst du in der App unter **⚙️ Einstellungen → 📹 Kameras** ein –
fertig. (Hinweis: Die App läuft über `https`, der Kamera-Stream im Heimnetz
oft über `http`; die App bietet dafür den „Live öffnen"-Knopf an.)

---

## Welcher Weg für dich?

- **Erst mal sehen, ob's gefällt →** Weg A (heute Abend machbar).
- **Dauerhaft + auch in der App, ohne ständiges Anmelden →** Weg B.

Du kannst klein mit A anfangen und später auf B umstellen – der TV-Kiosk
(`kamera-wand.sh`) bleibt in beiden Fällen gleich.
