#!/usr/bin/env bash
#
# Kamera-Wand im Vollbild (Kiosk) auf dem am Raspberry Pi angeschlossenen TV öffnen.
#
# Aufruf (eine Zeile):
#   ./kamera-wand.sh "https://security.eufylife.com/"          # eufy Web Portal
#   ./kamera-wand.sh "http://homeassistant.local:8123/lovelace/kamerawand?kiosk"
#
# Beenden: am Pi Alt+F4 drücken, oder in einem Terminal:  pkill chromium
#
set -euo pipefail

URL="${1:-}"
if [ -z "$URL" ]; then
  echo "Bitte die Adresse der Kamera-Wand angeben, z. B.:"
  echo "  $0 https://security.eufylife.com/"
  echo "  $0 http://homeassistant.local:8123/lovelace/kamerawand?kiosk"
  exit 1
fi

# Auf dem Pi ist der Bildschirm meist DISPLAY :0
export DISPLAY="${DISPLAY:-:0}"

# Bildschirmschoner und Energiesparen aus, damit der TV nicht dunkel wird
xset s off      2>/dev/null || true
xset s noblank  2>/dev/null || true
xset -dpms      2>/dev/null || true

# Mauszeiger ausblenden (falls 'unclutter' installiert ist)
command -v unclutter >/dev/null 2>&1 && unclutter -idle 1 >/dev/null 2>&1 &

# Chromium heißt je nach Pi-System chromium-browser oder chromium
BIN="$(command -v chromium-browser 2>/dev/null || command -v chromium 2>/dev/null || true)"
if [ -z "$BIN" ]; then
  echo "Chromium wurde nicht gefunden. Einmalig installieren mit:"
  echo "  sudo apt update && sudo apt install -y chromium-browser unclutter"
  exit 1
fi

exec "$BIN" \
  --kiosk \
  --incognito \
  --noerrdialogs \
  --disable-infobars \
  --disable-session-crashed-bubble \
  --check-for-update-interval=31536000 \
  --overscroll-history-navigation=0 \
  --autoplay-policy=no-user-gesture-required \
  "$URL"
