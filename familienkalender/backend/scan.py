"""KI-Terminerkennung: Termine aus einem Foto/Screenshot auslesen.

Schickt das Bild an die Claude-API (Vision) und lässt die Termine als
strukturiertes JSON zurückgeben. Erkennt z. B. Terminkarten vom Arzt,
Einladungen, E-Mail-Screenshots usw. – auch mehrere Termine pro Bild.
"""
import base64
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from config import FAMILY_TZ, SCAN_MODEL

log = logging.getLogger("scan")

WEEKDAYS_DE = ["Montag", "Dienstag", "Mittwoch", "Donnerstag",
               "Freitag", "Samstag", "Sonntag"]

# JSON-Schema für die strukturierte Antwort (garantiert gültiges JSON)
SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string"},        # YYYY-MM-DD oder ""
                    "start_time": {"type": "string"},  # HH:MM oder ""
                    "end_time": {"type": "string"},    # HH:MM oder ""
                    "all_day": {"type": "boolean"},
                    "location": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["title", "date", "start_time", "end_time",
                             "all_day", "location", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["events"],
    "additionalProperties": False,
}


def _prompt(today: datetime) -> str:
    weekday = WEEKDAYS_DE[today.weekday()]
    return (
        f"Heute ist {weekday}, der {today.strftime('%d.%m.%Y')} "
        f"(Zeitzone {FAMILY_TZ}).\n\n"
        "Lies ALLE Termine aus diesem Bild aus (Terminkarte, Einladung, "
        "Brief, E-Mail-Screenshot o. Ä.). Es können mehrere Termine enthalten "
        "sein – gib jeden einzeln zurück.\n\n"
        "Regeln:\n"
        "- date: Datum als YYYY-MM-DD. Relative Angaben ('nächsten Montag') "
        "anhand des heutigen Datums auflösen. Fehlt die Jahreszahl, nimm das "
        "nächste zukünftige Vorkommen. Ist kein Datum erkennbar, leer lassen.\n"
        "- start_time / end_time: Uhrzeit als HH:MM (24h). Fehlt sie, leer "
        "lassen.\n"
        "- all_day: true, wenn es ein ganztägiger Termin ohne Uhrzeit ist.\n"
        "- title: kurzer, sprechender Titel (z. B. 'Zahnarzt Kontrolle').\n"
        "- location: Ort/Praxis/Adresse, falls vorhanden, sonst leer.\n"
        "- description: weitere nützliche Infos (Telefonnummer, Hinweise wie "
        "'Versichertenkarte mitbringen'), sonst leer.\n"
        "- Erfinde nichts. Nur wirklich im Bild vorhandene Termine.\n"
        "Antworte ausschließlich im vorgegebenen JSON-Format."
    )


def extract_events(image_bytes: bytes, media_type: str) -> list[dict]:
    """Ruft Claude auf und liefert die Rohliste erkannter Termine."""
    import anthropic  # lokaler Import – nur nötig, wenn Scan aktiv ist

    client = anthropic.Anthropic()  # liest ANTHROPIC_API_KEY aus der Umgebung
    today = datetime.now(ZoneInfo(FAMILY_TZ))
    b64 = base64.standard_b64encode(image_bytes).decode()

    resp = client.messages.create(
        model=SCAN_MODEL,
        max_tokens=8000,
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": media_type, "data": b64}},
                {"type": "text", "text": _prompt(today)},
            ],
        }],
    )
    if getattr(resp, "stop_reason", None) == "refusal":
        log.warning("Scan abgelehnt (refusal)")
        return []
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    data = json.loads(text)
    return data.get("events", [])
