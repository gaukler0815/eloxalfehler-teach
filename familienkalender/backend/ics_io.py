"""Kalenderdateien (.ics / iCalendar) lesen und schreiben.

- parse_ics: Termine aus einer .ics-Datei auslesen (z. B. E-Mail-Anhang).
- build_feed: den Familienkalender als .ics-Feed ausgeben (für Outlook-Abo).
- fetch_external: einen externen ICS-Link (z. B. veröffentlichter Outlook-
  Kalender) laden und im Zeitfenster auffächern.
"""
import logging
import time
import urllib.request
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from config import FAMILY_TZ
from recurrence import expand_event, parse_dt

log = logging.getLogger("ics")
TZ = ZoneInfo(FAMILY_TZ)
UTC = ZoneInfo("UTC")


# --- Lesen ---------------------------------------------------------------
def _to_local(value):
    """(date|datetime) -> (ISO-String in lokaler Zeit, all_day?)."""
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(TZ).replace(tzinfo=None)
        return value.strftime("%Y-%m-%dT%H:%M"), False
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d"), True
    return None, False


def _rrule_to_string(r):
    def g(k):
        v = r.get(k)
        return v[0] if isinstance(v, list) and v else v
    freq = g("FREQ")
    if not freq:
        return None
    parts = [f"FREQ={freq}"]
    interval = g("INTERVAL")
    if interval and int(interval) > 1:
        parts.append(f"INTERVAL={int(interval)}")
    until = g("UNTIL")
    if until is not None and hasattr(until, "strftime"):
        parts.append("UNTIL=" + until.strftime("%Y%m%dT%H%M%S"))
    count = g("COUNT")
    if count:
        parts.append(f"COUNT={int(count)}")
    byday = r.get("BYDAY")
    if byday:
        parts.append("BYDAY=" + ",".join(byday if isinstance(byday, list) else [byday]))
    return ";".join(parts)


def parse_ics(data: bytes) -> list[dict]:
    """Liste von Terminen aus einer .ics-Datei (im Termin-Format der App)."""
    from icalendar import Calendar
    cal = Calendar.from_ical(data)
    events = []
    for comp in cal.walk("VEVENT"):
        dtstart = comp.get("DTSTART")
        if not dtstart:
            continue
        start_str, all_day = _to_local(dtstart.dt)
        if not start_str:
            continue
        end_str = None
        dtend = comp.get("DTEND")
        if dtend:
            end_str, _ = _to_local(dtend.dt)
            # Ganztägig: DTEND ist in iCal exklusiv -> einen Tag zurück
            if all_day and isinstance(dtend.dt, date) and not isinstance(dtend.dt, datetime):
                end_str = (dtend.dt - timedelta(days=1)).strftime("%Y-%m-%d")
        events.append({
            "title": str(comp.get("SUMMARY", "") or "Termin"),
            "location": str(comp.get("LOCATION", "") or ""),
            "description": str(comp.get("DESCRIPTION", "") or ""),
            "start": start_str, "end": end_str, "all_day": all_day,
            "rrule": _rrule_to_string(comp.get("RRULE")) if comp.get("RRULE") else None,
            "category": "general",
        })
    return events


# --- Schreiben (Feed für Outlook-Abo) ------------------------------------
def build_feed(events) -> bytes:
    """Erzeugt einen VCALENDAR-Feed aus DB-Terminen (Event-Objekte)."""
    from icalendar import Calendar
    from icalendar import Event as IEvent
    from icalendar.prop import vRecur

    cal = Calendar()
    cal.add("prodid", "-//Familienkalender//DE//")
    cal.add("version", "2.0")
    cal.add("x-wr-calname", "Familienkalender")
    cal.add("x-wr-timezone", FAMILY_TZ)

    for e in events:
        try:
            start = parse_dt(e.start)
        except Exception:
            continue
        ie = IEvent()
        ie.add("uid", f"fk-{e.id}@familienkalender")
        ie.add("summary", e.title or "Termin")
        if e.all_day:
            ie.add("dtstart", start.date())
            end_d = parse_dt(e.end).date() if e.end else start.date()
            ie.add("dtend", end_d + timedelta(days=1))  # exklusiv
        else:
            ie.add("dtstart", start.replace(tzinfo=TZ).astimezone(UTC))
            if e.end:
                ie.add("dtend", parse_dt(e.end).replace(tzinfo=TZ).astimezone(UTC))
        if e.location:
            ie.add("location", e.location)
        if e.description:
            ie.add("description", e.description)
        if e.rrule:
            try:
                ie.add("rrule", vRecur.from_ical(e.rrule))
            except Exception:
                pass
        cal.add_component(ie)
    return cal.to_ical()


# --- Externen Kalender abonnieren (Outlook -> Familie) -------------------
_ext_cache: dict = {}
_EXT_TTL = 15 * 60  # 15 Minuten


def fetch_external(url: str, window_start: datetime, window_end: datetime) -> list[dict]:
    """Lädt einen externen ICS-Link und liefert Einzeltermine im Fenster
    (schreibgeschützt, kind='extern')."""
    url = url.strip()
    if url.startswith("webcal://"):
        url = "https://" + url[len("webcal://"):]

    now = time.time()
    cached = _ext_cache.get(url)
    if cached and now - cached[0] < _EXT_TTL:
        parsed = cached[1]
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "Familienkalender"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        parsed = parse_ics(data)
        _ext_cache[url] = (now, parsed)

    result = []
    for ev in parsed:
        shim = SimpleNamespace(start=ev["start"], end=ev["end"], rrule=ev["rrule"])
        for occ in expand_event(shim, window_start, window_end):
            occ_str = occ.strftime("%Y-%m-%dT%H:%M")
            occ_end = None
            if ev["end"]:
                try:
                    delta = parse_dt(ev["end"]) - parse_dt(ev["start"])
                    occ_end = (occ + delta).strftime("%Y-%m-%dT%H:%M")
                except Exception:
                    occ_end = None
            result.append({
                "event_id": None, "title": ev["title"],
                "location": ev["location"], "category": "extern",
                "color": "#5a6b8c", "start": occ_str, "end": occ_end,
                "all_day": ev["all_day"], "person_ids": [], "recurring": bool(ev["rrule"]),
                "source": "extern",
            })
    return result
