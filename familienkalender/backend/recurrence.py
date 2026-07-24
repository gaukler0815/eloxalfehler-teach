"""Wiederholungen: einen Termin in konkrete Einzeltermine (Occurrences)
innerhalb eines Zeitfensters auffächern."""
from datetime import datetime, timedelta

from dateutil.rrule import rrulestr

ISO_FMT = "%Y-%m-%dT%H:%M"


def parse_dt(value: str) -> datetime:
    """Akzeptiert 'YYYY-MM-DDTHH:MM' und 'YYYY-MM-DD' (all-day)."""
    value = value.strip()
    for fmt in (ISO_FMT, "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    # Letzter Versuch: ISO-Parser
    return datetime.fromisoformat(value)


def expand_event(event, window_start: datetime, window_end: datetime):
    """Liefert eine Liste von Start-datetimes des Events im Fenster
    [window_start, window_end]. Ohne rrule genau ein Vorkommen (falls im
    Fenster). Mit rrule alle Wiederholungen im Fenster."""
    start = parse_dt(event.start)

    if not event.rrule:
        if window_start <= start <= window_end:
            return [start]
        # Auch mehrtägige Einzeltermine berücksichtigen, die ins Fenster ragen
        if event.end:
            end = parse_dt(event.end)
            if start <= window_end and end >= window_start:
                return [start]
        return []

    # Wiederkehrende Termine: dtstart als Anker
    rule = rrulestr(event.rrule, dtstart=start)
    # etwas Puffer, damit auch am Fensterrand beginnende Termine erscheinen
    occurrences = rule.between(window_start - timedelta(days=1),
                               window_end + timedelta(days=1), inc=True)
    return [o for o in occurrences if window_start <= o <= window_end]


def next_occurrences(event, after: datetime, horizon: datetime):
    """Alle Starttermine zwischen `after` und `horizon` – für die
    Erinnerungsberechnung."""
    return expand_event(event, after, horizon)
