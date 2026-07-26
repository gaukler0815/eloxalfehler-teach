"""Feiertage (offline berechnet) und Schulferien (über openHolidays-API)
für die deutschen Bundesländer."""
import json
import logging
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta

log = logging.getLogger("holidays")

# Bundesländer: Code -> Name
STATES = {
    "BW": "Baden-Württemberg", "BY": "Bayern", "BE": "Berlin",
    "BB": "Brandenburg", "HB": "Bremen", "HH": "Hamburg", "HE": "Hessen",
    "MV": "Mecklenburg-Vorpommern", "NI": "Niedersachsen",
    "NW": "Nordrhein-Westfalen", "RP": "Rheinland-Pfalz", "SL": "Saarland",
    "SN": "Sachsen", "ST": "Sachsen-Anhalt", "SH": "Schleswig-Holstein",
    "TH": "Thüringen",
}


def _easter(year: int) -> date:
    """Ostersonntag nach der anonymen gregorianischen Formel."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def public_holidays(state: str, year: int) -> list[tuple[date, str]]:
    """Gesetzliche Feiertage eines Bundeslandes für ein Jahr."""
    E = _easter(year)
    out: list[tuple[date, str]] = []

    def add(d: date, n: str):
        out.append((d, n))

    # bundesweit
    add(date(year, 1, 1), "Neujahr")
    add(E - timedelta(days=2), "Karfreitag")
    add(E + timedelta(days=1), "Ostermontag")
    add(date(year, 5, 1), "Tag der Arbeit")
    add(E + timedelta(days=39), "Christi Himmelfahrt")
    add(E + timedelta(days=50), "Pfingstmontag")
    add(date(year, 10, 3), "Tag der Deutschen Einheit")
    add(date(year, 12, 25), "1. Weihnachtstag")
    add(date(year, 12, 26), "2. Weihnachtstag")

    # länderspezifisch
    if state == "BB":
        add(E, "Ostersonntag")
        add(E + timedelta(days=49), "Pfingstsonntag")
    if state in ("BW", "BY", "ST"):
        add(date(year, 1, 6), "Heilige Drei Könige")
    if state == "BE" and year >= 2019:
        add(date(year, 3, 8), "Internationaler Frauentag")
    if state == "MV" and year >= 2023:
        add(date(year, 3, 8), "Internationaler Frauentag")
    if state in ("BW", "BY", "HE", "NW", "RP", "SL"):
        add(E + timedelta(days=60), "Fronleichnam")
    if state == "SL":
        add(date(year, 8, 15), "Mariä Himmelfahrt")
    if state == "TH" and year >= 2019:
        add(date(year, 9, 20), "Weltkindertag")
    if state in ("BB", "HB", "HH", "MV", "NI", "SN", "ST", "SH", "TH"):
        add(date(year, 10, 31), "Reformationstag")
    if state in ("BW", "BY", "NW", "RP", "SL"):
        add(date(year, 11, 1), "Allerheiligen")
    if state == "SN":  # Buß- und Bettag = Mittwoch zwischen 16. und 22.11.
        d = date(year, 11, 22)
        while d.weekday() != 2:
            d -= timedelta(days=1)
        add(d, "Buß- und Bettag")

    return sorted(out)


# --- Schulferien (openHolidays-API, mit einfachem Cache) ------------------
_cache: dict = {}
_CACHE_TTL = 6 * 3600  # 6 Stunden


def school_holidays(state: str, start: str, end: str) -> list[dict]:
    """Schulferien-Zeiträume eines Bundeslandes im Fenster [start, end]
    (ISO-Daten YYYY-MM-DD). Ergebnis: [{name, start, end}]."""
    key = (state, start, end)
    now = time.time()
    cached = _cache.get(key)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    params = urllib.parse.urlencode({
        "countryIsoCode": "DE",
        "subdivisionCode": f"DE-{state}",
        "languageIsoCode": "DE",
        "validFrom": start,
        "validTo": end,
    })
    url = "https://openholidaysapi.org/SchoolHolidays?" + params
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.load(resp)

    result = []
    for h in data:
        name = ""
        for n in h.get("name", []):
            if n.get("language") == "DE":
                name = n.get("text")
                break
        result.append({
            "name": name or "Ferien",
            "start": h.get("startDate"),
            "end": h.get("endDate"),
        })
    _cache[key] = (now, result)
    return result
