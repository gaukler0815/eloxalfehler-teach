"""Hintergrund-Zeitplaner: prüft jede Minute, welche Erinnerungen fällig
sind, und verschickt Push-Nachrichten an die betroffenen Personen (sofern
sie die App nutzen, d. h. ein Push-Abo haben)."""
import asyncio
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from config import FAMILY_TZ
from database import SessionLocal
from models import (Event, EventPerson, Person, PushSubscription, Reminder,
                    SentReminder)
from push import PushGone, send_push
from recurrence import next_occurrences

log = logging.getLogger("scheduler")
TZ = ZoneInfo(FAMILY_TZ)

CHECK_INTERVAL_SECONDS = 30
# Sicherheitsfenster: verpasste Erinnerungen der letzten Minuten nachholen
LOOKBACK_MINUTES = 5


def _local_now() -> datetime:
    return datetime.now(TZ).replace(tzinfo=None)


def _recipients_for_event(db: Session, event: Event) -> list[PushSubscription]:
    """Alle Push-Abos der Personen, die der Termin betrifft. Betrifft der
    Termin niemanden explizit, geht die Erinnerung an alle Nutzer (Ersteller
    inbegriffen)."""
    links = db.query(EventPerson).filter(EventPerson.event_id == event.id).all()
    user_ids: set[int] = set()

    if links:
        person_ids = [l.person_id for l in links]
        persons = db.query(Person).filter(Person.id.in_(person_ids)).all()
        for p in persons:
            if p.user_id:
                user_ids.add(p.user_id)
    else:
        # Kein Personenbezug -> alle Nutzer informieren
        for sub in db.query(PushSubscription).all():
            user_ids.add(sub.user_id)

    if event.created_by:
        user_ids.add(event.created_by)

    if not user_ids:
        return []
    return (db.query(PushSubscription)
              .filter(PushSubscription.user_id.in_(user_ids))
              .all())


def _process_once(db: Session):
    now = _local_now()
    window_start = now - timedelta(minutes=LOOKBACK_MINUTES)
    # max. Vorlaufzeit bestimmt, wie weit wir in die Zukunft schauen müssen
    max_lead = db.query(Reminder.minutes_before).order_by(
        Reminder.minutes_before.desc()).first()
    horizon_minutes = (max_lead[0] if max_lead else 0) + LOOKBACK_MINUTES + 1
    horizon = now + timedelta(minutes=horizon_minutes)

    events = db.query(Event).all()
    for event in events:
        if not event.reminders:
            continue
        occurrences = next_occurrences(event, now - timedelta(days=1), horizon)
        for occ in occurrences:
            for reminder in event.reminders:
                fire_at = occ - timedelta(minutes=reminder.minutes_before)
                if not (window_start <= fire_at <= now):
                    continue
                occ_key = occ.strftime("%Y-%m-%dT%H:%M")
                already = (db.query(SentReminder).filter_by(
                    event_id=event.id, occurrence=occ_key,
                    minutes_before=reminder.minutes_before).first())
                if already:
                    continue
                _fire(db, event, occ, reminder.minutes_before)
                db.add(SentReminder(event_id=event.id, occurrence=occ_key,
                                    minutes_before=reminder.minutes_before))
                db.commit()


def _human_lead(minutes: int) -> str:
    if minutes <= 0:
        return "jetzt"
    if minutes % 1440 == 0:
        d = minutes // 1440
        return f"in {d} Tag{'en' if d != 1 else ''}"
    if minutes % 60 == 0:
        h = minutes // 60
        return f"in {h} Stunde{'n' if h != 1 else ''}"
    return f"in {minutes} Minuten"


def _fire(db: Session, event: Event, occ: datetime, minutes: int):
    subs = _recipients_for_event(db, event)
    when = occ.strftime("%d.%m.%Y %H:%M") if not event.all_day else occ.strftime("%d.%m.%Y")
    payload = {
        "title": f"🔔 {event.title}",
        "body": f"{_human_lead(minutes)} · {when}"
                + (f" · {event.location}" if event.location else ""),
        "url": f"/?event={event.id}",
        "eventId": event.id,
        "tag": f"event-{event.id}-{occ.strftime('%Y%m%d%H%M')}",
    }
    for sub in subs:
        info = {"endpoint": sub.endpoint,
                "keys": {"p256dh": sub.p256dh, "auth": sub.auth}}
        try:
            send_push(info, payload)
        except PushGone:
            db.delete(sub)
        except Exception as exc:  # nie den Loop abbrechen
            log.warning("Push-Fehler: %s", exc)
    log.info("Erinnerung gesendet: '%s' an %d Abo(s)", event.title, len(subs))


async def scheduler_loop():
    log.info("Erinnerungs-Zeitplaner gestartet (TZ=%s)", FAMILY_TZ)
    while True:
        try:
            db = SessionLocal()
            try:
                _process_once(db)
            finally:
                db.close()
        except Exception as exc:
            log.exception("Fehler im Zeitplaner: %s", exc)
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)
