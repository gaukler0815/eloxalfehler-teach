"""Familienkalender – FastAPI-Backend.

Startet die API, den Erinnerungs-Zeitplaner und liefert das PWA-Frontend aus.
"""
import asyncio
import logging
import mimetypes
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import (Depends, FastAPI, File, HTTPException, Query, UploadFile)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

import schemas
from auth import (create_token, current_user, hash_password, verify_password)
from config import (FRONTEND_DIR, MAX_UPLOAD_MB, UPLOAD_DIR)
from database import get_db, init_db
from models import (Attachment, Event, EventPerson, Person, PushSubscription,
                    Reminder, User)
from push import PushGone, get_public_key, send_push
from recurrence import expand_event, parse_dt
from scheduler import scheduler_loop

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    get_public_key()  # VAPID-Schlüssel sicherstellen
    task = asyncio.create_task(scheduler_loop())
    log.info("Familienkalender bereit.")
    yield
    task.cancel()


app = FastAPI(title="Familienkalender", lifespan=lifespan)


# =========================================================================
# Hilfsfunktionen
# =========================================================================
def event_to_out(event: Event) -> dict:
    return {
        "id": event.id,
        "title": event.title,
        "description": event.description or "",
        "location": event.location or "",
        "category": event.category,
        "color": event.color,
        "start": event.start,
        "end": event.end,
        "all_day": event.all_day,
        "rrule": event.rrule,
        "created_by": event.created_by,
        "person_ids": [l.person_id for l in event.person_links],
        "reminders": [r.minutes_before for r in event.reminders],
        "attachments": [
            {"id": a.id, "filename": a.filename,
             "content_type": a.content_type, "size": a.size}
            for a in event.attachments
        ],
    }


def _set_event_relations(db: Session, event: Event, person_ids, reminders):
    # Personen
    db.query(EventPerson).filter(EventPerson.event_id == event.id).delete()
    for pid in dict.fromkeys(person_ids):
        if db.get(Person, pid):
            db.add(EventPerson(event_id=event.id, person_id=pid))
    # Erinnerungen
    db.query(Reminder).filter(Reminder.event_id == event.id).delete()
    for minutes in dict.fromkeys(reminders):
        db.add(Reminder(event_id=event.id, minutes_before=int(minutes)))


def _sync_birthday_event(db: Session, person: Person):
    """Legt für eine Person mit Geburtsdatum einen jährlichen Geburtstags-
    Termin an bzw. aktualisiert/entfernt ihn."""
    if person.birthday:
        start = person.birthday.strftime("%Y-%m-%d")
        title = f"🎂 {person.name} Geburtstag"
        if person.birthday_event_id and db.get(Event, person.birthday_event_id):
            ev = db.get(Event, person.birthday_event_id)
            ev.title = title
            ev.start = start
        else:
            ev = Event(title=title, category="birthday", color=person.color,
                       start=start, all_day=True, rrule="FREQ=YEARLY",
                       created_by=None)
            db.add(ev)
            db.flush()
            db.add(EventPerson(event_id=ev.id, person_id=person.id))
            person.birthday_event_id = ev.id
    elif person.birthday_event_id:
        ev = db.get(Event, person.birthday_event_id)
        if ev:
            db.delete(ev)
        person.birthday_event_id = None


# =========================================================================
# Konfiguration / Auth
# =========================================================================
@app.get("/api/config")
def api_config():
    from config import FAMILY_CODE
    return {"vapid_public_key": get_public_key(),
            "family_code_required": bool(FAMILY_CODE)}


@app.post("/api/auth/register", response_model=schemas.TokenOut)
def register(data: schemas.RegisterIn, db: Session = Depends(get_db)):
    from config import FAMILY_CODE
    if FAMILY_CODE and data.family_code.strip() != FAMILY_CODE:
        raise HTTPException(403, "Falscher Familien-Code")
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(400, "E-Mail ist bereits registriert")
    user = User(name=data.name, email=str(data.email),
                password_hash=hash_password(data.password))
    db.add(user)
    db.flush()
    # Automatisch eine Person zum Konto anlegen
    person = Person(name=data.name, user_id=user.id)
    db.add(person)
    db.commit()
    db.refresh(user)
    return {"token": create_token(user.id), "user": _user_out(db, user)}


@app.post("/api/auth/login", response_model=schemas.TokenOut)
def login(data: schemas.LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == str(data.email)).first()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(401, "E-Mail oder Passwort falsch")
    return {"token": create_token(user.id), "user": _user_out(db, user)}


def _user_out(db: Session, user: User) -> dict:
    person = db.query(Person).filter(Person.user_id == user.id).first()
    return {"id": user.id, "name": user.name, "email": user.email,
            "person_id": person.id if person else None}


@app.get("/api/me", response_model=schemas.UserOut)
def me(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return _user_out(db, user)


@app.get("/api/users")
def list_users(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return [{"id": u.id, "name": u.name, "email": u.email}
            for u in db.query(User).all()]


# =========================================================================
# Personen
# =========================================================================
def _person_out(p: Person) -> dict:
    return {"id": p.id, "name": p.name, "color": p.color,
            "birthday": p.birthday, "user_id": p.user_id,
            "has_app": p.user_id is not None}


@app.get("/api/persons")
def list_persons(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return [_person_out(p) for p in db.query(Person).order_by(Person.name).all()]


@app.post("/api/persons")
def create_person(data: schemas.PersonIn, user: User = Depends(current_user),
                  db: Session = Depends(get_db)):
    person = Person(name=data.name, color=data.color, birthday=data.birthday,
                    user_id=data.user_id)
    db.add(person)
    db.flush()
    _sync_birthday_event(db, person)
    db.commit()
    db.refresh(person)
    return _person_out(person)


@app.put("/api/persons/{person_id}")
def update_person(person_id: int, data: schemas.PersonIn,
                  user: User = Depends(current_user), db: Session = Depends(get_db)):
    person = db.get(Person, person_id)
    if not person:
        raise HTTPException(404, "Person nicht gefunden")
    person.name = data.name
    person.color = data.color
    person.birthday = data.birthday
    person.user_id = data.user_id
    _sync_birthday_event(db, person)
    db.commit()
    db.refresh(person)
    return _person_out(person)


@app.delete("/api/persons/{person_id}")
def delete_person(person_id: int, user: User = Depends(current_user),
                  db: Session = Depends(get_db)):
    person = db.get(Person, person_id)
    if not person:
        raise HTTPException(404, "Person nicht gefunden")
    if person.birthday_event_id:
        ev = db.get(Event, person.birthday_event_id)
        if ev:
            db.delete(ev)
    db.query(EventPerson).filter(EventPerson.person_id == person_id).delete()
    db.delete(person)
    db.commit()
    return {"ok": True}


# =========================================================================
# Termine
# =========================================================================
@app.get("/api/events")
def list_events(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return [event_to_out(e) for e in db.query(Event).order_by(Event.start).all()]


@app.get("/api/events/{event_id}")
def get_event(event_id: int, user: User = Depends(current_user),
              db: Session = Depends(get_db)):
    event = db.get(Event, event_id)
    if not event:
        raise HTTPException(404, "Termin nicht gefunden")
    return event_to_out(event)


@app.post("/api/events")
def create_event(data: schemas.EventIn, user: User = Depends(current_user),
                 db: Session = Depends(get_db)):
    event = Event(
        title=data.title, description=data.description, location=data.location,
        category=data.category, color=data.color, start=data.start,
        end=data.end, all_day=data.all_day, rrule=data.rrule or None,
        created_by=user.id,
    )
    db.add(event)
    db.flush()
    _set_event_relations(db, event, data.person_ids, data.reminders)
    db.commit()
    db.refresh(event)
    return event_to_out(event)


@app.put("/api/events/{event_id}")
def update_event(event_id: int, data: schemas.EventIn,
                 user: User = Depends(current_user), db: Session = Depends(get_db)):
    event = db.get(Event, event_id)
    if not event:
        raise HTTPException(404, "Termin nicht gefunden")
    event.title = data.title
    event.description = data.description
    event.location = data.location
    event.category = data.category
    event.color = data.color
    event.start = data.start
    event.end = data.end
    event.all_day = data.all_day
    event.rrule = data.rrule or None
    _set_event_relations(db, event, data.person_ids, data.reminders)
    db.commit()
    db.refresh(event)
    return event_to_out(event)


@app.delete("/api/events/{event_id}")
def delete_event(event_id: int, user: User = Depends(current_user),
                 db: Session = Depends(get_db)):
    event = db.get(Event, event_id)
    if not event:
        raise HTTPException(404, "Termin nicht gefunden")
    # Falls Geburtstags-Termin: Verknüpfung an der Person lösen
    person = db.query(Person).filter(Person.birthday_event_id == event_id).first()
    if person:
        person.birthday_event_id = None
    db.delete(event)
    db.commit()
    return {"ok": True}


@app.get("/api/occurrences")
def occurrences(start: str = Query(...), end: str = Query(...),
                user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Alle Einzeltermine (inkl. aufgelöster Wiederholungen) im Zeitfenster."""
    window_start = parse_dt(start)
    window_end = parse_dt(end)
    result = []
    for event in db.query(Event).all():
        person_ids = [l.person_id for l in event.person_links]
        for occ in expand_event(event, window_start, window_end):
            occ_str = occ.strftime("%Y-%m-%dT%H:%M")
            occ_end = None
            if event.end:
                try:
                    delta = parse_dt(event.end) - parse_dt(event.start)
                    occ_end = (occ + delta).strftime("%Y-%m-%dT%H:%M")
                except Exception:
                    occ_end = None
            result.append({
                "event_id": event.id, "title": event.title,
                "location": event.location or "", "category": event.category,
                "color": event.color, "start": occ_str, "end": occ_end,
                "all_day": event.all_day, "person_ids": person_ids,
                "recurring": bool(event.rrule),
            })
    result.sort(key=lambda x: x["start"])
    return result


@app.get("/api/search")
def search(q: str = Query(..., min_length=1), user: User = Depends(current_user),
           db: Session = Depends(get_db)):
    like = f"%{q.lower()}%"
    events = db.query(Event).all()
    hits = []
    for e in events:
        blob = " ".join([e.title or "", e.description or "",
                         e.location or ""]).lower()
        if q.lower() in blob:
            hits.append(event_to_out(e))
    hits.sort(key=lambda x: x["start"])
    return hits


# =========================================================================
# Anhänge (Dokumente & Bilder)
# =========================================================================
@app.post("/api/events/{event_id}/attachments")
async def upload_attachment(event_id: int, file: UploadFile = File(...),
                            user: User = Depends(current_user),
                            db: Session = Depends(get_db)):
    event = db.get(Event, event_id)
    if not event:
        raise HTTPException(404, "Termin nicht gefunden")
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Datei zu groß (max. {MAX_UPLOAD_MB} MB)")
    ext = ""
    if "." in (file.filename or ""):
        ext = "." + file.filename.rsplit(".", 1)[1]
    stored = f"{uuid.uuid4().hex}{ext}"
    (UPLOAD_DIR / stored).write_bytes(content)
    att = Attachment(
        event_id=event_id, filename=file.filename or stored,
        stored_name=stored,
        content_type=file.content_type or "application/octet-stream",
        size=len(content), uploaded_by=user.id,
    )
    db.add(att)
    db.commit()
    db.refresh(att)
    return {"id": att.id, "filename": att.filename,
            "content_type": att.content_type, "size": att.size}


@app.get("/api/attachments/{att_id}")
def download_attachment(att_id: int, token: str = Query(default=""),
                        db: Session = Depends(get_db)):
    # Download erlaubt Token als Query-Parameter (für <a href>/<img src>)
    from auth import verify_token
    if not verify_token(token):
        raise HTTPException(401, "Nicht angemeldet")
    att = db.get(Attachment, att_id)
    if not att:
        raise HTTPException(404, "Anhang nicht gefunden")
    path = UPLOAD_DIR / att.stored_name
    if not path.exists():
        raise HTTPException(404, "Datei fehlt")
    media = att.content_type or mimetypes.guess_type(att.filename)[0] \
        or "application/octet-stream"
    return FileResponse(path, media_type=media, filename=att.filename)


@app.delete("/api/attachments/{att_id}")
def delete_attachment(att_id: int, user: User = Depends(current_user),
                      db: Session = Depends(get_db)):
    att = db.get(Attachment, att_id)
    if not att:
        raise HTTPException(404, "Anhang nicht gefunden")
    path = UPLOAD_DIR / att.stored_name
    if path.exists():
        path.unlink()
    db.delete(att)
    db.commit()
    return {"ok": True}


# =========================================================================
# Push-Benachrichtigungen
# =========================================================================
@app.post("/api/push/subscribe")
def push_subscribe(data: schemas.PushSubscriptionIn,
                   user: User = Depends(current_user), db: Session = Depends(get_db)):
    existing = db.query(PushSubscription).filter_by(endpoint=data.endpoint).first()
    if existing:
        existing.user_id = user.id
        existing.p256dh = data.p256dh
        existing.auth = data.auth
    else:
        db.add(PushSubscription(user_id=user.id, endpoint=data.endpoint,
                                p256dh=data.p256dh, auth=data.auth))
    db.commit()
    return {"ok": True}


@app.post("/api/push/unsubscribe")
def push_unsubscribe(data: schemas.PushSubscriptionIn,
                     user: User = Depends(current_user), db: Session = Depends(get_db)):
    db.query(PushSubscription).filter_by(endpoint=data.endpoint).delete()
    db.commit()
    return {"ok": True}


@app.post("/api/push/test")
def push_test(user: User = Depends(current_user), db: Session = Depends(get_db)):
    subs = db.query(PushSubscription).filter_by(user_id=user.id).all()
    if not subs:
        raise HTTPException(400, "Keine Push-Abos für dein Konto")
    payload = {"title": "🔔 Test", "body": "Push-Benachrichtigungen funktionieren!",
               "url": "/"}
    sent = 0
    for sub in subs:
        info = {"endpoint": sub.endpoint,
                "keys": {"p256dh": sub.p256dh, "auth": sub.auth}}
        try:
            if send_push(info, payload):
                sent += 1
        except PushGone:
            db.delete(sub)
    db.commit()
    return {"sent": sent}


# =========================================================================
# Frontend (PWA) ausliefern
# =========================================================================
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.exception_handler(404)
def spa_fallback(request, exc):
    # Für unbekannte Nicht-API-Pfade die App laden (Single-Page-App)
    if not request.url.path.startswith("/api"):
        index = FRONTEND_DIR / "index.html"
        if index.exists():
            return FileResponse(index)
    return JSONResponse({"detail": "Not found"}, status_code=404)
