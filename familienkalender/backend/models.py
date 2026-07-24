"""Datenbankmodelle des Familienkalenders."""
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, Date, DateTime, ForeignKey, Integer, String, Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from database import Base

# Verknüpfungstabelle: welche Personen betrifft ein Termin?
event_persons = None  # (wir nutzen ein eigenes Modell für mehr Kontrolle)


class User(Base):
    """Ein Familienmitglied mit App-Zugang (kann sich anmelden)."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    person = relationship("Person", back_populates="user", uselist=False)
    subscriptions = relationship("PushSubscription", back_populates="user",
                                 cascade="all, delete-orphan")


class Person(Base):
    """Eine Person, die ein Termin betreffen kann.

    Kann optional mit einem User-Konto verknüpft sein – dann bekommt diese
    Person Push-Nachrichten. Personen ohne Konto (z. B. kleine Kinder) können
    trotzdem Terminen zugeordnet werden.
    """
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    color = Column(String, default="#4f7cff")
    birthday = Column(Date, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    # Auto-erzeugter Geburtstags-Termin (falls birthday gesetzt)
    birthday_event_id = Column(Integer, ForeignKey("events.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="person")
    birthday_event = relationship("Event", foreign_keys=[birthday_event_id])


class Event(Base):
    """Ein Termin im Kalender."""
    __tablename__ = "events"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text, default="")
    location = Column(String, default="")
    category = Column(String, default="general")   # general | birthday | ...
    color = Column(String, default="#4f7cff")

    # Lokale Zeit der Familie, gespeichert als ISO-String "YYYY-MM-DDTHH:MM"
    start = Column(String, nullable=False)
    end = Column(String, nullable=True)
    all_day = Column(Boolean, default=False)

    # Wiederholung als RRULE-String (iCal), z. B. "FREQ=WEEKLY;INTERVAL=1"
    rrule = Column(Text, nullable=True)

    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    reminders = relationship("Reminder", back_populates="event",
                             cascade="all, delete-orphan")
    attachments = relationship("Attachment", back_populates="event",
                               cascade="all, delete-orphan")
    person_links = relationship("EventPerson", back_populates="event",
                                cascade="all, delete-orphan")


class EventPerson(Base):
    """Zuordnung: Termin ↔ betroffene Person."""
    __tablename__ = "event_persons"
    __table_args__ = (UniqueConstraint("event_id", "person_id"),)

    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)

    event = relationship("Event", back_populates="person_links")
    person = relationship("Person")


class Reminder(Base):
    """Eine Erinnerung zu einem Termin (Vorlaufzeit in Minuten).

    Mehrere Erinnerungen pro Termin sind möglich, z. B. 2 Tage vorher UND
    2 Stunden vorher.
    """
    __tablename__ = "reminders"

    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    minutes_before = Column(Integer, nullable=False)

    event = relationship("Event", back_populates="reminders")


class Attachment(Base):
    """Hochgeladene Datei oder Bild zu einem Termin."""
    __tablename__ = "attachments"

    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    filename = Column(String, nullable=False)        # Originalname
    stored_name = Column(String, nullable=False)     # Name auf der Platte
    content_type = Column(String, default="application/octet-stream")
    size = Column(Integer, default=0)
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    event = relationship("Event", back_populates="attachments")


class PushSubscription(Base):
    """Web-Push-Abo eines Geräts eines Users (VAPID)."""
    __tablename__ = "push_subscriptions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    endpoint = Column(String, unique=True, nullable=False)
    p256dh = Column(String, nullable=False)
    auth = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="subscriptions")


class SentReminder(Base):
    """Merkt sich bereits versendete Erinnerungen, um Doppelversand zu
    verhindern (wichtig bei wiederkehrenden Terminen)."""
    __tablename__ = "sent_reminders"
    __table_args__ = (
        UniqueConstraint("event_id", "occurrence", "minutes_before"),
    )

    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    occurrence = Column(String, nullable=False)   # ISO-Startzeit der Instanz
    minutes_before = Column(Integer, nullable=False)
    sent_at = Column(DateTime, default=datetime.utcnow)
