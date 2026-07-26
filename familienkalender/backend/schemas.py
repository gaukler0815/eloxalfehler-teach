"""Pydantic-Schemas für die API."""
from datetime import date
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


# --- Auth -----------------------------------------------------------------
class RegisterIn(BaseModel):
    name: str
    email: EmailStr
    password: str = Field(min_length=6)
    family_code: str = ""


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    token: str
    user: "UserOut"


class UserOut(BaseModel):
    id: int
    name: str
    email: str
    person_id: Optional[int] = None

    class Config:
        from_attributes = True


# --- Personen -------------------------------------------------------------
class PersonIn(BaseModel):
    name: str
    color: str = "#4f7cff"
    birthday: Optional[date] = None
    user_id: Optional[int] = None


class PersonOut(BaseModel):
    id: int
    name: str
    color: str
    birthday: Optional[date] = None
    user_id: Optional[int] = None
    has_app: bool = False

    class Config:
        from_attributes = True


# --- Termine --------------------------------------------------------------
class ReminderIn(BaseModel):
    minutes_before: int


class AttachmentOut(BaseModel):
    id: int
    filename: str
    content_type: str
    size: int

    class Config:
        from_attributes = True


class EventIn(BaseModel):
    title: str
    description: str = ""
    private_note: str = ""          # nur für den jeweiligen Nutzer sichtbar
    location: str = ""
    category: str = "general"
    color: str = "#4f7cff"
    start: str                     # "YYYY-MM-DDTHH:MM" oder "YYYY-MM-DD"
    end: Optional[str] = None
    all_day: bool = False
    rrule: Optional[str] = None
    person_ids: list[int] = []
    reminders: list[int] = []      # Vorlaufzeiten in Minuten


class EventOut(BaseModel):
    id: int
    title: str
    description: str
    location: str
    category: str
    color: str
    start: str
    end: Optional[str]
    all_day: bool
    rrule: Optional[str]
    created_by: Optional[int]
    person_ids: list[int]
    reminders: list[int]
    attachments: list[AttachmentOut]

    class Config:
        from_attributes = True


class OccurrenceOut(BaseModel):
    """Ein konkretes Vorkommen eines (evtl. wiederkehrenden) Termins."""
    event_id: int
    title: str
    location: str
    category: str
    color: str
    start: str
    end: Optional[str]
    all_day: bool
    person_ids: list[int]
    recurring: bool


# --- Push -----------------------------------------------------------------
class PushSubscriptionIn(BaseModel):
    endpoint: str
    p256dh: str
    auth: str


# --- Feiertage / Schulferien ---------------------------------------------
class HolidaySettingsIn(BaseModel):
    state: str = ""            # Bundesland-Code, z. B. "BY"; leer = aus
    public_holidays: bool = False
    school_holidays: bool = False


class SubscriptionIn(BaseModel):
    url: str = ""             # externer ICS-Link (z. B. Outlook), leer = aus


TokenOut.model_rebuild()
