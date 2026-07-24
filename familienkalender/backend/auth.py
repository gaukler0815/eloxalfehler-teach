"""Authentifizierung: Passwort-Hashing und signierte Login-Tokens.

Bewusst nur mit der Standardbibliothek umgesetzt (PBKDF2 + HMAC), damit keine
nativen Abhängigkeiten (bcrypt-Kompilierung o. ä.) nötig sind.
"""
import base64
import hashlib
import hmac
import json
import os
import time

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from config import SECRET_KEY, TOKEN_TTL_SECONDS
from database import get_db
from models import User

_PBKDF2_ROUNDS = 200_000


# --- Passwörter -----------------------------------------------------------
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, rounds, salt_hex, hash_hex = stored.split("$")
        rounds = int(rounds)
        salt = bytes.fromhex(salt_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, rounds)
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


# --- Tokens ---------------------------------------------------------------
def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64d(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def create_token(user_id: int) -> str:
    payload = {"uid": user_id, "exp": int(time.time()) + TOKEN_TTL_SECONDS}
    body = _b64(json.dumps(payload).encode())
    sig = hmac.new(SECRET_KEY.encode(), body.encode(), hashlib.sha256).digest()
    return f"{body}.{_b64(sig)}"


def verify_token(token: str) -> int | None:
    try:
        body, sig = token.split(".")
        expected = hmac.new(SECRET_KEY.encode(), body.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64(expected), sig):
            return None
        payload = json.loads(_b64d(body))
        if payload.get("exp", 0) < int(time.time()):
            return None
        return int(payload["uid"])
    except Exception:
        return None


# --- FastAPI-Dependencies -------------------------------------------------
def current_user(authorization: str = Header(default=""),
                 db: Session = Depends(get_db)) -> User:
    token = authorization.removeprefix("Bearer ").strip()
    uid = verify_token(token) if token else None
    if not uid:
        raise HTTPException(status_code=401, detail="Nicht angemeldet")
    user = db.get(User, uid)
    if not user:
        raise HTTPException(status_code=401, detail="Benutzer nicht gefunden")
    return user
