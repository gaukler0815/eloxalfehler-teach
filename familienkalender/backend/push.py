"""Web-Push-Versand über VAPID.

Erzeugt beim ersten Start automatisch ein VAPID-Schlüsselpaar und speichert
es unter data/. Der öffentliche Schlüssel wird dem Frontend bereitgestellt,
damit sich Geräte für Push registrieren können.
"""
import base64
import json
import logging

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from pywebpush import WebPushException, webpush

from config import (VAPID_CONTACT, VAPID_PRIVATE_PATH, VAPID_PUBLIC_PATH)

log = logging.getLogger("push")


def _generate_keys():
    """Erzeugt ein EC-P256-Schlüsselpaar und legt es ab."""
    private_key = ec.generate_private_key(ec.SECP256R1())

    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    VAPID_PRIVATE_PATH.write_bytes(pem)

    # Öffentlicher Schlüssel als "raw" uncompressed point, base64url (für JS)
    public_numbers = private_key.public_key().public_numbers()
    raw = b"\x04" + public_numbers.x.to_bytes(32, "big") + public_numbers.y.to_bytes(32, "big")
    pub_b64 = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    VAPID_PUBLIC_PATH.write_text(pub_b64)
    return pub_b64


def ensure_keys() -> str:
    """Stellt sicher, dass Schlüssel existieren; gibt den öffentlichen
    Schlüssel (base64url) zurück."""
    if VAPID_PRIVATE_PATH.exists() and VAPID_PUBLIC_PATH.exists():
        return VAPID_PUBLIC_PATH.read_text().strip()
    return _generate_keys()


def get_public_key() -> str:
    return ensure_keys()


def send_push(subscription_info: dict, payload: dict) -> bool:
    """Sendet eine Push-Nachricht an ein Abo. Gibt True bei Erfolg zurück.

    Wirft PushGone, wenn das Abo nicht mehr gültig ist (404/410) – der
    Aufrufer sollte es dann löschen.
    """
    ensure_keys()
    try:
        webpush(
            subscription_info=subscription_info,
            data=json.dumps(payload),
            vapid_private_key=str(VAPID_PRIVATE_PATH),
            vapid_claims={"sub": VAPID_CONTACT},
        )
        return True
    except WebPushException as exc:
        status = getattr(exc.response, "status_code", None)
        if status in (404, 410):
            raise PushGone(str(exc)) from exc
        log.warning("Push fehlgeschlagen (%s): %s", status, exc)
        return False


class PushGone(Exception):
    """Das Push-Abo existiert nicht mehr und sollte entfernt werden."""
