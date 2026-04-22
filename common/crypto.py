"""Cryptographic signing and verification for federated training."""

import hashlib
import hmac
import json
from typing import Any
from pathlib import Path
import base64

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


def canonical_json(obj: Any) -> bytes:
    """Return canonical JSON (sorted keys, minimal whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_payload(secret: str, payload_obj: Any) -> str:
    """HMAC-SHA256 sign a payload object."""
    msg = canonical_json(payload_obj)
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def verify_signature(secret: str, payload_obj: Any, signature: str) -> bool:
    """Verify HMAC-SHA256 signature."""
    expected = sign_payload(secret, payload_obj)
    return hmac.compare_digest(expected, signature)


class Ed25519Key:
    """Ed25519 signing key pair."""

    def __init__(self, private_key=None):
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("cryptography library not installed. Install with: pip install cryptography")
        if private_key is None:
            self.private = ed25519.Ed25519PrivateKey.generate()
        else:
            self.private = private_key
        self.public = self.private.public_key()

    def sign(self, payload_obj: Any) -> str:
        """Sign payload and return base64-encoded signature."""
        msg = canonical_json(payload_obj)
        sig_bytes = self.private.sign(msg)
        return base64.b64encode(sig_bytes).decode("utf-8")

    def verify(self, public_key_pem: str, payload_obj: Any, signature_b64: str) -> bool:
        """Verify signature with a public key (PEM format)."""
        try:
            public = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
            msg = canonical_json(payload_obj)
            sig_bytes = base64.b64decode(signature_b64.encode("utf-8"))
            public.verify(sig_bytes, msg)
            return True
        except Exception:
            return False

    @property
    def public_pem(self) -> str:
        """Export public key as PEM."""
        pem = self.public.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode("utf-8")

    def save_private_pem(self, path: str) -> None:
        """Save private key to PEM file."""
        pem = self.private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        Path(path).write_bytes(pem)

    @classmethod
    def load_private_pem(cls, path: str):
        """Load private key from PEM file."""
        pem = Path(path).read_bytes()
        private = serialization.load_pem_private_key(
            pem,
            password=None
        )
        return cls(private_key=private)


def hash_checkpoint(data: bytes) -> str:
    """SHA256 hash of checkpoint data."""
    return hashlib.sha256(data).hexdigest()


def hash_object(obj: Any) -> str:
    """SHA256 hash of canonical JSON representation."""
    return hashlib.sha256(canonical_json(obj)).hexdigest()
