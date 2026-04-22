"""OpenMythos Swarm: Federated training on Mac Studios."""

__version__ = "0.1.0"

from .crypto import Ed25519Key, sign_payload, verify_signature, canonical_json, hash_object
from .manifest import (
    RoundSpec,
    WorkerSubmission,
    RoundResult,
    SignedManifest,
    TrainingConfig,
    create_round_spec,
    save_manifest_json,
    load_manifest_json,
)

__all__ = [
    "Ed25519Key",
    "sign_payload",
    "verify_signature",
    "canonical_json",
    "hash_object",
    "RoundSpec",
    "WorkerSubmission",
    "RoundResult",
    "SignedManifest",
    "TrainingConfig",
    "create_round_spec",
    "save_manifest_json",
    "load_manifest_json",
]
