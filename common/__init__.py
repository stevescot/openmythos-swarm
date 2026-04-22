"""OpenMythos Swarm: Federated training on Mac, NVIDIA, and AMD workers."""

__version__ = "0.2.0"

from .crypto import Ed25519Key, sign_payload, verify_signature, canonical_json, hash_object, hash_checkpoint
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
    "hash_checkpoint",
    "RoundSpec",
    "WorkerSubmission",
    "RoundResult",
    "SignedManifest",
    "TrainingConfig",
    "create_round_spec",
    "save_manifest_json",
    "load_manifest_json",
]
