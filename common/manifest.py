"""Federated training round manifests and checkpoint management."""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import json


@dataclass
class TrainingConfig:
    """Round training configuration."""
    seq_len: int
    micro_batch: int
    grad_accum: int
    train_loops: int
    learning_rate: float
    weight_decay: float
    target_tokens: int


@dataclass
class RoundSpec:
    """Federated training round specification."""
    round_id: int
    version: str  # e.g., "10b-mps-v1"
    config: TrainingConfig
    dataset_shard: str  # e.g., "fineweb-edu/sample-10BT"
    worker_count: int  # expected number of contributors
    prior_checkpoint_hash: str  # SHA256 of previous global checkpoint
    timestamp: str  # ISO8601
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {
            "round_id": self.round_id,
            "version": self.version,
            "config": asdict(self.config),
            "dataset_shard": self.dataset_shard,
            "worker_count": self.worker_count,
            "prior_checkpoint_hash": self.prior_checkpoint_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class WorkerSubmission:
    """Worker's contribution for a round."""
    round_id: int
    worker_id: str
    steps_completed: int
    final_loss: float
    gradient_norm: float
    delta_file: str  # path or URL to delta bytes
    delta_hash: str  # SHA256 hash
    timestamp: str  # ISO8601
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {
            "round_id": self.round_id,
            "worker_id": self.worker_id,
            "steps_completed": self.steps_completed,
            "final_loss": final_loss,
            "gradient_norm": self.gradient_norm,
            "delta_file": self.delta_file,
            "delta_hash": self.delta_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class RoundResult:
    """Master-aggregated round result."""
    round_id: int
    status: str  # "success" | "failed"
    worker_submissions: int
    valid_submissions: int
    aggregation_method: str  # "fedavg" | "trimmed_mean" | etc.
    global_checkpoint_hash: str  # SHA256 of aggregated weights
    global_checkpoint_url: str  # path to checkpoint
    timestamp: str  # ISO8601
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SignedManifest:
    """A signed round spec or result."""
    payload: Dict[str, Any]  # RoundSpec/RoundResult as dict
    signature: str  # Ed25519 signature, base64-encoded
    signer_public_key: str  # PEM
    manifest_hash: str  # SHA256 of payload


def create_round_spec(
    round_id: int,
    version: str,
    config: TrainingConfig,
    dataset_shard: str,
    worker_count: int,
    prior_checkpoint_hash: str,
    metadata: Optional[Dict] = None,
) -> RoundSpec:
    """Create a new round specification."""
    return RoundSpec(
        round_id=round_id,
        version=version,
        config=config,
        dataset_shard=dataset_shard,
        worker_count=worker_count,
        prior_checkpoint_hash=prior_checkpoint_hash,
        timestamp=datetime.utcnow().isoformat() + "Z",
        metadata=metadata or {},
    )


def save_manifest_json(manifest: SignedManifest, path: str) -> None:
    """Save manifest to JSON file."""
    obj = {
        "payload": manifest.payload,
        "signature": manifest.signature,
        "signer_public_key": manifest.signer_public_key,
        "manifest_hash": manifest.manifest_hash,
    }
    Path(path).write_text(json.dumps(obj, indent=2))


def load_manifest_json(path: str) -> SignedManifest:
    """Load manifest from JSON file."""
    obj = json.loads(Path(path).read_text())
    return SignedManifest(
        payload=obj["payload"],
        signature=obj["signature"],
        signer_public_key=obj["signer_public_key"],
        manifest_hash=obj["manifest_hash"],
    )
