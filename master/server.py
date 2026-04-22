"""Master coordinator node for federated training."""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import asdict
import sys

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import (
    Ed25519Key,
    RoundSpec,
    TrainingConfig,
    RoundResult,
    SignedManifest,
    canonical_json,
    hash_object,
    create_round_spec,
    save_manifest_json,
    load_manifest_json,
)


class MasterCoordinator:
    """Master node orchestrating federated training rounds."""

    def __init__(self, state_dir: str = "./master_state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Key management
        self.keys_dir = self.state_dir / "keys"
        self.keys_dir.mkdir(exist_ok=True)
        self.private_key_path = self.keys_dir / "master_private.pem"
        self.public_key_path = self.keys_dir / "master_public.pem"
        
        # State
        self.rounds_dir = self.state_dir / "rounds"
        self.rounds_dir.mkdir(exist_ok=True)
        self.submissions_dir = self.state_dir / "submissions"
        self.submissions_dir.mkdir(exist_ok=True)
        
        self.current_round = 0
        self.key = self._load_or_create_key()
        print(f"[Master] Initialized at {self.state_dir}")
        print(f"[Master] Public key (PEM):\n{self.key.public_pem[:100]}...")

    def _load_or_create_key(self) -> Ed25519Key:
        """Load existing key or generate new one."""
        if self.private_key_path.exists():
            print(f"[Master] Loading existing key from {self.private_key_path}")
            key = Ed25519Key.load_private_pem(str(self.private_key_path))
        else:
            print(f"[Master] Generating new Ed25519 key pair...")
            key = Ed25519Key()
            key.save_private_pem(str(self.private_key_path))
            self.public_key_path.write_text(key.public_pem)
        return key

    def publish_round(
        self,
        version: str,
        config: TrainingConfig,
        dataset_shard: str,
        worker_count: int,
        prior_checkpoint_hash: str = "0" * 64,
        metadata: Optional[Dict] = None,
    ) -> SignedManifest:
        """Publish a new federated training round."""
        self.current_round += 1
        round_id = self.current_round
        
        spec = create_round_spec(
            round_id=round_id,
            version=version,
            config=config,
            dataset_shard=dataset_shard,
            worker_count=worker_count,
            prior_checkpoint_hash=prior_checkpoint_hash,
            metadata=metadata or {},
        )
        
        spec_dict = spec.to_dict()
        sig = self.key.sign(spec_dict)
        manifest_hash = hash_object(spec_dict)
        
        manifest = SignedManifest(
            payload=spec_dict,
            signature=sig,
            signer_public_key=self.key.public_pem,
            manifest_hash=manifest_hash,
        )
        
        # Save
        round_dir = self.rounds_dir / f"round_{round_id:04d}"
        round_dir.mkdir(exist_ok=True)
        spec_file = round_dir / "spec.json"
        save_manifest_json(manifest, str(spec_file))
        
        print(f"[Master] Published round {round_id}")
        print(f"  Config: seq_len={config.seq_len}, loops={config.train_loops}")
        print(f"  Expected workers: {worker_count}")
        print(f"  Manifest hash: {manifest_hash[:16]}...")
        
        return manifest

    def accept_submission(self, round_id: int, submission_path: str) -> bool:
        """Accept and validate a worker submission."""
        try:
            data = json.loads(Path(submission_path).read_text())
            
            # Verify structure (basic checks)
            assert "worker_id" in data, "Missing worker_id"
            assert "round_id" in data, "Missing round_id"
            assert data["round_id"] == round_id, f"Round mismatch: expected {round_id}, got {data['round_id']}"
            
            # Store
            submission_dir = self.submissions_dir / f"round_{round_id:04d}"
            submission_dir.mkdir(parents=True, exist_ok=True)
            worker_id = data["worker_id"]
            out_path = submission_dir / f"{worker_id}.json"
            out_path.write_text(Path(submission_path).read_text())
            
            print(f"[Master] Accepted submission from {worker_id} for round {round_id}")
            return True
        except Exception as e:
            print(f"[Master] ERROR accepting submission: {e}")
            return False

    def finalize_round(self, round_id: int) -> SignedManifest:
        """Aggregate submissions and finalize round."""
        submission_dir = self.submissions_dir / f"round_{round_id:04d}"
        
        if not submission_dir.exists():
            raise ValueError(f"No submissions found for round {round_id}")
        
        submissions = list(submission_dir.glob("*.json"))
        print(f"[Master] Aggregating {len(submissions)} submissions for round {round_id}")
        
        # Simplified: just count and report stats
        total_loss = 0.0
        valid_count = 0
        for sub_file in submissions:
            try:
                data = json.loads(sub_file.read_text())
                total_loss += data.get("final_loss", 0.0)
                valid_count += 1
            except:
                pass
        
        avg_loss = total_loss / max(valid_count, 1)
        
        result = RoundResult(
            round_id=round_id,
            status="success" if valid_count > 0 else "failed",
            worker_submissions=len(submissions),
            valid_submissions=valid_count,
            aggregation_method="simple_avg",
            global_checkpoint_hash="0" * 64,  # Placeholder
            global_checkpoint_url=f"s3://checkpoints/round_{round_id:04d}.pt",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            metadata={"avg_loss": avg_loss},
        )
        
        result_dict = result.to_dict()
        sig = self.key.sign(result_dict)
        manifest_hash = hash_object(result_dict)
        
        manifest = SignedManifest(
            payload=result_dict,
            signature=sig,
            signer_public_key=self.key.public_pem,
            manifest_hash=manifest_hash,
        )
        
        # Save
        round_dir = self.rounds_dir / f"round_{round_id:04d}"
        result_file = round_dir / "result.json"
        save_manifest_json(manifest, str(result_file))
        
        print(f"[Master] Finalized round {round_id}")
        print(f"  Valid submissions: {valid_count}/{len(submissions)}")
        print(f"  Avg loss: {avg_loss:.4f}")
        print(f"  Result hash: {manifest_hash[:16]}...")
        
        return manifest

    def status(self) -> Dict:
        """Return coordinator status."""
        return {
            "current_round": self.current_round,
            "state_dir": str(self.state_dir),
            "public_key": self.key.public_pem[:100] + "...",
        }


if __name__ == "__main__":
    # Example usage
    master = MasterCoordinator(state_dir="./test_master_state")
    
    # Publish a round
    config = TrainingConfig(
        seq_len=1024,
        micro_batch=1,
        grad_accum=4,
        train_loops=8,
        learning_rate=2e-4,
        weight_decay=0.1,
        target_tokens=5_000_000,
    )
    
    manifest = master.publish_round(
        version="10b-mps-v1",
        config=config,
        dataset_shard="fineweb-edu/sample-10BT",
        worker_count=1,
    )
    print(f"\nPublished manifest:\n{json.dumps(manifest.payload, indent=2)}\n")
    
    print(f"Master status: {master.status()}")
