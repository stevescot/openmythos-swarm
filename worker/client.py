"""Worker node for federated training on Mac Studios."""

import json
import time
from pathlib import Path
from typing import Optional, Dict
from dataclasses import asdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import (
    SignedManifest,
    load_manifest_json,
    hash_object,
)


class WorkerClient:
    """Worker participant in federated training."""

    def __init__(self, worker_id: str, master_state_dir: str = "./master_state", local_dir: str = "./worker_state"):
        self.worker_id = worker_id
        self.master_state_dir = Path(master_state_dir)
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_round = None
        self.round_spec = None
        self.master_public_key = None
        
        # Load master public key
        pub_key_file = self.master_state_dir / "keys" / "master_public.pem"
        if pub_key_file.exists():
            self.master_public_key = pub_key_file.read_text()
            print(f"[Worker {worker_id}] Loaded master public key")
        else:
            print(f"[Worker {worker_id}] WARNING: Master public key not found")
        
        print(f"[Worker {worker_id}] Initialized")

    def fetch_round_spec(self, round_id: int) -> Optional[SignedManifest]:
        """Fetch round specification from master."""
        spec_file = self.master_state_dir / f"rounds/round_{round_id:04d}/spec.json"
        
        if not spec_file.exists():
            print(f"[Worker {self.worker_id}] Round {round_id} spec not available yet")
            return None
        
        try:
            manifest = load_manifest_json(str(spec_file))
            self.current_round = round_id
            self.round_spec = manifest.payload
            
            print(f"[Worker {self.worker_id}] Fetched round {round_id} spec")
            print(f"  Config: seq_len={self.round_spec['config']['seq_len']}, loops={self.round_spec['config']['train_loops']}")
            print(f"  Dataset: {self.round_spec['dataset_shard']}")
            
            return manifest
        except Exception as e:
            print(f"[Worker {self.worker_id}] ERROR fetching spec: {e}")
            return None

    def train_locally(self, steps: int = 5) -> Dict:
        """Simulate local training."""
        if self.round_spec is None:
            raise ValueError("No round spec loaded")
        
        print(f"[Worker {self.worker_id}] Starting local training (simulated, {steps} steps)")
        
        # Simulate training
        loss_values = [10.0 - (i * 0.1) for i in range(steps)]
        final_loss = loss_values[-1]
        gradient_norm = 25.0
        
        # Simulate delta file
        delta_dir = self.local_dir / f"round_{self.current_round:04d}"
        delta_dir.mkdir(parents=True, exist_ok=True)
        delta_file = delta_dir / f"{self.worker_id}_delta.bin"
        delta_file.write_bytes(b"mock_delta_" + self.worker_id.encode() + b"_" + str(self.current_round).encode())
        
        delta_hash = hash_object({"worker": self.worker_id, "round": self.current_round})
        
        result = {
            "worker_id": self.worker_id,
            "round_id": self.current_round,
            "steps_completed": steps,
            "final_loss": final_loss,
            "gradient_norm": gradient_norm,
            "delta_file": str(delta_file),
            "delta_hash": delta_hash,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": {"loss_history": loss_values},
        }
        
        print(f"[Worker {self.worker_id}] Training complete")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Gradient norm: {gradient_norm:.2f}")
        print(f"  Delta hash: {delta_hash[:16]}...")
        
        return result

    def submit_results(self, training_result: Dict) -> bool:
        """Submit training results to master."""
        try:
            round_id = training_result["round_id"]
            
            # Save submission locally
            sub_dir = self.local_dir / f"round_{round_id:04d}"
            sub_dir.mkdir(parents=True, exist_ok=True)
            sub_file = sub_dir / "submission.json"
            sub_file.write_text(json.dumps(training_result, indent=2))
            
            # Also copy to master submissions directory
            master_sub_dir = self.master_state_dir / f"submissions/round_{round_id:04d}"
            master_sub_dir.mkdir(parents=True, exist_ok=True)
            master_sub_file = master_sub_dir / f"{self.worker_id}.json"
            master_sub_file.write_text(json.dumps(training_result, indent=2))
            
            print(f"[Worker {self.worker_id}] Submitted results for round {round_id}")
            return True
        except Exception as e:
            print(f"[Worker {self.worker_id}] ERROR submitting: {e}")
            return False

    def status(self) -> Dict:
        """Return worker status."""
        return {
            "worker_id": self.worker_id,
            "current_round": self.current_round,
            "local_dir": str(self.local_dir),
        }


if __name__ == "__main__":
    # Example usage
    worker = WorkerClient(worker_id="mac_studio_01", master_state_dir="./test_master_state")
    
    # Fetch round spec
    spec = worker.fetch_round_spec(round_id=1)
    
    if spec:
        # Train
        result = worker.train_locally(steps=5)
        
        # Submit
        worker.submit_results(result)
        
        print(f"\nWorker status: {worker.status()}")
    else:
        print("Could not fetch round spec")
