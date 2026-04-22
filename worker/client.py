"""Worker node for federated training on Mac Studios."""

import json
import time
import argparse
import shutil
from pathlib import Path
from typing import Optional, Dict
import sys
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import (
    Ed25519Key,
    SignedManifest,
    load_manifest_json,
    hash_checkpoint,
    hash_object,
)


class WorkerClient:
    """Worker participant in federated training."""

    def __init__(self, worker_id: str, master_state_dir: str = "./master_state", local_dir: str = "./worker_state"):
        self.worker_id = worker_id
        self.master_state_dir = Path(master_state_dir)
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.keys_dir = self.local_dir / "keys"
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        self.private_key_path = self.keys_dir / f"{self.worker_id}_private.pem"
        self.worker_key = self._load_or_create_worker_key()
        self.worker_public_key = self.worker_key.public_pem
        
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

    def _load_or_create_worker_key(self) -> Ed25519Key:
        """Load an existing worker signing key or create one."""
        if self.private_key_path.exists():
            return Ed25519Key.load_private_pem(str(self.private_key_path))
        key = Ed25519Key()
        key.save_private_pem(str(self.private_key_path))
        return key

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
        
        # Simulate tensor delta file
        delta_dir = self.local_dir / f"round_{self.current_round:04d}"
        delta_dir.mkdir(parents=True, exist_ok=True)
        delta_file = delta_dir / f"{self.worker_id}_delta.pt"
        delta_tensors = {
            "layer_0.weight": torch.ones(4, 4, dtype=torch.float32) * (0.001 * steps),
            "layer_0.bias": torch.ones(4, dtype=torch.float32) * (0.0001 * steps),
        }
        torch.save(
            {
                "worker_id": self.worker_id,
                "round_id": self.current_round,
                "delta": delta_tensors,
            },
            delta_file,
        )

        delta_hash = hash_checkpoint(delta_file.read_bytes())
        
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

        challenge_inputs = self.round_spec.get("metadata", {}).get("challenge_inputs", [])
        challenge_input_hash = hash_object(challenge_inputs)
        challenge_pre_loss = final_loss + 0.5
        challenge_post_loss = final_loss
        challenge_pre_output_hash = hash_object(
            {
                "challenge_inputs": challenge_inputs,
                "worker_id": self.worker_id,
                "round_id": self.current_round,
                "phase": "pre",
            }
        )
        challenge_post_output_hash = hash_object(
            {
                "challenge_inputs": challenge_inputs,
                "worker_id": self.worker_id,
                "round_id": self.current_round,
                "phase": "post",
                "delta_hash": delta_hash,
                "steps": steps,
            }
        )

        receipt_fields = {
            "round_id": self.current_round,
            "worker_id": self.worker_id,
            "dataset_manifest_hash": self.round_spec.get("metadata", {}).get("dataset_manifest_hash"),
            "challenge_input_hash": challenge_input_hash,
            "challenge_pre_loss": challenge_pre_loss,
            "challenge_post_loss": challenge_post_loss,
            "challenge_pre_output_hash": challenge_pre_output_hash,
            "challenge_post_output_hash": challenge_post_output_hash,
            "delta_hash": delta_hash,
            "steps_completed": steps,
            "timestamp": result["timestamp"],
        }
        work_receipt_hash = hash_object(receipt_fields)

        attestation_payload = {
            "round_id": self.current_round,
            "worker_id": self.worker_id,
            "delta_hash": delta_hash,
            "dataset_manifest_hash": self.round_spec.get("metadata", {}).get("dataset_manifest_hash"),
            "challenge_input_hash": challenge_input_hash,
            "challenge_pre_loss": challenge_pre_loss,
            "challenge_post_loss": challenge_post_loss,
            "challenge_pre_output_hash": challenge_pre_output_hash,
            "challenge_post_output_hash": challenge_post_output_hash,
            "steps_completed": steps,
            "timestamp": result["timestamp"],
            "work_receipt_hash": work_receipt_hash,
        }
        result["attestation_payload"] = attestation_payload
        result["submission_signature"] = self.worker_key.sign(attestation_payload)
        result["worker_public_key"] = self.worker_public_key
        
        print(f"[Worker {self.worker_id}] Training complete")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Gradient norm: {gradient_norm:.2f}")
        print(f"  Delta hash: {delta_hash[:16]}...")
        
        return result

    def submit_results(self, training_result: Dict) -> bool:
        """Submit training results to master."""
        try:
            round_id = training_result["round_id"]
            training_result = dict(training_result)
            
            # Save submission locally
            sub_dir = self.local_dir / f"round_{round_id:04d}"
            sub_dir.mkdir(parents=True, exist_ok=True)
            sub_file = sub_dir / "submission.json"
            sub_file.write_text(json.dumps(training_result, indent=2))
            
            # Also copy to master submissions directory
            master_sub_dir = self.master_state_dir / f"submissions/round_{round_id:04d}"
            master_sub_dir.mkdir(parents=True, exist_ok=True)

            # Copy tensor delta into master submission dir for tensor-level aggregation
            delta_src = Path(str(training_result.get("delta_file", "")))
            if delta_src.exists():
                delta_dst = master_sub_dir / f"{self.worker_id}_delta.pt"
                shutil.copy2(delta_src, delta_dst)
                training_result["delta_file"] = str(delta_dst)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenMythos Swarm basic worker client")
    parser.add_argument("--worker-id", default="mac_studio_01", help="Worker identifier")
    parser.add_argument("--master-state-dir", default="./test_master_state", help="Master state directory")
    parser.add_argument("--round", type=int, default=1, help="Round ID to fetch")
    parser.add_argument("--steps", type=int, default=5, help="Local simulated steps")
    parser.add_argument("--print-public-key", action="store_true", help="Print worker public key and exit")
    args = parser.parse_args()

    worker = WorkerClient(worker_id=args.worker_id, master_state_dir=args.master_state_dir)
    if args.print_public_key:
        print(worker.worker_public_key)
        return

    spec = worker.fetch_round_spec(round_id=args.round)

    if spec:
        result = worker.train_locally(steps=args.steps)
        worker.submit_results(result)
        print(f"\nWorker status: {worker.status()}")
    else:
        print("Could not fetch round spec")


if __name__ == "__main__":
    main()
