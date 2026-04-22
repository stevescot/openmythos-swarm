"""Master coordinator node for federated training."""

import json
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Callable
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
        
        # Weighted aggregation over submitted worker metrics.
        # Weight each submission by steps_completed so longer local training
        # contributes proportionally more.
        weighted_loss_sum = 0.0
        weighted_gnorm_sum = 0.0
        total_weight = 0
        valid_count = 0
        worker_ids: List[str] = []
        worker_hashes: List[str] = []

        for sub_file in submissions:
            try:
                data = json.loads(sub_file.read_text())
                steps = int(data.get("steps_completed", 0))
                loss = float(data.get("final_loss", 0.0))
                gnorm = float(data.get("gradient_norm", 0.0))
                worker_id = str(data.get("worker_id", sub_file.stem))
                delta_hash = str(data.get("delta_hash", ""))

                if steps <= 0:
                    continue

                weighted_loss_sum += loss * steps
                weighted_gnorm_sum += gnorm * steps
                total_weight += steps
                valid_count += 1
                worker_ids.append(worker_id)
                if delta_hash:
                    worker_hashes.append(delta_hash)
            except Exception:
                continue

        avg_loss = weighted_loss_sum / max(total_weight, 1)
        avg_gnorm = weighted_gnorm_sum / max(total_weight, 1)

        # Deterministic checkpoint hash placeholder derived from submitted deltas.
        # (Next step for full tensor FedAvg: load delta tensors and merge weights.)
        aggregate_fingerprint = {
            "round_id": round_id,
            "worker_ids": sorted(worker_ids),
            "delta_hashes": sorted(worker_hashes),
            "total_weight": total_weight,
            "avg_loss": avg_loss,
            "avg_gnorm": avg_gnorm,
        }
        global_ckpt_hash = hash_object(aggregate_fingerprint)
        
        result = RoundResult(
            round_id=round_id,
            status="success" if valid_count > 0 else "failed",
            worker_submissions=len(submissions),
            valid_submissions=valid_count,
            aggregation_method="weighted_step_avg",
            global_checkpoint_hash=global_ckpt_hash,
            global_checkpoint_url=f"s3://checkpoints/round_{round_id:04d}.pt",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            metadata={
                "avg_loss": avg_loss,
                "avg_gradient_norm": avg_gnorm,
                "total_weight_steps": total_weight,
                "workers_included": len(worker_ids),
            },
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
        print(f"  Avg grad norm: {avg_gnorm:.4f}")
        print(f"  Total weight steps: {total_weight}")
        print(f"  Result hash: {manifest_hash[:16]}...")
        
        return manifest

    def status(self) -> Dict:
        """Return coordinator status."""
        return {
            "current_round": self.current_round,
            "state_dir": str(self.state_dir),
            "public_key": self.key.public_pem[:100] + "...",
        }

    def start_scheduler(
        self,
        config_fn: Callable[[int], TrainingConfig],
        worker_count: int = 3,
        interval_seconds: int = 3600,
        max_rounds: Optional[int] = None,
        submission_wait_seconds: int = 1800,
    ) -> threading.Thread:
        """Start automatic round publishing in background thread.
        
        Args:
            config_fn: Function(round_num) -> TrainingConfig for that round
            worker_count: Expected number of workers per round
            interval_seconds: Time between round start and next round (default 1hr)
            max_rounds: Max rounds to run (None = infinite)
            submission_wait_seconds: Time to wait for submissions (default 30min)
        
        Returns:
            threading.Thread running the scheduler (daemon=False, can join)
        
        Example:
            def get_config(round_num):
                if round_num <= 5:
                    return TrainingConfig(
                        seq_len=1024, train_loops=8, target_tokens=5_000_000
                    )  # Stabilization
                else:
                    return TrainingConfig(
                        seq_len=4096, train_loops=24, target_tokens=500_000_000
                    )  # Production
            
            thread = master.start_scheduler(
                config_fn=get_config,
                worker_count=3,
                interval_seconds=3600,
                max_rounds=100,
            )
            thread.join()  # Run forever (or until max_rounds)
        """
        
        def _scheduler_loop():
            round_num = 1
            prior_hash = "0" * 64
            
            while max_rounds is None or round_num <= max_rounds:
                try:
                    # Get config for this round
                    config = config_fn(round_num)
                    
                    # Publish round
                    manifest = self.publish_round(
                        version=f"10b-mps-v{round_num}",
                        config=config,
                        dataset_shard=f"fineweb-edu/sample-round{round_num}",
                        worker_count=worker_count,
                        prior_checkpoint_hash=prior_hash,
                        metadata={"scheduler_round": round_num},
                    )
                    
                    print(f"[Scheduler] Published round {round_num}, waiting {submission_wait_seconds}s for submissions...")
                    
                    # Wait for submissions
                    time.sleep(submission_wait_seconds)
                    
                    # Aggregate
                    try:
                        result = self.finalize_round(round_num)
                        prior_hash = result.payload.get("global_checkpoint_hash", "0" * 64)
                        print(f"[Scheduler] Finalized round {round_num} → next in {interval_seconds}s")
                    except Exception as e:
                        print(f"[Scheduler] Error finalizing round {round_num}: {e}")
                        prior_hash = "0" * 64  # Reset on error
                    
                    # Wait for interval (total time from round start)
                    time.sleep(max(0, interval_seconds - submission_wait_seconds))
                    round_num += 1
                    
                except Exception as e:
                    print(f"[Scheduler] ERROR in round loop: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(60)  # Retry after 1 min on error
        
        thread = threading.Thread(target=_scheduler_loop, daemon=False, name="MasterScheduler")
        thread.start()
        print(f"[Scheduler] Started background thread (max_rounds={max_rounds}, interval={interval_seconds}s)")
        return thread


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
