#!/usr/bin/env python3
"""Integration test for federated training."""

import sys
import json
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import TrainingConfig
from master.server import MasterCoordinator
from worker.client import WorkerClient


def test_basic_round():
    """Test a complete training round with master + multiple workers."""
    
    # Use temp directory for test state
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        master_state = tmpdir / "master"
        master_state.mkdir()
        
        print("=" * 80)
        print("FEDERATED TRAINING INTEGRATION TEST")
        print("=" * 80)
        
        # Initialize master
        print("\n[TEST] Initializing master coordinator...")
        master = MasterCoordinator(state_dir=str(master_state))
        
        # Publish round 1
        print("\n[TEST] Publishing round 1...")
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
            worker_count=3,
            prior_checkpoint_hash="0" * 64,
            metadata={"test": True},
        )
        
        assert manifest is not None, "Failed to publish round"
        assert manifest.payload["round_id"] == 1, "Round ID mismatch"
        print("[TEST] ✓ Round published successfully")
        
        # Initialize workers
        print("\n[TEST] Initializing workers...")
        workers = [
            WorkerClient(f"mac_studio_{i:02d}", master_state_dir=str(master_state), local_dir=str(tmpdir / f"worker_{i}"))
            for i in range(3)
        ]
        print(f"[TEST] ✓ {len(workers)} workers initialized")
        
        # Workers fetch spec
        print("\n[TEST] Workers fetching round spec...")
        for i, worker in enumerate(workers):
            spec = worker.fetch_round_spec(round_id=1)
            assert spec is not None, f"Worker {i} failed to fetch spec"
        print("[TEST] ✓ All workers fetched spec")
        
        # Workers train
        print("\n[TEST] Workers training locally...")
        for i, worker in enumerate(workers):
            result = worker.train_locally(steps=5)
            assert result is not None, f"Worker {i} training failed"
            assert result["worker_id"] == f"mac_studio_{i:02d}", f"Worker ID mismatch for worker {i}"
        print("[TEST] ✓ All workers completed training")
        
        # Workers submit
        print("\n[TEST] Workers submitting results...")
        for i, worker in enumerate(workers):
            result = worker.train_locally(steps=5)
            success = worker.submit_results(result)
            assert success, f"Worker {i} failed to submit"
        print("[TEST] ✓ All workers submitted results")
        
        # Master aggregates
        print("\n[TEST] Master aggregating round...")
        result_manifest = master.finalize_round(round_id=1)
        assert result_manifest is not None, "Failed to finalize round"
        assert result_manifest.payload["valid_submissions"] == 3, "Expected 3 valid submissions"
        print("[TEST] ✓ Round aggregated successfully")
        
        # Verify manifest structure
        print("\n[TEST] Verifying manifest structure...")
        assert result_manifest.signature, "Missing signature in manifest"
        assert result_manifest.manifest_hash, "Missing manifest hash"
        assert len(result_manifest.signature) > 0, "Empty signature"
        assert "round_id" in result_manifest.payload, "Missing round_id in payload"
        print("[TEST] ✓ Manifest structure valid")
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        
        return True


if __name__ == "__main__":
    try:
        success = test_basic_round()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[TEST] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
