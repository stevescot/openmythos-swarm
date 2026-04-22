#!/usr/bin/env python3
"""Tests for Ethereum receipt hashing/export pipeline."""

import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import TrainingConfig
from eth.receipt import receipt_payload, receipt_hash_hex
from master.server import MasterCoordinator
from worker.client import WorkerClient


def test_receipt_hash_deterministic() -> None:
    payload = receipt_payload(
        worker_id="alice",
        round_id=7,
        dataset_manifest_hash="0xaaa",
        delta_hash="0xbbb",
        challenge_input_hash="0xccc",
        challenge_pre_output_hash="0xddd",
        challenge_post_output_hash="0xeee",
        challenge_pre_loss=4.2,
        challenge_post_loss=3.9,
        steps_completed=8,
        timestamp="2026-04-22T14:00:00Z",
    )

    h1 = receipt_hash_hex(payload)
    h2 = receipt_hash_hex(dict(payload))

    assert h1 == h2
    assert h1.startswith("0x")
    assert len(h1) == 66  # 0x + 64 hex chars


def test_export_round_receipts() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        master_state = root / "master"
        master_state.mkdir()

        master = MasterCoordinator(state_dir=str(master_state))

        cfg = TrainingConfig(
            seq_len=1024,
            micro_batch=1,
            grad_accum=4,
            train_loops=8,
            learning_rate=2e-4,
            weight_decay=0.1,
            target_tokens=5_000_000,
        )

        master.publish_round(
            version="10b-mps-v1",
            config=cfg,
            dataset_shard="fineweb-edu/sample-10BT",
            worker_count=1,
            prior_checkpoint_hash="0" * 64,
            metadata={"test": True},
        )

        worker = WorkerClient(
            worker_id="alice",
            master_state_dir=str(master_state),
            local_dir=str(root / "worker_alice"),
        )
        master.register_worker(worker.worker_id, worker.worker_public_key, metadata={"test": True})

        spec = worker.fetch_round_spec(round_id=1)
        assert spec is not None
        result = worker.train_locally(steps=5)
        assert worker.submit_results(result)

        master.finalize_round(round_id=1)

        exported = master.export_round_receipts(round_id=1, credits_per_step=2)
        assert len(exported) == 1

        receipt_obj = __import__("json").loads(exported[0].read_text())
        assert receipt_obj["worker_id"] == "alice"
        assert receipt_obj["round_id"] == 1
        assert receipt_obj["credits"] == 10
        assert receipt_obj["receipt_hash"].startswith("0x")
        assert receipt_obj["receipt_payload"]["steps_completed"] == 5


if __name__ == "__main__":
    test_receipt_hash_deterministic()
    test_export_round_receipts()
    print("eth receipt tests passed")
