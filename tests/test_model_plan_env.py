#!/usr/bin/env python3
"""Tests for model plans and worker trainer env propagation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.model_plans import get_model_plan, resolve_plan_script
from worker.contrib import WorkerContributor


def test_model_plans():
    plan = get_model_plan("10b-fineweb")
    assert plan["model_variant"] == "mythos_10b"
    assert plan["status"] == "runnable"
    assert resolve_plan_script("10b-fineweb", "mps") == "training/10b_apple_silicon.py"

    plan = get_model_plan("100b-blueprint")
    assert plan["status"] == "planned"
    print("  ✓ model_plans")


def test_parse_dataset_shard():
    parsed = WorkerContributor.parse_dataset_shard("HuggingFaceFW/fineweb-edu:sample-10BT")
    assert parsed["dataset"] == "HuggingFaceFW/fineweb-edu"
    assert parsed["subset"] == "sample-10BT"

    parsed = WorkerContributor.parse_dataset_shard("fineweb-edu/sample-10BT")
    assert parsed["dataset"] == "HuggingFaceFW/fineweb-edu"
    assert parsed["subset"] == "sample-10BT"

    parsed = WorkerContributor.parse_dataset_shard("c4:en")
    assert parsed["dataset"] == "c4"
    assert parsed["subset"] == "en"
    print("  ✓ parse_dataset_shard")


def test_build_trainer_env():
    worker = WorkerContributor.__new__(WorkerContributor)
    worker.device = "mps"
    worker.device_info = {
        "seq_len_default": 4096,
        "batch_default": 2,
        "grad_accum_default": 16,
    }

    env = worker.build_trainer_env(
        config={
            "seq_len": 512,
            "micro_batch": 1,
            "grad_accum": 4,
            "train_loops": 8,
            "target_tokens": 5_000_000,
            "learning_rate": 2e-4,
            "weight_decay": 0.1,
        },
        dataset_shard="HuggingFaceFW/fineweb-edu:sample-10BT",
        model_plan_id="10b-fineweb",
        round_metadata={"default_dataset_subset": "sample-10BT"},
    )

    assert env["MYTHOS_SEQ_LEN"] == "512"
    assert env["MYTHOS_MICRO_BATCH"] == "1"
    assert env["MYTHOS_GRAD_ACCUM"] == "4"
    assert env["MYTHOS_TRAIN_LOOPS"] == "8"
    assert env["MYTHOS_TARGET_TOKENS"] == "5000000"
    assert env["MYTHOS_DATASET"] == "HuggingFaceFW/fineweb-edu"
    assert env["MYTHOS_DATASET_SUBSET"] == "sample-10BT"
    assert env["MYTHOS_DATASET_SHARD"] == "HuggingFaceFW/fineweb-edu:sample-10BT"
    assert env["MYTHOS_MODEL_PLAN"] == "10b-fineweb"
    assert env["MYTHOS_MODEL_VARIANT"] == "mythos_10b"
    assert env["MYTHOS_MODEL_SIZE"] == "10b"
    print("  ✓ build_trainer_env")


def run_all():
    print("\n" + "=" * 60)
    print("MODEL PLAN + ENV TESTS")
    print("=" * 60)
    test_model_plans()
    test_parse_dataset_shard()
    test_build_trainer_env()
    print("\nAll model plan/env tests passed ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all()
