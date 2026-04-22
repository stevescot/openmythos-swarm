#!/usr/bin/env python3
"""
Auto-scheduler for federated training rounds.

Runs continuously, publishing new training tasks and aggregating results.
Start this on your master node and it will handle rounds automatically.
"""

import sys
from pathlib import Path
from typing import Dict, List, Callable

sys.path.insert(0, str(Path(__file__).parent))

from common import TrainingConfig
from master.server import MasterCoordinator


DATASET_PROFILES: Dict[str, List[str]] = {
    # Web-scale educational/common crawl text (recommended default)
    "fineweb": [
        "HuggingFaceFW/fineweb-edu:sample-10BT",
        "HuggingFaceFW/fineweb-edu:CC-MAIN-2023-06",
        "HuggingFaceFW/fineweb-edu:CC-MAIN-2023-14",
        "HuggingFaceFW/fineweb-edu:CC-MAIN-2023-23",
        "HuggingFaceFW/fineweb-edu:CC-MAIN-2023-40",
    ],
    # Mixed open corpora for broad language modeling coverage
    "open-mix": [
        "HuggingFaceFW/fineweb-edu:sample-10BT",
        "c4:en",
        "allenai/dolma:v1_7",
        "togethercomputer/RedPajama-Data-1T:common_crawl",
        "HuggingFaceFW/fineweb-edu:CC-MAIN-2023-23",
    ],
    # Instruction-heavy profile (fine-tuning / alignment style rounds)
    "instruction": [
        "OpenAssistant/oasst1:default",
        "databricks/databricks-dolly-15k:train",
        "HuggingFaceH4/ultrachat_200k:train_sft",
    ],
}


def print_dataset_profiles() -> None:
    print("Available open dataset profiles:")
    for name, shards in DATASET_PROFILES.items():
        print(f"\n- {name} ({len(shards)} shards)")
        for idx, shard in enumerate(shards, start=1):
            print(f"  {idx:>2}. {shard}")


def make_dataset_shard_fn(shards: List[str]) -> Callable[[int], str]:
    def _pick(round_num: int) -> str:
        # Round-robin over provided shards so workers get sequential chunks
        i = (round_num - 1) % len(shards)
        return shards[i]

    return _pick


def get_config_stabilization(round_num: int) -> TrainingConfig:
    """Stabilization phase: small rounds to verify setup."""
    return TrainingConfig(
        seq_len=1024,
        micro_batch=1,
        grad_accum=4,
        train_loops=8,
        learning_rate=2e-4,
        weight_decay=0.1,
        target_tokens=5_000_000,  # 5M tokens per round (small)
    )


def get_config_production(round_num: int) -> TrainingConfig:
    """Production phase: full training."""
    return TrainingConfig(
        seq_len=4096,
        micro_batch=2,
        grad_accum=8,
        train_loops=24,
        learning_rate=2e-4,
        weight_decay=0.1,
        target_tokens=100_000_000,  # 100M tokens per round (large)
    )


def get_config_mixed(round_num: int) -> TrainingConfig:
    """Adaptive config based on round number.
    
    Rounds 1-5: Stabilization (5M tokens, 8 loops)
    Rounds 6+: Production (100M tokens, 24 loops)
    """
    if round_num <= 5:
        return get_config_stabilization(round_num)
    else:
        return get_config_production(round_num)


def get_config_micro_real(round_num: int) -> TrainingConfig:
    """Micro-round profile for normal internet/GPU contributors (hours, not days)."""
    return TrainingConfig(
        seq_len=512,
        micro_batch=1,
        grad_accum=4,
        train_loops=4,
        learning_rate=2e-4,
        weight_decay=0.1,
        target_tokens=5_000_000,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Auto-scheduler for federated training")
    parser.add_argument(
        "--state-dir",
        default="./master_state",
        help="Master state directory (default: ./master_state)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Expected number of workers per round (default: 3)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Seconds between round start and next round (default: 3600 = 1hr)"
    )
    parser.add_argument(
        "--submission-wait",
        type=int,
        default=1800,
        help="Seconds to wait for submissions before aggregating (default: 1800 = 30min)"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Max rounds to run (default: infinite)"
    )
    parser.add_argument(
        "--config",
        choices=["stabilization", "production", "mixed", "micro-real"],
        default="mixed",
        help="Configuration strategy (default: mixed = stabilization then production)"
    )
    parser.add_argument(
        "--dataset-profile",
        choices=["fineweb", "open-mix", "instruction"],
        default="fineweb",
        help="Open dataset profile to rotate across rounds (default: fineweb)",
    )
    parser.add_argument(
        "--dataset-shards",
        default=None,
        help="Optional comma-separated shard list to override dataset profile",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print available open dataset profiles and exit",
    )
    
    args = parser.parse_args()

    if args.list_datasets:
        print_dataset_profiles()
        return
    
    # Select config strategy
    config_map = {
        "stabilization": get_config_stabilization,
        "production": get_config_production,
        "mixed": get_config_mixed,
        "micro-real": get_config_micro_real,
    }
    config_fn = config_map[args.config]

    if args.dataset_shards:
        shard_list = [x.strip() for x in args.dataset_shards.split(",") if x.strip()]
        if not shard_list:
            raise ValueError("--dataset-shards was provided but no valid shard values were parsed")
    else:
        shard_list = DATASET_PROFILES[args.dataset_profile]

    dataset_shard_fn = make_dataset_shard_fn(shard_list)
    
    # Start master
    print("=" * 80)
    print("OpenMythos Swarm — Auto Scheduler")
    print("=" * 80)
    print(f"State dir: {args.state_dir}")
    print(f"Workers per round: {args.workers}")
    print(f"Interval: {args.interval}s")
    print(f"Submission wait: {args.submission_wait}s")
    print(f"Max rounds: {args.max_rounds or 'infinite'}")
    print(f"Config strategy: {args.config}")
    print(f"Dataset profile: {args.dataset_profile}")
    print(f"Dataset shards ({len(shard_list)}):")
    for idx, shard in enumerate(shard_list, start=1):
        print(f"  {idx:>2}. {shard}")
    print("=" * 80)
    
    master = MasterCoordinator(state_dir=args.state_dir)
    
    # Start scheduler
    thread = master.start_scheduler(
        config_fn=config_fn,
        worker_count=args.workers,
        interval_seconds=args.interval,
        max_rounds=args.max_rounds,
        submission_wait_seconds=args.submission_wait,
        dataset_shard_fn=dataset_shard_fn,
    )
    
    # Run until user interrupts
    try:
        thread.join()
    except KeyboardInterrupt:
        print("\n[Scheduler] Interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
