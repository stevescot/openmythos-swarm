#!/usr/bin/env python3
"""
Auto-scheduler for federated training rounds.

Runs continuously, publishing new training tasks and aggregating results.
Start this on your master node and it will handle rounds automatically.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import TrainingConfig
from master.server import MasterCoordinator


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


if __name__ == "__main__":
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
        choices=["stabilization", "production", "mixed"],
        default="mixed",
        help="Configuration strategy (default: mixed = stabilization then production)"
    )
    
    args = parser.parse_args()
    
    # Select config strategy
    config_map = {
        "stabilization": get_config_stabilization,
        "production": get_config_production,
        "mixed": get_config_mixed,
    }
    config_fn = config_map[args.config]
    
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
    print("=" * 80)
    
    master = MasterCoordinator(state_dir=args.state_dir)
    
    # Start scheduler
    thread = master.start_scheduler(
        config_fn=config_fn,
        worker_count=args.workers,
        interval_seconds=args.interval,
        max_rounds=args.max_rounds,
        submission_wait_seconds=args.submission_wait,
    )
    
    # Run until user interrupts
    try:
        thread.join()
    except KeyboardInterrupt:
        print("\n[Scheduler] Interrupted by user")
        sys.exit(0)
