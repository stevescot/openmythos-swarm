#!/usr/bin/env python3
"""
Auto-scheduler for federated training rounds.

Runs continuously, publishing new training tasks and aggregating results.
Start this on your master node and it will handle rounds automatically.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Callable, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

from common import TrainingConfig
from master.server import MasterCoordinator
from data.dataset_ledger import (
    DatasetLedger,
    ShardStatus,
    build_ledger_from_profiles,
)


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


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_code_open_safe_shards(repo_root: Path) -> List[str]:
    """Load approved code shards from manifest.

    Manifest format supported:
    - {"shards": ["dataset:split", ...]}
    - {"shards": [{"id": "dataset:split", ...}, ...]}
    """
    manifest_path = repo_root / "data" / "approved_code_shards.json"
    if not manifest_path.exists():
        return []

    obj = _load_json(manifest_path)
    raw_shards = obj.get("shards", []) if isinstance(obj, dict) else []
    out: List[str] = []
    for s in raw_shards:
        if isinstance(s, str) and s.strip():
            out.append(s.strip())
        elif isinstance(s, dict):
            sid = str(s.get("id", "")).strip()
            if sid:
                out.append(sid)
    return out


def load_license_allowlist(repo_root: Path) -> List[str]:
    allowlist_path = repo_root / "data" / "licenses_allowlist.json"
    if not allowlist_path.exists():
        return []
    obj = _load_json(allowlist_path)
    licenses = obj.get("allowed_licenses", []) if isinstance(obj, dict) else []
    return [str(x).strip() for x in licenses if str(x).strip()]


def get_dataset_profiles(repo_root: Path) -> Dict[str, List[str]]:
    profiles = dict(DATASET_PROFILES)
    code_shards = load_code_open_safe_shards(repo_root)
    if code_shards:
        profiles["code-open-safe"] = code_shards
    return profiles


def print_dataset_profiles(profiles: Dict[str, List[str]], repo_root: Path) -> None:
    print("Available open dataset profiles:")
    for name, shards in profiles.items():
        print(f"\n- {name} ({len(shards)} shards)")
        for idx, shard in enumerate(shards, start=1):
            print(f"  {idx:>2}. {shard}")

    allow = load_license_allowlist(repo_root)
    if allow:
        print("\nLicense allowlist used by code-open-safe:")
        for lic in allow:
            print(f"  - {lic}")


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

    repo_root = Path(__file__).resolve().parent.parent
    dataset_profiles = get_dataset_profiles(repo_root)
    dataset_profile_choices = sorted(set(list(dataset_profiles.keys()) + ["code-open-safe"]))

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
        choices=dataset_profile_choices,
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
        print_dataset_profiles(dataset_profiles, repo_root)
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
        if args.dataset_profile not in dataset_profiles:
            raise ValueError(
                f"Dataset profile '{args.dataset_profile}' is unavailable. "
                "If using code-open-safe, ensure data/approved_code_shards.json exists."
            )
        shard_list = dataset_profiles[args.dataset_profile]

    # ------------------------------------------------------------------
    # Dataset ledger — build/load, seed from profiles, print summary
    # ------------------------------------------------------------------
    ledger_dir = Path(args.state_dir)
    ledger = DatasetLedger(ledger_dir)

    # Seed ledger from scheduler profiles (idempotent — skips existing entries)
    approved_code = load_code_open_safe_shards(repo_root)
    build_ledger_from_profiles(
        ledger=ledger,
        profiles_dir=repo_root / "data",
        scheduler_profiles=dataset_profiles,
        approved_code_shards=approved_code,
    )

    # Build a ledger-aware shard selection function that:
    #   - uses curriculum phase ratios
    #   - picks highest quality / least trained / oldest used shard
    #   - avoids immediate repeats (recency balancing)
    #   - marks shards SCHEDULED on pick, COMPLETED on finalize
    _last_shard: list[Optional[str]] = [None]

    def ledger_shard_fn(round_num: int) -> str:
        chosen = ledger.next_shard(last_shard_id=_last_shard[0])
        if chosen is None:
            # All shards exhausted — fall back to plain round-robin
            i = (round_num - 1) % len(shard_list)
            chosen = shard_list[i]
        ledger.mark_scheduled(chosen, round_num)
        _last_shard[0] = chosen
        return chosen

    dataset_shard_fn = ledger_shard_fn

    # ------------------------------------------------------------------
    # Startup banner
    # ------------------------------------------------------------------
    print("=" * 80)
    print("OpenMythos Swarm — Auto Scheduler")
    print("=" * 80)
    print(f"State dir: {args.state_dir}")
    print(f"Workers per round: {args.workers}")
    print(f"Interval: {args.interval}s")
    print(f"Submission wait: {args.submission_wait}s")
    print(f"Max rounds: {args.max_rounds or 'infinite'}")
    print(f"Config strategy: {args.config}")
    print(f"Dataset profile: {args.dataset_profile} (ledger-aware)")
    print(f"Dataset shards ({len(shard_list)}):")
    for idx, shard in enumerate(shard_list, start=1):
        print(f"  {idx:>2}. {shard}")
    ledger.print_summary()
    print("=" * 80)

    master = MasterCoordinator(state_dir=args.state_dir)

    # ------------------------------------------------------------------
    # Wrap finalize_round to update ledger after each completed round
    # ------------------------------------------------------------------
    _orig_finalize = master.finalize_round

    def _finalize_with_ledger(round_id: int):
        result = _orig_finalize(round_id)
        shard_id = result.payload.get("dataset_shard") or _last_shard[0]
        if shard_id and shard_id in ledger._shards:
            tokens = result.payload.get("total_tokens_trained", 0)
            chk = result.payload.get("global_checkpoint_hash")
            ledger.mark_completed(shard_id, tokens_consumed=tokens, checkpoint_hash=chk)
            print(f"[Ledger] Marked '{shard_id}' completed (+{tokens:,} tokens)")
        return result

    master.finalize_round = _finalize_with_ledger

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
