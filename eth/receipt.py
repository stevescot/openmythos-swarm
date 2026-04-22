#!/usr/bin/env python3
"""Generate deterministic contribution receipt hashes for Ethereum anchoring.

Example:
    python -m eth.receipt --worker-id alice --round-id 12 --delta-hash 0xabc... --dataset-hash 0xdef...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Dict


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def receipt_payload(
    worker_id: str,
    round_id: int,
    dataset_manifest_hash: str,
    delta_hash: str,
    challenge_input_hash: str,
    challenge_pre_output_hash: str,
    challenge_post_output_hash: str,
    challenge_pre_loss: float,
    challenge_post_loss: float,
    steps_completed: int,
    timestamp: str,
) -> Dict[str, Any]:
    return {
        "worker_id": worker_id,
        "round_id": round_id,
        "dataset_manifest_hash": dataset_manifest_hash,
        "delta_hash": delta_hash,
        "challenge_input_hash": challenge_input_hash,
        "challenge_pre_output_hash": challenge_pre_output_hash,
        "challenge_post_output_hash": challenge_post_output_hash,
        "challenge_pre_loss": challenge_pre_loss,
        "challenge_post_loss": challenge_post_loss,
        "steps_completed": steps_completed,
        "timestamp": timestamp,
    }


def receipt_hash_hex(payload: Dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    return "0x" + digest


def receipt_hash_bytes32(payload: Dict[str, Any]) -> bytes:
    """Receipt hash as raw bytes32 (for on-chain anchoring)."""
    return hashlib.sha256(canonical_json(payload)).digest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic ZKPoT receipt hash for ETH anchoring")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--dataset-hash", required=True)
    parser.add_argument("--delta-hash", required=True)
    parser.add_argument("--challenge-input-hash", required=True)
    parser.add_argument("--challenge-pre-output-hash", required=True)
    parser.add_argument("--challenge-post-output-hash", required=True)
    parser.add_argument("--challenge-pre-loss", type=float, required=True)
    parser.add_argument("--challenge-post-loss", type=float, required=True)
    parser.add_argument("--steps-completed", type=int, required=True)
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()

    payload = receipt_payload(
        worker_id=args.worker_id,
        round_id=args.round_id,
        dataset_manifest_hash=args.dataset_hash,
        delta_hash=args.delta_hash,
        challenge_input_hash=args.challenge_input_hash,
        challenge_pre_output_hash=args.challenge_pre_output_hash,
        challenge_post_output_hash=args.challenge_post_output_hash,
        challenge_pre_loss=args.challenge_pre_loss,
        challenge_post_loss=args.challenge_post_loss,
        steps_completed=args.steps_completed,
        timestamp=args.timestamp,
    )

    print(json.dumps(payload, indent=2))
    print("receipt_hash:", receipt_hash_hex(payload))


if __name__ == "__main__":
    main()
