#!/usr/bin/env python3
"""Prepare Ethereum anchor payloads from exported receipt files.

This utility intentionally avoids requiring Web3 for initial workflows.
It validates receipt structure and prints the canonical receipt hash + calldata-ready fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare anchor payload from receipt JSON")
    parser.add_argument("--receipt-file", required=True, help="Path to exported receipt JSON")
    parser.add_argument("--credits", type=int, default=0, help="Credits to mint for this receipt")
    parser.add_argument("--worker-address", default="0x0000000000000000000000000000000000000000", help="EVM address for worker")
    args = parser.parse_args()

    p = Path(args.receipt_file)
    if not p.exists():
        raise SystemExit(f"Receipt file not found: {p}")

    obj = json.loads(p.read_text())
    payload = obj.get("receipt_payload", {})
    receipt_hash = obj.get("receipt_hash")

    if not receipt_hash or not isinstance(receipt_hash, str) or not receipt_hash.startswith("0x"):
        raise SystemExit("Invalid or missing receipt_hash in receipt file")

    worker_id = payload.get("worker_id", obj.get("worker_id", "unknown"))
    round_id = int(payload.get("round_id", obj.get("round_id", 0)))

    print(json.dumps(
        {
            "contract_function": "anchorReceipt(bytes32,address,string,uint256,uint256)",
            "receipt_hash": receipt_hash,
            "worker_address": args.worker_address,
            "worker_id": worker_id,
            "round_id": round_id,
            "credits": args.credits,
            "note": "Use this payload with your preferred Web3/Foundry broadcast flow.",
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
