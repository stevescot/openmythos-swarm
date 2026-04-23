#!/usr/bin/env python3
"""Tests for DatasetLedger: lifecycle, ordering, ratios, coverage."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_ledger import (
    DatasetLedger,
    ShardStatus,
    PHASE_THRESHOLDS,
    build_ledger_from_profiles,
)


def test_register_and_persist():
    with tempfile.TemporaryDirectory() as d:
        ledger = DatasetLedger(Path(d))
        ledger.register_shard("fineweb:shard-A", domain="text", quality_score=0.9, token_estimate=1_000_000)
        assert "fineweb:shard-A" in ledger._shards

        # Reload from disk
        ledger2 = DatasetLedger(Path(d))
        assert "fineweb:shard-A" in ledger2._shards
        assert ledger2._shards["fineweb:shard-A"].quality_score == 0.9
    print("  ✓ register_and_persist")


def test_lifecycle_transitions():
    with tempfile.TemporaryDirectory() as d:
        ledger = DatasetLedger(Path(d))
        ledger.register_shard("shard-A", domain="text", status=ShardStatus.APPROVED)
        assert ledger._shards["shard-A"].status == ShardStatus.APPROVED

        ledger.mark_scheduled("shard-A", round_id=1)
        assert ledger._shards["shard-A"].status == ShardStatus.SCHEDULED
        assert ledger._shards["shard-A"].times_scheduled == 1

        ledger.mark_in_progress("shard-A")
        assert ledger._shards["shard-A"].status == ShardStatus.IN_PROGRESS

        ledger.mark_completed("shard-A", tokens_consumed=500_000, checkpoint_hash="abc123")
        assert ledger._shards["shard-A"].status == ShardStatus.REPLAYABLE
        assert ledger._shards["shard-A"].tokens_trained == 500_000
        assert ledger.total_tokens_trained == 500_000
    print("  ✓ lifecycle_transitions")


def test_quality_ordering():
    with tempfile.TemporaryDirectory() as d:
        ledger = DatasetLedger(Path(d))
        ledger.register_shard("low",  domain="text", quality_score=0.3, token_estimate=1_000_000)
        ledger.register_shard("high", domain="text", quality_score=0.9, token_estimate=1_000_000)
        ledger.register_shard("mid",  domain="text", quality_score=0.6, token_estimate=1_000_000)

        chosen = ledger.next_shard()
        assert chosen == "high", f"Expected high-quality shard first, got {chosen}"
    print("  ✓ quality_ordering")


def test_least_trained_first():
    with tempfile.TemporaryDirectory() as d:
        ledger = DatasetLedger(Path(d))
        ledger.register_shard("A", domain="text", quality_score=0.8)
        ledger.register_shard("B", domain="text", quality_score=0.8)

        # Schedule A once
        ledger.mark_scheduled("A", round_id=1)
        ledger.mark_completed("A", tokens_consumed=0)
        # Re-approve A for selection
        ledger._shards["A"].status = ShardStatus.APPROVED
        ledger.save()

        # B is untrained → should come first
        chosen = ledger.next_shard()
        assert chosen == "B", f"Expected least-trained shard B, got {chosen}"
    print("  ✓ least_trained_first")


def test_recency_balancing():
    with tempfile.TemporaryDirectory() as d:
        ledger = DatasetLedger(Path(d))
        ledger.register_shard("X", domain="text", quality_score=0.9)
        ledger.register_shard("Y", domain="text", quality_score=0.8)

        first = ledger.next_shard()        # X (higher quality)
        second = ledger.next_shard(last_shard_id=first)  # should avoid X
        assert second == "Y", f"Expected recency balancing to pick Y, got {second}"
    print("  ✓ recency_balancing")


def test_coverage_tracking():
    with tempfile.TemporaryDirectory() as d:
        ledger = DatasetLedger(Path(d))
        ledger.register_shard("S", domain="text", token_estimate=1_000_000)
        ledger.mark_scheduled("S", round_id=1)
        ledger.mark_completed("S", tokens_consumed=1_000_000)

        s = ledger._shards["S"]
        assert s.coverage == 1.0
        assert s.status == ShardStatus.COMPLETED
    print("  ✓ coverage_tracking")


def test_phase_ratios():
    with tempfile.TemporaryDirectory() as d:
        ledger = DatasetLedger(Path(d))
        assert ledger.current_phase == "bootstrap"

        # Simulate enough tokens to advance to broaden
        ledger._total_tokens_trained = PHASE_THRESHOLDS["broaden"] + 1
        assert ledger.current_phase == "broaden"

        ledger._total_tokens_trained = PHASE_THRESHOLDS["specialise"] + 1
        assert ledger.current_phase == "specialise"
    print("  ✓ phase_ratios")


def test_build_from_profiles():
    with tempfile.TemporaryDirectory() as d:
        ledger = DatasetLedger(Path(d))
        profiles = {
            "fineweb": ["HuggingFaceFW/fineweb-edu:sample-10BT", "c4:en"],
            "instruction": ["OpenAssistant/oasst1:default"],
        }
        code_shards = ["bigcode/the-stack-v2-train-smol-ids:python"]
        build_ledger_from_profiles(
            ledger=ledger,
            profiles_dir=Path(d),
            scheduler_profiles=profiles,
            approved_code_shards=code_shards,
        )
        assert len(ledger._shards) == 4
        assert ledger._shards["bigcode/the-stack-v2-train-smol-ids:python"].domain == "code"
        assert ledger._shards["OpenAssistant/oasst1:default"].domain == "instruction"
        assert ledger._shards["HuggingFaceFW/fineweb-edu:sample-10BT"].domain == "text"
    print("  ✓ build_from_profiles")


def test_summary():
    with tempfile.TemporaryDirectory() as d:
        ledger = DatasetLedger(Path(d))
        ledger.register_shard("t1", domain="text")
        ledger.register_shard("t2", domain="text")
        ledger.register_shard("c1", domain="code")
        s = ledger.summary()
        assert s["shard_count"] == 3
        assert s["by_domain"]["text"]["shards"] == 2
        assert s["by_domain"]["code"]["shards"] == 1
    print("  ✓ summary")


def run_all():
    print("\n" + "=" * 60)
    print("DATASET LEDGER TESTS")
    print("=" * 60)
    test_register_and_persist()
    test_lifecycle_transitions()
    test_quality_ordering()
    test_least_trained_first()
    test_recency_balancing()
    test_coverage_tracking()
    test_phase_ratios()
    test_build_from_profiles()
    test_summary()
    print("\nAll ledger tests passed ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all()
