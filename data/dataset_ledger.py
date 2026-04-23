#!/usr/bin/env python3
"""
Dataset ledger — tracks shard lifecycle, training coverage, recency, and quality.

Provides curriculum-aware shard selection for the scheduler with:
  - quality-based ordering
  - "least trained first" prioritisation
  - recency balancing (avoid back-to-back repeats)
  - shard lifecycle state tracking
  - token/window coverage tracking
  - controlled text/code/instruction ratio mixing
"""

from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEDGER_FILE = "dataset_ledger.json"

# Curriculum phase token targets (cumulative tokens trained so far)
PHASE_THRESHOLDS = {
    "bootstrap": 0,           # 0 – 5B  tokens
    "broaden":   5_000_000_000,  # 5B – 30B
    "specialise": 30_000_000_000, # 30B+
}

# Domain ratios per phase  {phase: {domain: weight}}
PHASE_RATIOS: Dict[str, Dict[str, float]] = {
    "bootstrap":  {"text": 1.00, "code": 0.00, "instruction": 0.00},
    "broaden":    {"text": 0.80, "code": 0.15, "instruction": 0.05},
    "specialise": {"text": 0.70, "code": 0.20, "instruction": 0.10},
}


class ShardStatus(str, Enum):
    CANDIDATE   = "candidate"    # Known, not yet reviewed
    VALIDATED   = "validated"    # Quality checks passed, not yet approved
    APPROVED    = "approved"     # Ready to train on
    SCHEDULED   = "scheduled"    # Assigned to an upcoming/in-progress round
    IN_PROGRESS = "in_progress"  # Currently being trained
    COMPLETED   = "completed"    # All windows fully trained
    REPLAYABLE  = "replayable"   # Completed but eligible for replay
    RETIRED     = "retired"      # Never use again


# ---------------------------------------------------------------------------
# Shard record
# ---------------------------------------------------------------------------

class ShardRecord:
    def __init__(
        self,
        shard_id: str,
        domain: str = "text",
        quality_score: float = 0.5,
        token_estimate: int = 0,
        status: ShardStatus = ShardStatus.APPROVED,
        tokens_trained: int = 0,
        times_scheduled: int = 0,
        times_completed: int = 0,
        last_round_id: Optional[int] = None,
        last_checkpoint_hash: Optional[str] = None,
        curriculum_phase: str = "bootstrap",
        notes: str = "",
    ):
        self.shard_id = shard_id
        self.domain = domain
        self.quality_score = quality_score
        self.token_estimate = token_estimate
        self.status = ShardStatus(status)
        self.tokens_trained = tokens_trained
        self.times_scheduled = times_scheduled
        self.times_completed = times_completed
        self.last_round_id = last_round_id
        self.last_checkpoint_hash = last_checkpoint_hash
        self.curriculum_phase = curriculum_phase
        self.notes = notes

    # fraction of estimated tokens consumed (0–1, clamped)
    @property
    def coverage(self) -> float:
        if self.token_estimate <= 0:
            return 0.0
        return min(1.0, self.tokens_trained / self.token_estimate)

    def to_dict(self) -> dict:
        return {
            "shard_id": self.shard_id,
            "domain": self.domain,
            "quality_score": self.quality_score,
            "token_estimate": self.token_estimate,
            "status": self.status.value,
            "tokens_trained": self.tokens_trained,
            "times_scheduled": self.times_scheduled,
            "times_completed": self.times_completed,
            "last_round_id": self.last_round_id,
            "last_checkpoint_hash": self.last_checkpoint_hash,
            "curriculum_phase": self.curriculum_phase,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ShardRecord":
        return cls(**d)


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

class DatasetLedger:
    """Persistent dataset ledger with curriculum-aware shard selection."""

    def __init__(self, ledger_dir: Path):
        self.ledger_dir = Path(ledger_dir)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.ledger_dir / LEDGER_FILE
        self._shards: Dict[str, ShardRecord] = {}
        self._total_tokens_trained: int = 0
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.ledger_path.exists():
            return
        with self.ledger_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self._total_tokens_trained = raw.get("total_tokens_trained", 0)
        for entry in raw.get("shards", []):
            r = ShardRecord.from_dict(entry)
            self._shards[r.shard_id] = r

    def save(self) -> None:
        payload = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_tokens_trained": self._total_tokens_trained,
            "shards": [s.to_dict() for s in self._shards.values()],
        }
        with self.ledger_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_shard(
        self,
        shard_id: str,
        domain: str = "text",
        quality_score: float = 0.5,
        token_estimate: int = 0,
        status: ShardStatus = ShardStatus.APPROVED,
        notes: str = "",
        overwrite: bool = False,
    ) -> ShardRecord:
        if shard_id in self._shards and not overwrite:
            return self._shards[shard_id]
        r = ShardRecord(
            shard_id=shard_id,
            domain=domain,
            quality_score=quality_score,
            token_estimate=token_estimate,
            status=status,
            notes=notes,
        )
        self._shards[shard_id] = r
        self.save()
        return r

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def mark_scheduled(self, shard_id: str, round_id: int) -> None:
        r = self._shards[shard_id]
        r.status = ShardStatus.SCHEDULED
        r.times_scheduled += 1
        r.last_round_id = round_id
        self.save()

    def mark_in_progress(self, shard_id: str) -> None:
        r = self._shards[shard_id]
        r.status = ShardStatus.IN_PROGRESS
        self.save()

    def mark_completed(
        self,
        shard_id: str,
        tokens_consumed: int = 0,
        checkpoint_hash: Optional[str] = None,
        replay: bool = False,
    ) -> None:
        r = self._shards[shard_id]
        r.tokens_trained += tokens_consumed
        self._total_tokens_trained += tokens_consumed
        r.last_checkpoint_hash = checkpoint_hash
        r.times_completed += 1
        if replay or r.coverage < 0.99:
            r.status = ShardStatus.REPLAYABLE
        else:
            r.status = ShardStatus.COMPLETED
        self.save()

    def mark_retired(self, shard_id: str) -> None:
        self._shards[shard_id].status = ShardStatus.RETIRED
        self.save()

    # ------------------------------------------------------------------
    # Curriculum phase
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> str:
        t = self._total_tokens_trained
        if t >= PHASE_THRESHOLDS["specialise"]:
            return "specialise"
        if t >= PHASE_THRESHOLDS["broaden"]:
            return "broaden"
        return "bootstrap"

    @property
    def total_tokens_trained(self) -> int:
        return self._total_tokens_trained

    # ------------------------------------------------------------------
    # Shard selection — curriculum-aware
    # ------------------------------------------------------------------

    def next_shard(
        self,
        last_shard_id: Optional[str] = None,
        force_domain: Optional[str] = None,
    ) -> Optional[str]:
        """Return the best shard ID for the next round.

        Selection criteria (in priority order):
        1. Status must be APPROVED or REPLAYABLE (not completed/retired/scheduled)
        2. Domain must match phase ratio (unless force_domain is set)
        3. Sort by: quality DESC, times_trained ASC, last_round_id ASC
        4. Never return last_shard_id twice in a row (recency balancing)
        """
        phase = self.current_phase
        ratios = PHASE_RATIOS[phase]

        eligible = [
            s for s in self._shards.values()
            if s.status in (ShardStatus.APPROVED, ShardStatus.REPLAYABLE)
        ]

        if not eligible:
            return None

        # Domain selection from phase ratio
        domain = force_domain
        if domain is None:
            domain = self._sample_domain(ratios, eligible)

        candidates = [s for s in eligible if s.domain == domain]
        if not candidates:
            candidates = eligible  # fall back to any domain

        # Sort: quality DESC, times_trained ASC, last_round_id ASC (oldest first)
        candidates.sort(key=lambda s: (
            -s.quality_score,
            s.times_scheduled,
            s.last_round_id if s.last_round_id is not None else -1,
        ))

        # Recency balancing: skip last used if alternatives exist
        if last_shard_id and len(candidates) > 1:
            non_recent = [c for c in candidates if c.shard_id != last_shard_id]
            if non_recent:
                candidates = non_recent

        return candidates[0].shard_id

    def _sample_domain(
        self,
        ratios: Dict[str, float],
        eligible: List[ShardRecord],
    ) -> str:
        """Choose a domain according to phase ratios, biased toward what's eligible."""
        import random
        available_domains = {s.domain for s in eligible}
        filtered = {d: w for d, w in ratios.items() if d in available_domains and w > 0}
        if not filtered:
            return next(iter(available_domains))
        total = sum(filtered.values())
        r = random.random() * total
        cumulative = 0.0
        for domain, weight in filtered.items():
            cumulative += weight
            if r <= cumulative:
                return domain
        return next(iter(filtered))

    # ------------------------------------------------------------------
    # Coverage report
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        phase = self.current_phase
        by_status: Dict[str, int] = {}
        by_domain: Dict[str, dict] = {}
        for s in self._shards.values():
            by_status[s.status.value] = by_status.get(s.status.value, 0) + 1
            if s.domain not in by_domain:
                by_domain[s.domain] = {"shards": 0, "tokens_trained": 0, "tokens_estimated": 0}
            by_domain[s.domain]["shards"] += 1
            by_domain[s.domain]["tokens_trained"] += s.tokens_trained
            by_domain[s.domain]["tokens_estimated"] += s.token_estimate
        return {
            "current_phase": phase,
            "total_tokens_trained": self._total_tokens_trained,
            "phase_thresholds": PHASE_THRESHOLDS,
            "phase_ratios": PHASE_RATIOS[phase],
            "shard_count": len(self._shards),
            "by_status": by_status,
            "by_domain": by_domain,
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n=== Dataset Ledger ===")
        print(f"Phase: {s['current_phase']}")
        print(f"Total tokens trained: {s['total_tokens_trained']:,}")
        print(f"Total shards: {s['shard_count']}")
        print(f"Phase ratios: {s['phase_ratios']}")
        print(f"\nBy status:")
        for status, count in s["by_status"].items():
            print(f"  {status:>12}: {count}")
        print(f"\nBy domain:")
        for domain, info in s["by_domain"].items():
            trained = info["tokens_trained"]
            estimated = info["tokens_estimated"]
            pct = 100.0 * trained / estimated if estimated > 0 else 0.0
            print(f"  {domain:>12}: {info['shards']} shards, {trained:,} / {estimated:,} tokens ({pct:.1f}%)")
        print()


# ---------------------------------------------------------------------------
# Populate ledger from scheduler profiles + code manifest
# ---------------------------------------------------------------------------

def build_ledger_from_profiles(
    ledger: DatasetLedger,
    profiles_dir: Path,
    scheduler_profiles: dict,
    approved_code_shards: Optional[list] = None,
) -> None:
    """Seed the ledger from scheduler profiles if not already present."""

    # Quality scores for known sources
    QUALITY = {
        "HuggingFaceFW/fineweb-edu": 0.95,
        "c4": 0.80,
        "allenai/dolma": 0.75,
        "togethercomputer/RedPajama-Data-1T": 0.70,
        "OpenAssistant/oasst1": 0.90,
        "databricks/databricks-dolly-15k": 0.85,
        "HuggingFaceH4/ultrachat_200k": 0.80,
        "bigcode/the-stack-v2-train-smol-ids": 0.85,
    }

    # Token estimates per shard (approximate billions → raw tokens)
    TOKEN_EST = {
        "HuggingFaceFW/fineweb-edu:sample-10BT":    10_000_000_000,
        "HuggingFaceFW/fineweb-edu:CC-MAIN-2023-06": 40_000_000_000,
        "HuggingFaceFW/fineweb-edu:CC-MAIN-2023-14": 40_000_000_000,
        "HuggingFaceFW/fineweb-edu:CC-MAIN-2023-23": 40_000_000_000,
        "HuggingFaceFW/fineweb-edu:CC-MAIN-2023-40": 40_000_000_000,
        "c4:en":                                      350_000_000_000,
        "allenai/dolma:v1_7":                         3_000_000_000_000,
        "togethercomputer/RedPajama-Data-1T:common_crawl": 800_000_000_000,
        "OpenAssistant/oasst1:default":               100_000_000,
        "databricks/databricks-dolly-15k:train":       10_000_000,
        "HuggingFaceH4/ultrachat_200k:train_sft":     200_000_000,
        "bigcode/the-stack-v2-train-smol-ids:python":  20_000_000_000,
        "bigcode/the-stack-v2-train-smol-ids:javascript": 15_000_000_000,
        "bigcode/the-stack-v2-train-smol-ids:typescript":  8_000_000_000,
    }

    def _domain(shard_id: str) -> str:
        if any(x in shard_id for x in ("the-stack", "code", "python", "javascript", "typescript")):
            return "code"
        if any(x in shard_id for x in ("oasst", "dolly", "ultrachat", "instruction")):
            return "instruction"
        return "text"

    def _quality(shard_id: str) -> float:
        for prefix, q in QUALITY.items():
            if shard_id.startswith(prefix):
                return q
        return 0.50

    # Text + instruction shards from scheduler profiles
    registered = set()
    for profile_name, shards in scheduler_profiles.items():
        if profile_name == "code-open-safe":
            continue
        for sid in shards:
            if sid in registered:
                continue
            ledger.register_shard(
                shard_id=sid,
                domain=_domain(sid),
                quality_score=_quality(sid),
                token_estimate=TOKEN_EST.get(sid, 5_000_000_000),
                status=ShardStatus.APPROVED,
            )
            registered.add(sid)

    # Code shards from approved manifest
    if approved_code_shards:
        for sid in approved_code_shards:
            if sid in registered:
                continue
            ledger.register_shard(
                shard_id=sid,
                domain="code",
                quality_score=_quality(sid),
                token_estimate=TOKEN_EST.get(sid, 5_000_000_000),
                status=ShardStatus.APPROVED,
            )
            registered.add(sid)

    ledger.save()
