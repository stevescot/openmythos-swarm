#!/usr/bin/env python3
"""Model training plans for swarm scheduling and worker execution.

A plan describes:
- what model family/size a round is targeting
- which trainer script should be used (if available)
- which devices are supported
- default dataset profile to pair with the plan
- whether the plan is runnable today or just a blueprint
"""

from __future__ import annotations

from typing import Dict, List, Optional


MODEL_PLANS: Dict[str, Dict] = {
    "1b-fineweb": {
        "id": "1b-fineweb",
        "display_name": "OpenMythos 1B on FineWeb-Edu",
        "status": "experimental",
        "model_variant": "mythos_1b",
        "model_size": "1b",
        "version_prefix": "1b-fineweb",
        "default_dataset_profile": "fineweb",
        "default_dataset_subset": "sample-10BT",
        "trainer_type": "inline-or-custom",
        "script_map": {},
        "supported_devices": ["mps", "cuda", "rocm", "cpu"],
        "notes": "Usable for smaller proof runs and local experimentation. Requires a custom/inline runner or future upstream script.",
    },
    "10b-fineweb": {
        "id": "10b-fineweb",
        "display_name": "OpenMythos 10B on FineWeb-Edu",
        "status": "runnable",
        "model_variant": "mythos_10b",
        "model_size": "10b",
        "version_prefix": "10b-fineweb",
        "default_dataset_profile": "fineweb",
        "default_dataset_subset": "sample-10BT",
        "trainer_type": "upstream-script",
        "script_map": {
            "mps": "training/10b_apple_silicon.py",
        },
        "supported_devices": ["mps"],
        "notes": "Best-supported upstream trainer path today. Scheduler + worker should pass dataset subset via env.",
    },
    "100b-blueprint": {
        "id": "100b-blueprint",
        "display_name": "OpenMythos 100B blueprint",
        "status": "planned",
        "model_variant": "mythos_100b",
        "model_size": "100b",
        "version_prefix": "100b-blueprint",
        "default_dataset_profile": "fineweb",
        "default_dataset_subset": "sample-10BT",
        "trainer_type": "planned",
        "script_map": {},
        "supported_devices": [],
        "notes": "Planning artifact only until upstream training scripts and hardware path exist.",
    },
}


def get_model_plan(plan_id: str) -> Dict:
    if plan_id not in MODEL_PLANS:
        raise KeyError(f"Unknown model plan: {plan_id}")
    return MODEL_PLANS[plan_id]


def list_model_plans() -> List[Dict]:
    return [MODEL_PLANS[k] for k in sorted(MODEL_PLANS.keys())]


def print_model_plans() -> None:
    print("Available model plans:")
    for plan in list_model_plans():
        print(f"\n- {plan['id']} [{plan['status']}]")
        print(f"  Name: {plan['display_name']}")
        print(f"  Variant: {plan['model_variant']}")
        print(f"  Size: {plan['model_size']}")
        print(f"  Default dataset profile: {plan['default_dataset_profile']}")
        print(f"  Supported devices: {', '.join(plan['supported_devices']) if plan['supported_devices'] else 'none yet'}")
        if plan.get('notes'):
            print(f"  Notes: {plan['notes']}")


def resolve_plan_script(plan_id: str, device: str) -> Optional[str]:
    plan = get_model_plan(plan_id)
    return plan.get("script_map", {}).get(device)
