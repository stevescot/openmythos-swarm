#!/usr/bin/env python3
"""
Worker client with automatic backend selection (Mac MPS or NVIDIA CUDA).

For Mac Studios: Auto-detects MPS, uses 10b_apple_silicon.py
For NVIDIA GPUs: Auto-detects CUDA, uses 10b_cross_platform.py
For CPU: Falls back to CPU (slow, testing only)

Run:
    python worker/contrib.py --worker-id "mac_studio_01" --master-url ./master_state

Or with custom config:
    python worker/contrib.py \
        --worker-id "gpu_node_05" \
        --master-url https://raw.githubusercontent.com/you/openmythos-swarm/main \
        --training-repo ../openmythos-10b-apple-silicon \
        --device mps
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict
from dataclasses import asdict
import argparse


def detect_device() -> str:
    """Auto-detect compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_device_info(device: str) -> Dict:
    """Get device capabilities and requirements."""
    specs = {
        "mps": {
            "name": "Apple Silicon (MPS)",
            "min_ram_gb": 64,
            "recommended_ram_gb": 256,
            "min_vram_gb": 0,  # Unified memory
            "supported_models": ["10b"],
            "script": "10b_apple_silicon.py",
            "seq_len_default": 4096,
            "batch_default": 2,
            "grad_accum_default": 16,
            "frameworks": ["PyTorch 2.0+"],
            "hardware_examples": [
                "Mac Studio M1 Ultra (384 GB RAM) ✓",
                "Mac Studio M2 Ultra (512 GB RAM) ✓",
                "Mac Studio M3 Ultra (512 GB RAM) ✓",
                "MacBook Pro M3 Max (128 GB RAM) ✓ (slow)",
                "iMac M3 (32 GB RAM) ~ (marginal)",
            ]
        },
        "cuda": {
            "name": "NVIDIA CUDA GPU",
            "min_ram_gb": 32,
            "recommended_ram_gb": 128,
            "min_vram_gb": 40,
            "supported_models": ["10b"],
            "script": "10b_cross_platform.py",
            "seq_len_default": 8192,
            "batch_default": 8,
            "grad_accum_default": 4,
            "frameworks": ["PyTorch 2.0+", "NVIDIA CUDA 12.0+"],
            "hardware_examples": [
                "NVIDIA H100 (80GB) ✓✓ (ideal)",
                "NVIDIA A100 (40GB) ✓ (good)",
                "NVIDIA A100 (80GB) ✓✓ (good)",
                "NVIDIA L40S (48GB) ✓ (good)",
                "NVIDIA RTX 6000 Ada (48GB) ✓ (good)",
                "NVIDIA RTX 4090 (24GB) ~ (marginal)",
            ]
        },
        "cpu": {
            "name": "CPU (Fallback)",
            "min_ram_gb": 8,
            "recommended_ram_gb": 64,
            "min_vram_gb": 0,
            "supported_models": ["10b (very slow)"],
            "script": "10b_cross_platform.py",
            "seq_len_default": 512,
            "batch_default": 1,
            "grad_accum_default": 1,
            "frameworks": ["PyTorch 2.0+"],
            "hardware_examples": [
                "Testing only",
                "Multi-core CPU recommended (not practical for production)",
            ]
        },
    }
    return specs.get(device, {})


class WorkerContributor:
    """Worker that auto-detects device and runs appropriate training."""

    def __init__(
        self,
        worker_id: str,
        master_url: str = "./master_state",
        training_repo: str = "../openmythos-10b-apple-silicon",
        device: Optional[str] = None,
        force_device: bool = False,
    ):
        self.worker_id = worker_id
        self.master_url = master_url
        self.training_repo = Path(training_repo)
        self.device = device or detect_device()
        self.force_device = force_device
        
        if not force_device and device:
            print(f"[Worker {worker_id}] Using forced device: {device}")
        
        device_info = get_device_info(self.device)
        if not device_info:
            raise ValueError(f"Unknown device: {self.device}")
        
        self.device_info = device_info
        
        print(f"[Worker {worker_id}] Device: {self.device_info['name']}")
        print(f"  Min RAM: {self.device_info['min_ram_gb']} GB")
        print(f"  Min VRAM: {self.device_info['min_vram_gb']} GB")
        print(f"  Script: {self.device_info['script']}")

    def verify_requirements(self) -> bool:
        """Check if system meets minimum requirements."""
        import psutil
        
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"\n[Worker {self.worker_id}] System Check")
        print(f"  Available RAM: {ram_gb:.1f} GB (min required: {self.device_info['min_ram_gb']} GB)")
        
        if ram_gb < self.device_info['min_ram_gb']:
            print(f"  ✗ FAIL: Insufficient RAM")
            return False
        
        if self.device == "cuda":
            try:
                import torch
                for i in range(torch.cuda.device_count()):
                    vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    print(f"  GPU {i} VRAM: {vram:.1f} GB (min required: {self.device_info['min_vram_gb']} GB)")
                    if vram < self.device_info['min_vram_gb']:
                        print(f"  ✗ FAIL: GPU {i} has insufficient VRAM")
                        return False
            except Exception as e:
                print(f"  ⚠ Could not detect CUDA devices: {e}")
                return False
        
        print(f"  ✓ System meets requirements\n")
        return True

    def get_training_script(self) -> Path:
        """Get path to training script for this device."""
        script_name = self.device_info["script"]
        script_path = self.training_repo / script_name
        
        if not script_path.exists():
            raise FileNotFoundError(
                f"Training script not found: {script_path}\n"
                f"Make sure openmythos-10b-apple-silicon is cloned to: {self.training_repo}"
            )
        
        return script_path

    def train_round(self, round_id: int, config: Dict) -> Dict:
        """Run training for a round."""
        print(f"\n[Worker {self.worker_id}] Starting training for round {round_id}")
        print(f"  Config: seq_len={config.get('seq_len')}, loops={config.get('train_loops')}")
        
        script = self.get_training_script()
        env = os.environ.copy()
        env["MYTHOS_SEQ_LEN"] = str(config.get("seq_len", self.device_info["seq_len_default"]))
        env["MYTHOS_MICRO_BATCH"] = str(config.get("micro_batch", self.device_info["batch_default"]))
        env["MYTHOS_GRAD_ACCUM"] = str(config.get("grad_accum", self.device_info["grad_accum_default"]))
        env["MYTHOS_TRAIN_LOOPS"] = str(config.get("train_loops", 8))
        env["MYTHOS_DEVICE"] = self.device
        
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                env=env,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            if result.returncode != 0:
                print(f"  ✗ Training failed:")
                print(result.stderr)
                return None
            
            # Parse output (in real impl, read from checkpoint metadata)
            print(f"  ✓ Training completed")
            return {
                "worker_id": self.worker_id,
                "round_id": round_id,
                "device": self.device,
                "script": str(script),
                "status": "success",
            }
        
        except subprocess.TimeoutExpired:
            print(f"  ✗ Training timed out (>1 hour)")
            return None
        except Exception as e:
            print(f"  ✗ Training error: {e}")
            return None

    def show_device_specs(self):
        """Print device specifications."""
        print("\n" + "=" * 80)
        print(f"Device: {self.device_info['name']}")
        print("=" * 80)
        print(f"Minimum RAM: {self.device_info['min_ram_gb']} GB")
        print(f"Recommended RAM: {self.device_info['recommended_ram_gb']} GB")
        print(f"Minimum VRAM: {self.device_info['min_vram_gb']} GB")
        print(f"Supported Models: {', '.join(self.device_info['supported_models'])}")
        print(f"Default seq_len: {self.device_info['seq_len_default']}")
        print(f"Default batch: {self.device_info['batch_default']}")
        print(f"Default grad_accum: {self.device_info['grad_accum_default']}")
        print(f"\nFrameworks: {', '.join(self.device_info['frameworks'])}")
        print(f"\nExample Hardware:")
        for hw in self.device_info["hardware_examples"]:
            print(f"  • {hw}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated training worker with auto-backend selection")
    parser.add_argument("--worker-id", default="contributor_001", help="Worker identifier")
    parser.add_argument("--master-url", default="./master_state", help="Master state dir or URL")
    parser.add_argument("--training-repo", default="../openmythos-10b-apple-silicon", help="Path to training repo")
    parser.add_argument("--device", help="Force device (mps/cuda/cpu); auto-detect if unset")
    parser.add_argument("--show-specs", action="store_true", help="Show device specs and exit")
    parser.add_argument("--verify-only", action="store_true", help="Verify requirements and exit")
    
    args = parser.parse_args()
    
    worker = WorkerContributor(
        worker_id=args.worker_id,
        master_url=args.master_url,
        training_repo=args.training_repo,
        device=args.device,
    )
    
    if args.show_specs:
        worker.show_device_specs()
        sys.exit(0)
    
    if args.verify_only:
        if worker.verify_requirements():
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Verify requirements
    if not worker.verify_requirements():
        print("[Worker] System does not meet requirements")
        sys.exit(1)
    
    print("[Worker] Ready to contribute!")
    print(f"Run: python worker/contrib.py --worker-id {args.worker_id}")
