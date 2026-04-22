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

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import Ed25519Key
from common.deps import ensure_dependencies, auto_install_enabled


def detect_device() -> str:
    """Auto-detect compute device.

    Returns:
      - "rocm" for AMD ROCm builds
      - "cuda" for NVIDIA CUDA builds
      - "mps" for Apple Silicon
      - "cpu" fallback
    """
    try:
        import torch

        if torch.cuda.is_available():
            # ROCm builds expose CUDA-compatible API but set torch.version.hip
            if getattr(torch.version, "hip", None):
                return "rocm"
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
        "rocm": {
            "name": "AMD ROCm GPU",
            "min_ram_gb": 32,
            "recommended_ram_gb": 128,
            "min_vram_gb": 40,
            "supported_models": ["10b"],
            "script": "10b_cross_platform.py",
            "seq_len_default": 8192,
            "batch_default": 8,
            "grad_accum_default": 4,
            "frameworks": ["PyTorch ROCm", "ROCm 6.0+"],
            "hardware_examples": [
                "AMD Instinct MI210 (64GB) ✓✓ (ideal)",
                "AMD Instinct MI250 (128GB) ✓✓ (ideal)",
                "AMD Instinct MI300X (192GB) ✓✓✓ (best)",
                "AMD Radeon PRO W7900 (48GB) ✓ (good)",
                "AMD Radeon RX 7900 XTX (24GB) ~ (marginal, ROCm support varies)",
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

        self.keys_dir = Path.home() / ".openmythos-swarm" / "keys"
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        self.private_key_path = self.keys_dir / f"{self.worker_id}_private.pem"
        self.worker_key = self._load_or_create_worker_key()
        self.worker_public_key = self.worker_key.public_pem
        
        print(f"[Worker {worker_id}] Device: {self.device_info['name']}")
        print(f"  Min RAM: {self.device_info['min_ram_gb']} GB")
        print(f"  Min VRAM: {self.device_info['min_vram_gb']} GB")
        print(f"  Script: {self.device_info['script']}")

    def _load_or_create_worker_key(self) -> Ed25519Key:
        if self.private_key_path.exists():
            return Ed25519Key.load_private_pem(str(self.private_key_path))
        key = Ed25519Key()
        key.save_private_pem(str(self.private_key_path))
        return key

    def verify_requirements(self) -> bool:
        """Check if system meets minimum requirements."""
        import psutil
        
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"\n[Worker {self.worker_id}] System Check")
        print(f"  Available RAM: {ram_gb:.1f} GB (min required: {self.device_info['min_ram_gb']} GB)")
        
        if ram_gb < self.device_info['min_ram_gb']:
            print(f"  ✗ FAIL: Insufficient RAM")
            return False
        
        if self.device in {"cuda", "rocm"}:
            try:
                import torch
                if self.device == "rocm" and not getattr(torch.version, "hip", None):
                    print("  ✗ FAIL: ROCm backend requested but this PyTorch build is not ROCm-enabled")
                    return False
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
        # cross-platform trainer uses torch.device("cuda") for both CUDA and ROCm
        env["MYTHOS_DEVICE"] = "cuda" if self.device in {"cuda", "rocm"} else self.device
        env["MYTHOS_BACKEND"] = self.device
        
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated training worker with auto-backend selection")
    parser.add_argument("--worker-id", default="contributor_001", help="Worker identifier")
    parser.add_argument("--master-url", default="./master_state", help="Master state dir or URL")
    parser.add_argument("--training-repo", default="../openmythos-10b-apple-silicon", help="Path to training repo")
    parser.add_argument("--device", help="Force device (mps/cuda/rocm/cpu); auto-detect if unset")
    parser.add_argument("--show-specs", action="store_true", help="Show device specs and exit")
    parser.add_argument("--verify-only", action="store_true", help="Verify requirements and exit")
    parser.add_argument("--print-public-key", action="store_true", help="Print worker public key and exit")
    parser.add_argument("--auto-install-deps", action="store_true", help="Attempt to auto-install missing Python deps")
    
    args = parser.parse_args()

    ensure_dependencies(
        {
            "cryptography": "cryptography>=41.0.0",
            "torch": "torch>=2.0.0",
            "numpy": "numpy>=1.26.0",
            "psutil": "psutil>=5.9.0",
        },
        auto_install=auto_install_enabled(args.auto_install_deps),
    )
    
    worker = WorkerContributor(
        worker_id=args.worker_id,
        master_url=args.master_url,
        training_repo=args.training_repo,
        device=args.device,
    )
    
    if args.show_specs:
        worker.show_device_specs()
        return

    if args.print_public_key:
        print(worker.worker_public_key)
        return
    
    if args.verify_only:
        if worker.verify_requirements():
            return
        sys.exit(1)
    
    # Verify requirements
    if not worker.verify_requirements():
        print("[Worker] System does not meet requirements")
        sys.exit(1)
    
    print("[Worker] Ready to contribute!")
    print(f"Run: openmythos-worker --worker-id {args.worker_id}")


if __name__ == "__main__":
    main()
