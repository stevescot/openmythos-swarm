# Worker Setup Guide

## Quick Start

### Mac Studio

```bash
# Clone training repo
git clone https://github.com/stevescot/openmythos-10b-apple-silicon.git
cd openmythos-10b-apple-silicon
pip install -r requirements.txt

# Check hardware
python worker/contrib.py --worker-id "mac_studio_01" --show-specs

# Verify requirements
python worker/contrib.py --worker-id "mac_studio_01" --verify-only

# Join training network
python worker/contrib.py --worker-id "mac_studio_01"
```

### NVIDIA GPU

```bash
# Clone training repo
git clone https://github.com/stevescot/openmythos-10b-apple-silicon.git
cd openmythos-10b-apple-silicon
pip install -r requirements.txt

# Check hardware
python worker/contrib.py --worker-id "gpu_node_01" --show-specs

# Verify requirements
python worker/contrib.py --worker-id "gpu_node_01" --verify-only

# Join training network
python worker/contrib.py --worker-id "gpu_node_01"
```

## Hardware Requirements

### Apple Silicon (Mac Studio)

| Spec | Minimum | Recommended | Production |
|------|---------|-------------|------------|
| **CPU** | M1 Ultra | M2/M3 Ultra | M3 Ultra |
| **RAM** | 64 GB | 256 GB | 512 GB |
| **Storage** | 500 GB | 1 TB | 2 TB |
| **Network** | 100 Mbps | 1 Gbps | 10 Gbps |
| **Power** | 600W | 800W | 1000W |

**Examples:**
- ✅ **Mac Studio M3 Ultra (512GB)** — Ideal, proven
- ✅ **Mac Studio M2 Ultra (384GB)** — Good, tested
- ✅ **Mac Studio M1 Ultra (384GB)** — Works, slower
- ⚠️ **MacBook Pro M3 Max (128GB)** — Marginal, may OOM
- ❌ **iMac M3 (32GB)** — Too small

**Cost:** $7k-$20k for production setup

### NVIDIA GPU

| Spec | Minimum | Recommended | Production |
|------|---------|-------------|------------|
| **GPU VRAM** | 40 GB | 80 GB | 80GB+ (H100) |
| **CPU RAM** | 32 GB | 128 GB | 256 GB+ |
| **Storage** | 500 GB | 1 TB | 2 TB NVMe |
| **CUDA** | 12.0+ | 12.2+ | 12.2+ |
| **Network** | 1 Gbps | 10 Gbps | 40 Gbps |
| **Power** | 800W | 1500W | 2000W+ |

**Examples:**
- ✅✅ **H100 (80GB)** — Ideal, 2-3x faster than Mac
- ✅ **A100 (80GB)** — Good, proven
- ✅ **A100 (40GB)** — Works, slower
- ⚠️ **L40S (48GB)** — Good for training, newer
- ⚠️ **RTX 6000 Ada (48GB)** — Good, expensive
- ❌ **RTX 4090 (24GB)** — Too small

**Cost:** $8k-$50k+ for production setup (rental: $1-5/hour)

## Backend Auto-Detection

The worker automatically selects the right backend:

```python
# worker/contrib.py
worker = WorkerContributor("mac_studio_01")

# Auto-detects:
# - Mac with M-series → uses MPS (10b_apple_silicon.py)
# - NVIDIA GPU → uses CUDA (10b_cross_platform.py)
# - No GPU → uses CPU (slow, for testing)
```

### Force a Specific Backend

```bash
# Force CUDA even if not auto-detected
python worker/contrib.py --worker-id "gpu_01" --device cuda

# Force MPS even if not auto-detected
python worker/contrib.py --worker-id "mac_01" --device mps

# Use CPU (for testing)
python worker/contrib.py --worker-id "test_01" --device cpu
```

## System Specifications

### Check Your Mac

```bash
# CPU and RAM
system_profiler SPHardwareDataType | grep "Chip:\|Memory:"

# Example output:
#   Chip: Apple M3 Ultra
#   Memory: 512 GB
```

### Check Your NVIDIA GPU

```bash
# GPU details
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Example output:
#   NVIDIA H100, 81392 MiB
#   NVIDIA A100, 81396 MiB
```

## Installation

### Mac Studio

```bash
# 1. Install Python 3.10+ (via Homebrew or pyenv)
brew install python@3.11

# 2. Clone repos
git clone https://github.com/stevescot/openmythos-swarm.git
git clone https://github.com/stevescot/openmythos-10b-apple-silicon.git

# 3. Install dependencies
cd openmythos-10b-apple-silicon
pip install -r requirements.txt

# 4. Verify
python -c "import torch; print(torch.backends.mps.is_available())"  # Should print True
```

### NVIDIA GPU

```bash
# 1. Install CUDA 12.0+ and cuDNN
#    https://developer.nvidia.com/cuda-downloads

# 2. Install Python 3.10+
python --version  # Should be 3.10+

# 3. Clone repos
git clone https://github.com/stevescot/openmythos-swarm.git
git clone https://github.com/stevescot/openmythos-10b-apple-silicon.git

# 4. Install dependencies (PyTorch will use CUDA automatically)
cd openmythos-10b-apple-silicon
pip install -r requirements.txt

# 5. Verify
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

## Running a Worker

### Step 1: Check Your Hardware

```bash
cd openmythos-10b-apple-silicon
python worker/contrib.py --worker-id "your_id" --show-specs
```

Output:
```
================================================================================
Device: Apple Silicon (MPS)
================================================================================
Minimum RAM: 64 GB
Recommended RAM: 256 GB
Minimum VRAM: 0 GB
Supported Models: 10b
Default seq_len: 4096
Default batch: 2
Default grad_accum: 16

Frameworks: PyTorch 2.0+

Example Hardware:
  • Mac Studio M1 Ultra (384 GB RAM) ✓
  • Mac Studio M2 Ultra (512 GB RAM) ✓
  • Mac Studio M3 Ultra (512 GB RAM) ✓
  • MacBook Pro M3 Max (128 GB RAM) ✓ (slow)
  • iMac M3 (32 GB RAM) ~ (marginal)
================================================================================
```

### Step 2: Verify Requirements

```bash
python worker/contrib.py --worker-id "your_id" --verify-only
```

Output:
```
[Worker your_id] Device: Apple Silicon (MPS)
  Min RAM: 64 GB
  Min VRAM: 0 GB
  Script: 10b_apple_silicon.py

[Worker your_id] System Check
  Available RAM: 512.0 GB (min required: 64 GB)
  ✓ System meets requirements
```

### Step 3: Point to Master

```bash
# Option A: Local master (if running on same network)
python worker/contrib.py \
    --worker-id "mac_studio_01" \
    --master-url ./master_state

# Option B: Remote master (via HTTP)
python worker/contrib.py \
    --worker-id "mac_studio_01" \
    --master-url https://master.example.com:5000
```

### Step 4: Join Training

```bash
python worker/contrib.py --worker-id "mac_studio_01"
```

The worker will:
1. ✓ Auto-detect your device (MPS or CUDA)
2. ✓ Fetch training task from master
3. ✓ Download dataset
4. ✓ Run training loop
5. ✓ Submit gradients to master
6. ✓ Wait for next task

## Monitoring Training

### Mac Studio

```bash
# In separate terminal, watch resource usage
watch -n 1 'top -l1 | head -15'  # CPU + RAM
# Or use Activity Monitor GUI
```

### NVIDIA GPU

```bash
# In separate terminal, watch GPU
watch nvidia-smi

# Or detailed monitoring
nvidia-smi dmon  # Persistent stats
```

## Troubleshooting

### "Insufficient RAM"

```
[Worker] System Check
  Available RAM: 32.0 GB (min required: 64 GB)
  ✗ FAIL: Insufficient RAM
```

**Solution:**
- Upgrade RAM to at least 64 GB (recommended 256+ GB)
- Or contribute on a different machine

### "No CUDA devices found" (NVIDIA)

```
[Worker] Device: NVIDIA CUDA GPU
  ✗ FAIL: GPU has insufficient VRAM
```

**Solution:**
- Ensure CUDA 12.0+ is installed: `nvidia-smi`
- Reinstall PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

### "MPS not available" (Mac)

```
[Worker] Device: Apple Silicon (MPS)
  ✗ FAIL: MPS backend not detected
```

**Solution:**
- Ensure Python 3.10+ on Apple Silicon Mac
- Reinstall PyTorch: `pip install --upgrade torch`

### Training too slow

**On Mac:**
- Lower seq_len: `MYTHOS_SEQ_LEN=2048`
- Lower batch: `MYTHOS_MICRO_BATCH=1`
- Disable gradient checkpointing if MPS overhead

**On NVIDIA:**
- Increase batch size: `MYTHOS_MICRO_BATCH=16`
- Use larger seq_len: `MYTHOS_SEQ_LEN=8192`

### Out of Memory (OOM)

**On Mac:**
```bash
# Reduce batch size
MYTHOS_MICRO_BATCH=1 MYTHOS_GRAD_ACCUM=8 python 10b_apple_silicon.py
```

**On NVIDIA:**
```bash
# Reduce batch size or seq_len
MYTHOS_MICRO_BATCH=4 MYTHOS_SEQ_LEN=4096 python 10b_cross_platform.py
```

## Advanced: Custom Config

Override training parameters:

```bash
# Mac: small test
MYTHOS_SEQ_LEN=512 \
MYTHOS_MICRO_BATCH=1 \
MYTHOS_GRAD_ACCUM=1 \
MYTHOS_TRAIN_LOOPS=4 \
python worker/contrib.py --worker-id "test_01"

# NVIDIA: aggressive training
MYTHOS_SEQ_LEN=8192 \
MYTHOS_MICRO_BATCH=16 \
MYTHOS_GRAD_ACCUM=2 \
MYTHOS_TRAIN_LOOPS=24 \
python worker/contrib.py --worker-id "gpu_prod_01"
```

## Next Steps

1. **Check hardware specs:** Run `--show-specs`
2. **Verify requirements:** Run `--verify-only`
3. **Get master URL:** Ask coordinator for `master_url`
4. **Start training:** Run worker with `--master-url`
5. **Monitor:** Watch logs and resource usage
6. **Scale:** Add more workers as needed

## Support

- **Issues?** Open GitHub issue with:
  - Output of `--show-specs`
  - Output of `--verify-only`
  - Error messages from training run
- **Questions?** Join community discussion

---

**Remember:** The federated network works best with diverse hardware! Both Mac and NVIDIA contributors are welcome.

