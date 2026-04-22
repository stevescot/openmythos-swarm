# OpenMythos Swarm — Federated Training on Mac, NVIDIA, and AMD

Decentralized, volunteer-driven training for the [OpenMythos](https://github.com/kyegomez/OpenMythos/tree/main) 10B language model on Apple Silicon, NVIDIA CUDA, and AMD ROCm GPUs.

> **Note:** This is a complementary project to [OpenMythos](https://github.com/kyegomez/OpenMythos/tree/main), designed to enable distributed collaborative training. It is not a fork of OpenMythos itself, but rather a federated training coordinator that integrates with OpenMythos model code.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Master Coordinator                      │
│  (owns keys, publishes rounds, aggregates submissions)       │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
         ┌──────────▼─┐   ┌──▼────────┐   ┌──▼────────┐
         │   Worker   │   │  Worker   │   │  Worker   │
         │  (mac_1)   │   │(Nvidia_2) │   │ (mac_3)   │
         │            │   │           │   │           │
         │  - Fetch   │   │  - Fetch  │   │  - Fetch  │
         │  - Train   │   │  - Train  │   │  - Train  │
         │  - Submit  │   │  - Submit │   │  - Submit │
         └────────────┘   └───────────┘   └───────────┘
```

### Design Principles

1. **One Master, Many Contributors**
   - Master holds signing keys and authorizes rounds
   - Workers are untrusted volunteers (can fail, lag, submit bad updates)
   - Robust aggregation filters and combines worker updates

2. **Published Entities**
   - Round specs, submissions, and aggregated results are signed JSON manifests
   - Can be published to any durable store (GitHub, S3, IPFS, etc.)

3. **Hierarchical Aggregation (Future)**
   - Combiner nodes can aggregate N workers, then submit to master
   - Reduces network traffic and improves robustness

## Project Structure

```
openmythos-swarm/
├── common/              # Shared utilities
│   ├── crypto.py        # Ed25519 signing, HMAC verification
│   ├── manifest.py      # RoundSpec, WorkerSubmission, RoundResult
│   ├── __init__.py
├── master/              # Coordinator server
│   ├── server.py        # MasterCoordinator class
│   ├── scheduler.py     # ⭐ Auto-scheduler
│   └── __init__.py
├── worker/              # Worker client
│   ├── contrib.py       # ⭐ Auto-detecting worker
│   ├── client.py        # WorkerClient class
│   ├── __init__.py
├── tests/               # Integration tests
│   └── test_basic.py    # End-to-end round test
├── README.md            # This file
├── WORKER_SETUP.md
├── DEPLOYMENT.md
├── GITHUB_SETUP.md
└── requirements.txt     # Dependencies
```

## Supported Hardware & Backend Selection

The swarm auto-detects your device and uses the optimal backend:

| Hardware | Backend | Script | Status | Specs |
|----------|---------|--------|--------|-------|
| **Mac M-series** | PyTorch MPS | `10b_apple_silicon.py` | ✅ Tested | 64GB+ RAM |
| **NVIDIA GPU** | PyTorch CUDA | `10b_cross_platform.py` | ✅ Supported | 40GB+ VRAM |
| **AMD GPU (ROCm)** | PyTorch ROCm | `10b_cross_platform.py` | ✅ Supported (Linux ROCm) | 40GB+ VRAM |
| **Mixed (Mac + NVIDIA + AMD)** | Auto-select | Both | ✅ Compatible | Diverse |

**Minimum Requirements:**
- **Mac Studio**: 64 GB unified memory (256GB recommended)
- **NVIDIA GPU**: 40 GB VRAM (80GB H100 ideal)
- **AMD GPU (ROCm)**: 40 GB VRAM (MI210/MI250/MI300 preferred)
- **CPU**: Works for testing only (not practical)

**[See WORKER_SETUP.md for detailed hardware specs, installation, and troubleshooting](WORKER_SETUP.md)**

## Setup

### Prerequisites

- Python 3.10+

### Installation (pip)

1. Clone the repo:
   ```bash
   git clone https://github.com/stevescot/openmythos-swarm.git
   cd openmythos-swarm
   ```

2. Install with pip (recommended):
   ```bash
   pip install .
   ```

   For editable development install:
   ```bash
   pip install -e .
   ```

3. Optional: CLI commands are installed automatically:
   - `openmythos-master`
   - `openmythos-scheduler`
   - `openmythos-worker`
   - `openmythos-register`
   - `openmythos-eth-receipt`
   - `openmythos-eth-anchor`

   > On Windows, these may install under `%APPDATA%\Python\Python313\Scripts`. Add that directory to `PATH` if commands are not found.

## Quick Start

### 1. Initialize Master

```bash
openmythos-master --state-dir ./master_state --publish-demo
```

This will:
- Generate Ed25519 key pair
- Store state in `./master_state/`
- Publish round 1 specification

### 2. Run Worker(s)

**[See WORKER_SETUP.md for detailed hardware requirements and setup](WORKER_SETUP.md)**

In another terminal:

```bash
# Mac Studio (auto-detects MPS)
openmythos-worker --worker-id "mac_studio_01"

# NVIDIA GPU (auto-detects CUDA)
openmythos-worker --worker-id "gpu_node_01"

# AMD ROCm GPU (auto-detects ROCm)
openmythos-worker --worker-id "amd_node_01"

# Check your hardware specs first
openmythos-worker --worker-id "contributor_01" --show-specs
```

This will:
- Auto-detect your device (Apple Silicon MPS, NVIDIA CUDA, or AMD ROCm)
- Load master's public key
- Use a persistent worker Ed25519 key for signed attestations
- Fetch round spec
- Run actual training (or simulated if testing)
- Submit results to master

If dependencies are missing on a contributor machine, you can opt-in to automatic install:

```bash
# one-shot
openmythos-worker --worker-id "contributor_01" --auto-install-deps

# or via environment variable
OPENMYTHOS_AUTO_INSTALL_DEPS=1 openmythos-worker --worker-id "contributor_01"
```

**Supported devices:**
- ✅ **Mac Studio** (M1/M2/M3 Ultra, 64GB+ RAM)
- ✅ **NVIDIA GPU** (H100/A100/RTX, 40GB+ VRAM)
- ✅ **AMD ROCm GPU** (MI210/MI250/MI300 or ROCm-supported Radeon PRO)
- ✅ **Mixed networks** (Mac + NVIDIA + AMD together)

### 3. Run Auto-Scheduler (Continuous)

For production, run the auto-scheduler to publish rounds automatically:

```bash
# Stabilization: 5 rounds × 5M tokens each
openmythos-scheduler --config stabilization --max-rounds 5

# Production: 100 rounds × 100M tokens each  
openmythos-scheduler --config production --max-rounds 100

# Mixed (recommended): stabilization then production
openmythos-scheduler --config mixed --workers 5 --interval 3600 --submission-wait 1800
```

This runs in a loop:
1. Publishes a new round spec
2. Waits for worker submissions (30 min default)
3. Aggregates results into next round
4. Repeats

**Args:**
```
--state-dir       Master state directory (default: ./master_state)
--workers N       Expected workers per round (default: 3)
--interval SECS   Total time per round (default: 3600 = 1hr)
--submission-wait SECS  Time to wait for submissions (default: 1800 = 30min)
--max-rounds N    Stop after N rounds (default: infinite)
--config STRATEGY stabilization | production | mixed (default: mixed)
```

### 3.5 Micro-rounds: completing a step in hours, not days

By default a round is large (100M–500M tokens). For contributors on a normal home connection or consumer GPU, a **micro-round** configuration breaks training into 2–4 hour chunks. Multiple contributors each complete one micro-round and the master chains them together.

#### Bandwidth per round — what actually moves

| Artifact | Size | Direction | Notes |
|---|---|---|---|
| Round spec (JSON) | ~2 KB | ↓ download | Tiny — config + hashes |
| Dataset shard | 10–100 MB | ↓ download | Only once per shard, re-used across rounds |
| Gradient delta (`.pt`) | **~40 GB** (full, fp32) | ↑ upload | 10B model × 4 bytes/param |
| Gradient delta (compressed/quantized) | **~5–10 GB** | ↑ upload | bf16 or int8 delta compression |
| Round result manifest | ~2 KB | ↓ download | Signed result hash |

**On a 100 Mbps home upload:** a full fp32 delta takes ~55 minutes. With bf16 compression: ~7–14 minutes.

#### Practical micro-round sizes

| Mode | `target_tokens` | `train_loops` | Estimated time (Mac M3 Ultra) | Upload size (bf16 delta) |
|---|---|---|---|---|
| **Quick smoke test** | 500K | 2 | 10–20 min | ~5 GB |
| **Micro-round** | 5M | 4 | 1–2 hrs | ~5 GB |
| **Standard round** | 50M | 8 | 4–8 hrs | ~5 GB |
| **Production round** | 500M | 24 | 2–3 days | ~5 GB |

> **Note:** Delta size is determined by model size (10B params), not by `target_tokens`. Compression is the key lever for upload time. Longer rounds just train more steps before the upload happens.

#### Recommended micro-round config for home contributors

```python
# ~1–2 hour round, ~5 GB upload (bf16 delta)
TrainingConfig(
    seq_len=512,          # shorter sequences = more steps per GB
    micro_batch=1,
    grad_accum=4,
    train_loops=4,        # 4 inner loops per round
    learning_rate=2e-4,
    weight_decay=0.1,
    target_tokens=5_000_000,  # 5M tokens per round
)
```

Schedule this with:

```bash
openmythos-scheduler \
  --config stabilization \
  --workers 3 \
  --interval 7200 \
  --submission-wait 3600 \
  --max-rounds 20
```

Each round closes every 2 hours. 20 rounds × 5M tokens = 100M tokens total — a meaningful contribution on a standard broadband connection.

#### How chaining works

```
Round 1 (contributor A, 5M tokens)
   └→ aggregated_delta_1.pt + checkpoint_hash_1
Round 2 (contributor B, 5M tokens) ← prior_checkpoint_hash = checkpoint_hash_1
   └→ aggregated_delta_2.pt + checkpoint_hash_2
Round 3 (contributor A+C, 5M tokens) ← prior_checkpoint_hash = checkpoint_hash_2
   └→ ...
```

Contributors do **not** need to hold the full model checkpoint locally — only their delta for that round. The master chains them via hashes.

### 3.5 Register Worker Identity (one-liner after `pip install`)

Workers sign every submission with a persistent Ed25519 private key.
After `pip install openmythos-swarm`, register yourself in a single command:

```bash
# Local master (same machine or shared directory)
openmythos-register --worker-id alice --master-state ./master_state

# With optional identity metadata (GitHub / email)
openmythos-register --worker-id alice --master-state ./master_state \
  --github alice-gh --email alice@example.com

# Remote master — print your public key and send it to the master operator
openmythos-register --worker-id alice --print-public-key

# Master operator registers the key they received
openmythos-master --state-dir ./master_state \
  --register-worker-id alice \
  --register-worker-pubkey-file alice_pub.pem

# List all registered contributors
openmythos-master --state-dir ./master_state --list-workers
```

Keys are stored in `~/.openmythos-swarm/keys/` and re-used across rounds.
If an allowlist exists, submissions are accepted only when `worker_id` and public key match the registered record.

### 3.6 zk-eth runbook (receipt export + anchor payload)

On the `zk-eth` branch, you can export finalized round receipts and prepare Ethereum anchor payloads:

```bash
# 1) Finalize a round first (normal flow)
openmythos-master --state-dir ./master_state --status

# 2) Export anchor-ready receipts from finalized round N
openmythos-master --state-dir ./master_state \
   --export-round-receipts N \
   --credits-per-step 2

# 3) Prepare anchor calldata payload from one exported receipt
openmythos-eth-anchor \
   --receipt-file ./master_state/rounds/round_000N/receipts/<worker>_receipt.json \
   --worker-address 0x0000000000000000000000000000000000000000 \
   --credits 100
```

The exported receipt contains:
- deterministic `receipt_payload`
- `receipt_hash` (SHA256 canonical hash, hex)
- suggested `credits`

These values are the inputs for `anchorReceipt(...)` on `ZkPotContributionRegistry.sol`.

### 3.7 How contribution tracking and rewards are managed

This project uses a two-layer accounting model:

1. **Off-chain verified work receipts** (already live in this repo)
2. **On-chain anchoring / reward claiming** (zk-eth branch)

#### Identity model (who did the work)

- Every worker has a persistent **Ed25519 keypair** (`openmythos-register` / worker key files).
- Submissions include a signed attestation payload with:
  - `worker_id`, `round_id`
  - `dataset_manifest_hash`
  - `delta_hash`
  - challenge/receipt fields (`challenge_*`, `work_receipt_hash`)
- Master verifies signature + allowlist binding (`worker_id` ↔ public key).

This is the canonical off-chain contribution identity.

#### Work tracking model (what was done)

- For each finalized round, master can export deterministic receipt files using:
  - `openmythos-master --export-round-receipts N --credits-per-step X`
- Each receipt includes:
  - `receipt_payload`
  - `receipt_hash`
  - suggested `credits`

Receipts are reproducible and auditable from round artifacts.

#### Reward model (how contributors get paid)

- **Tokenless mode (recommended pre-launch):**
  - generate/export receipts
  - prepare anchor payloads
  - no token deployment required yet
- **On-chain mode (when ready):**
  - anchor receipt hashes to `ZkPotContributionRegistry`
  - then claim/mint reward credits/tokens per policy

#### Gas payment model

You can run a self-pay claim model:

- contributor uses their ETH address to submit claim tx and pay gas
- the same ETH address can be treated as payout identity

For production, use signed claim payloads (EIP-712 style) with at least:

- `recipient_address`
- `amount`
- `nonce`
- `expiry`
- `chain_id`
- `contract_address`

This prevents replay and binds rewards to the intended claimant.

#### Operational lifecycle (end-to-end)

1. Worker trains and submits signed attestation
2. Master verifies and finalizes round
3. Master exports receipt files + receipt hash
4. Operator prepares anchor payload (`openmythos-eth-anchor`)
5. (Optional) Broadcast anchor tx on testnet/mainnet
6. Contributor claims reward according to policy

This gives a clear path from verifiable work to accountable rewards.

### 4. Run Integration Test

```bash
python tests/test_basic.py
```

Expected output:
```
================================================================================
FEDERATED TRAINING INTEGRATION TEST
================================================================================

[TEST] Initializing master coordinator...
[Master] Initialized at ./test_master_state
[Master] Public key (PEM):
-----BEGIN PUBLIC KEY-----
...
[TEST] ✓ Master initialized

[TEST] Publishing round 1...
[Master] Published round 1
  Config: seq_len=1024, loops=8
  Expected workers: 3
[TEST] ✓ Round published successfully

[TEST] Initializing workers...
[TEST] ✓ 3 workers initialized

[TEST] Workers fetching round spec...
[Worker mac_studio_00] Fetched round 1 spec
...
[TEST] ✓ All workers fetched spec

[TEST] Workers training locally...
[TEST] ✓ All workers completed training

[TEST] Workers submitting results...
[TEST] ✓ All workers submitted results

[TEST] Master aggregating round...
[Master] Finalized round 1
  Valid submissions: 3/3
  Avg loss: 9.6000
  Avg grad norm: 25.0000
  Total weight steps: 15
  Tensor contributors: 3
  Merged tensors: 2
[TEST] ✓ Round aggregated successfully

[TEST] Verifying manifest structure...
[TEST] ✓ Manifest structure valid

================================================================================
ALL TESTS PASSED ✓
================================================================================
```

## How It Works

### Round Lifecycle

1. **Master publishes RoundSpec**
   - Specifies training config, dataset, prior checkpoint hash
   - Signed with master's private key
   - Stored in `master_state/rounds/round_XXXX/spec.json`

2. **Workers fetch spec**
   - Verify master's signature
   - Load configuration
   - Fetch dataset shard

3. **Workers train locally**
   - Run training for N steps
   - Compute gradient deltas
   - Save results locally

4. **Workers submit**
   - Create WorkerSubmission with training stats
   - Include signed attestation payload (`worker_id`, `round_id`, `dataset_manifest_hash`, `delta_hash`, challenge receipt fields)
   - Submit to master's `submissions/` directory

5. **Master aggregates**
   - Collect all valid submissions
   - Verify worker attestation signature with worker public key
   - Verify worker identity binding against optional allowlist registry
   - Verify submitted delta file hash matches declared `delta_hash`
   - Verify challenge-bound receipt hash (`work_receipt_hash`) for contribution integrity
   - Load submitted tensor deltas (`.pt`) from workers
   - Compute weighted tensor merge (`steps_completed` as weights)
   - Save `aggregated_delta.pt` and hash it as canonical round artifact
   - Publish signed RoundResult
   - Optional: export Ethereum anchor-ready receipts for reward pipeline

6. **Next round**
   - Master publishes new RoundSpec with `prior_checkpoint_hash` = aggregated result

## Security & Trust

### What Master Controls

- Private signing key (cannot be stolen by workers)
- Canonical round specifications
- Global checkpoint lineage (hash-chained across rounds)

### What Master Verifies

- Worker submissions: valid worker signatures, identity binding, and delta hash integrity
- Aggregation: outlier detection, norm clipping
- Reproducibility: deterministic shard assignments via RNG seed

### Future Enhancements

- **Multi-signature masters** (N-of-M threshold)
- **Client reputation scores** (penalize bad updates)
- **Privacy-preserving aggregation** (differential privacy, secure multiparty compute)
- **Decentralized governance** (voting on round parameters)

## Deployment Options

### Option 1: Single Master + Volunteer Workers

Master runs on always-on node (e.g., your Mac Studio).  
Workers submit via published manifests to shared storage (GitHub, S3).

```bash
# Master (runs continuously)
openmythos-scheduler --state-dir ./master_state --config mixed --workers 5

# Worker (runs on volunteer node)
openmythos-worker --worker-id contributor_01 --master-url ./master_state
```

### Option 2: Hierarchical (Future)

Combiner nodes aggregate regional workers:

```
          Master
           │
      ┌────┼────┐
  Combiner  Combiner  Combiner
    │          │         │
  Worker   Worker   Worker  ...
  Worker   Worker
```

Each combiner submits aggregated delta to master.

### Option 3: Fully Decentralized (Future)

Use Nostr/Matrix for coordination, IPFS for artifact storage.  
Master role becomes an elected role or multisig contract.

## Testing

Run the integration test:

```bash
python tests/test_basic.py
```

Or test individual components:

```python
from common import Ed25519Key, TrainingConfig
from master.server import MasterCoordinator
from worker.client import WorkerClient

# Create master
master = MasterCoordinator()

# Publish round
config = TrainingConfig(...)
manifest = master.publish_round(...)

# Create worker
worker = WorkerClient("mac_studio_01")
spec = worker.fetch_round_spec(1)
result = worker.train_locally()
worker.submit_results(result)
```

## Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a feature branch
3. Add tests for new code
4. Submit a pull request

Areas to help:

- Robust aggregation strategies (trimmed mean, coordinate-wise median)
- Storage backends (S3, IPFS, GitHub)
- Worker failure recovery
- Performance optimization
- Documentation

## License

MIT

## References

- [Federated Learning: Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) — McMahan et al.
- [FedAdam: A Robust Adaptive Aggregation Algorithm for Federated Learning](https://arxiv.org/abs/2003.00295) — Reddi et al.
- [Byzantine-Robust Distributed Learning via Gradient Compression](https://arxiv.org/abs/1902.06932) — Bernstein et al.
- [Flower: A Federated Learning Framework](https://flower.ai/) — Open source FL framework

## Related Projects

- **[OpenMythos](https://github.com/kyegomez/OpenMythos/tree/main)** — The 10B language model being trained collaboratively by this swarm
- **[Flower](https://flower.ai/)** — Production federated learning framework (consider migrating to for scale)
- **[NVFlare](https://github.com/NVIDIA/NVFlare)** — Enterprise federated learning platform

This project is a volunteer-driven **federated training coordinator** for OpenMythos, not a fork. It depends on OpenMythos model code and tokenizer.

## Contact

Questions? Open an issue or reach out to the maintainers.

---

**Status**: Alpha (v0.1.0)  
**Last updated**: 2026-04-22
