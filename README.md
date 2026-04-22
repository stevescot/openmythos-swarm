# OpenMythos Swarm — Federated Training on Mac Studios

Decentralized, volunteer-driven training for the [OpenMythos](https://github.com/open-mythos/openmythos) 10B language model on Mac Silicon.

> **Note:** This is a complementary project to [OpenMythos](https://github.com/open-mythos/openmythos), designed to enable distributed collaborative training on Mac Studios. It is not a fork of OpenMythos itself, but rather a federated training coordinator that integrates with OpenMythos model code.

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
         │  (mac_1)   │   │ (mac_2)   │   │ (mac_3)   │
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
│   └── __init__.py
├── master/              # Coordinator server
│   ├── server.py        # MasterCoordinator class
│   └── __init__.py
├── worker/              # Worker client
│   ├── client.py        # WorkerClient class
│   └── __init__.py
├── tests/               # Integration tests
│   └── test_basic.py    # End-to-end round test
├── README.md            # This file
└── requirements.txt     # Dependencies
```

## Setup

### Prerequisites

- Python 3.10+
- `cryptography` library (for Ed25519)

### Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your-org/openmythos-swarm.git
   cd openmythos-swarm
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Initialize Master

```bash
python master/server.py
```

This will:
- Generate Ed25519 key pair
- Store state in `./master_state/`
- Publish round 1 specification

### 2. Run Worker(s)

In another terminal:

```bash
python worker/client.py
```

This will:
- Load master's public key
- Fetch round 1 spec
- Simulate training (5 steps)
- Submit results to master

### 3. Run Integration Test

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
[Master] Aggregated round 1: 3 valid submissions
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
   - Submit to master's `submissions/` directory

5. **Master aggregates**
   - Collect all valid submissions
   - Apply robust aggregation (clipping, outlier removal)
   - Compute new global checkpoint
   - Publish signed RoundResult

6. **Next round**
   - Master publishes new RoundSpec with `prior_checkpoint_hash` = aggregated result

## Security & Trust

### What Master Controls

- Private signing key (cannot be stolen by workers)
- Canonical round specifications
- Global checkpoint lineage (hash-chained across rounds)

### What Master Verifies

- Worker submissions: valid signatures, reasonable gradients
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
python master/server.py --output-dir s3://my-bucket/rounds --poll-interval 300

# Worker (runs on volunteer Mac)
python worker/client.py --master-url https://raw.githubusercontent.com/you/openmythos-swarm/main/rounds
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

- **[OpenMythos](https://github.com/open-mythos/openmythos)** — The 10B language model being trained collaboratively by this swarm
- **[Flower](https://flower.ai/)** — Production federated learning framework (consider migrating to for scale)
- **[NVFlare](https://github.com/NVIDIA/NVFlare)** — Enterprise federated learning platform

This project is a volunteer-driven **federated training coordinator** for OpenMythos, not a fork. It depends on OpenMythos model code and tokenizer.

## Contact

Questions? Open an issue or reach out to the maintainers.

---

**Status**: Alpha (v0.1.0)  
**Last updated**: 2026-04-22
