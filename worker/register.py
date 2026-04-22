#!/usr/bin/env python3
"""
One-liner worker registration for OpenMythos Swarm.

After 'pip install openmythos-swarm', run:

    openmythos-register --worker-id alice --master-state ./master_state

That command:
  1. Generates (or loads) a persistent Ed25519 key in ~/.openmythos-swarm/keys/
  2. Registers the public key in the master's authorized_workers.json
  3. Prints your public key fingerprint for confirmation

Share your fingerprint with the master operator to prove your identity
if the master state is on a remote machine (GitHub, S3, etc.).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Allow running directly without pip install
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import Ed25519Key
from common.deps import ensure_dependencies, auto_install_enabled


def _key_fingerprint(pem: str) -> str:
    """Return a short human-readable SHA256 fingerprint of a PEM public key."""
    raw = pem.strip().encode("utf-8")
    full = hashlib.sha256(raw).hexdigest()
    # Format like SSH fingerprints: groups of 4 hex chars
    return ":".join(full[i : i + 4] for i in range(0, 32, 4))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="openmythos-register",
        description="Register as a worker contributor (generates key + registers with master).",
    )
    parser.add_argument(
        "--worker-id",
        required=True,
        help="Your unique worker ID (e.g. 'alice', 'mac_studio_01')",
    )
    parser.add_argument(
        "--master-state",
        default=None,
        help="Path to master state directory (local). "
             "If omitted, prints public key so you can send it to the master operator.",
    )
    parser.add_argument(
        "--keys-dir",
        default=str(Path.home() / ".openmythos-swarm" / "keys"),
        help="Directory to store your private key (default: ~/.openmythos-swarm/keys/)",
    )
    parser.add_argument(
        "--github",
        default=None,
        help="Your GitHub username, stored as identity metadata alongside your key.",
    )
    parser.add_argument(
        "--email",
        default=None,
        help="Your email address, stored as identity metadata alongside your key.",
    )
    parser.add_argument(
        "--print-public-key",
        action="store_true",
        help="Print full public key PEM and exit (useful for sending to master operator).",
    )
    parser.add_argument(
        "--auto-install-deps",
        action="store_true",
        help="Attempt to auto-install missing Python deps",
    )
    args = parser.parse_args()

    ensure_dependencies(
        {
            "cryptography": "cryptography>=41.0.0",
        },
        auto_install=auto_install_enabled(args.auto_install_deps),
    )

    # ------------------------------------------------------------------ key --
    keys_dir = Path(args.keys_dir)
    keys_dir.mkdir(parents=True, exist_ok=True)
    private_key_path = keys_dir / f"{args.worker_id}_private.pem"

    if private_key_path.exists():
        key = Ed25519Key.load_private_pem(str(private_key_path))
        print(f"[register] Loaded existing key: {private_key_path}")
    else:
        key = Ed25519Key()
        key.save_private_pem(str(private_key_path))
        print(f"[register] Generated new Ed25519 key: {private_key_path}")

    pub_pem = key.public_pem
    fingerprint = _key_fingerprint(pub_pem)

    print(f"\n  Worker ID  : {args.worker_id}")
    print(f"  Fingerprint: {fingerprint}")
    if args.github:
        print(f"  GitHub     : {args.github}")
    if args.email:
        print(f"  Email      : {args.email}")

    if args.print_public_key:
        print("\n--- PUBLIC KEY ---")
        print(pub_pem.strip())
        print("-----------------\n")

    # ---------------------------------------------------------- register -----
    if args.master_state:
        master_state = Path(args.master_state)
        auth_dir = master_state / "auth"
        auth_dir.mkdir(parents=True, exist_ok=True)
        auth_file = auth_dir / "authorized_workers.json"

        registry: dict = {}
        if auth_file.exists():
            try:
                registry = json.loads(auth_file.read_text())
            except Exception:
                registry = {}

        import time
        metadata: dict = {"source": "openmythos-register"}
        if args.github:
            metadata["github"] = args.github
        if args.email:
            metadata["email"] = args.email

        registry[args.worker_id] = {
            "public_key": pub_pem,
            "fingerprint": fingerprint,
            "metadata": metadata,
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        auth_file.write_text(json.dumps(registry, indent=2))
        print(f"\n[register] ✓ Registered '{args.worker_id}' in {auth_file}")
    else:
        print("\n[register] No --master-state provided.")
        print("  To register remotely, send your public key to the master operator:")
        print(f"\n    openmythos-register --worker-id {args.worker_id} --print-public-key\n")
        print("  Then the master operator runs:")
        print(f"    openmythos-master --state-dir ./master_state \\")
        print(f"      --register-worker-id {args.worker_id} \\")
        print(f"      --register-worker-pubkey-file <your_key.pem>\n")

    print("[register] Done. You can now submit training contributions.\n")


if __name__ == "__main__":
    main()
