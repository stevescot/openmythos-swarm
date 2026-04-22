"""Runtime dependency bootstrap helpers.

This module provides opt-in dependency installation for CLI entrypoints.
It is intentionally conservative: nothing is auto-installed unless explicitly requested.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from typing import Dict, List


def _missing_modules(module_to_package: Dict[str, str]) -> List[str]:
    missing: List[str] = []
    for module in module_to_package:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    return missing


def ensure_dependencies(module_to_package: Dict[str, str], auto_install: bool = False) -> None:
    """Ensure required modules are importable.

    Args:
        module_to_package: map of import module name -> pip package spec
        auto_install: if True, attempt to install missing packages with pip

    Raises:
        RuntimeError: when dependencies are missing and cannot be installed.
    """
    missing = _missing_modules(module_to_package)
    if not missing:
        return

    packages = sorted({module_to_package[m] for m in missing})

    if not auto_install:
        raise RuntimeError(
            "Missing Python dependencies: "
            + ", ".join(missing)
            + "\nInstall with: "
            + f"{sys.executable} -m pip install "
            + " ".join(packages)
            + "\nOr rerun with --auto-install-deps (or OPENMYTHOS_AUTO_INSTALL_DEPS=1)."
        )

    # First attempt: user install (safe default)
    cmd = [sys.executable, "-m", "pip", "install", "--user", *packages]
    first = subprocess.run(cmd, capture_output=True, text=True)
    if first.returncode == 0:
        return

    # Retry for PEP 668 externally managed envs (macOS/Homebrew etc.)
    stderr_lower = (first.stderr or "").lower()
    if "externally-managed-environment" in stderr_lower:
        retry_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--user",
            "--break-system-packages",
            *packages,
        ]
        retry = subprocess.run(retry_cmd, capture_output=True, text=True)
        if retry.returncode == 0:
            return
        raise RuntimeError(
            "Automatic dependency installation failed (PEP668 retry also failed).\n"
            f"Command: {' '.join(retry_cmd)}\n"
            f"stderr:\n{retry.stderr[-4000:]}"
        )

    raise RuntimeError(
        "Automatic dependency installation failed.\n"
        f"Command: {' '.join(cmd)}\n"
        f"stderr:\n{(first.stderr or '')[-4000:]}"
    )


def auto_install_enabled(flag_value: bool) -> bool:
    """Return whether auto install is enabled by arg or env var."""
    if flag_value:
        return True
    env = os.getenv("OPENMYTHOS_AUTO_INSTALL_DEPS", "").strip().lower()
    return env in {"1", "true", "yes", "on"}
