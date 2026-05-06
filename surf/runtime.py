"""
Global runtime configuration: device selection, AMP settings, provenance logging,
and shared utility helpers.
"""

import subprocess
import datetime
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np


@dataclass
class RuntimeConfig:
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    use_amp: bool = False
    amp_dtype: torch.dtype = torch.float32


_RUNTIME: Optional[RuntimeConfig] = None


def setup(device="auto") -> RuntimeConfig:
    """Create and store the global runtime config.

    Mirrors the DEVICE / USE_AMP / AMP_DTYPE logic from main.py.
    """
    global _RUNTIME

    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    use_amp = dev.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    if use_amp:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Using device: {dev}")

    _RUNTIME = RuntimeConfig(device=dev, use_amp=use_amp, amp_dtype=amp_dtype)
    return _RUNTIME


def get() -> RuntimeConfig:
    """Return the current runtime config, auto-setting up if not initialized."""
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = setup()
    return _RUNTIME


def log_provenance():
    """Print git commit, timestamp, and CLI args at the start of every run.

    Helps diagnose which code version produced which results file, so we
    don't conflate runs across commits (e.g. EMD vs Sinkhorn, old vs new
    score net, etc.).
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        short = commit[:10]
    except Exception:
        commit = "<not a git repo>"
        short = "unknown"
    try:
        subject = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        subject = ""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = " (DIRTY: uncommitted changes)" if status else ""
    except Exception:
        dirty = ""
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    print("=" * 78)
    print(f"RUN PROVENANCE")
    print(f"  timestamp : {ts}")
    print(f"  commit    : {short}{dirty}")
    if subject:
        print(f"  subject   : {subject}")
    print(f"  command   : {' '.join(sys.argv)}")
    print("=" * 78)


def _subsample_tensor(X, n, rng):
    """Random subsample rows of a torch tensor."""
    if len(X) <= n:
        return X, np.arange(len(X))
    idx = rng.choice(len(X), n, replace=False)
    return X[idx], idx
