"""Memory and disk limits for a MacBook Air M4 with 24 GB unified memory.

No function in this module downloads a model. Optional checkpoints larger than
8 GB are rejected. A 27B checkpoint is rejected unconditionally.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

OPTIONAL_CHECKPOINT_LIMIT_BYTES = 8 * 1000**3
PEAK_PROCESS_LIMIT_BYTES = 18 * 1024**3
UNIFIED_MEMORY_BYTES = 24 * 1024**3

BLOCKED_27B = ("27b", "medgemma-27")
# Official file listing: model-00001-of-00002.safetensors 4.96 GB and
# model-00002-of-00002.safetensors 3.64 GB. Sum is above the 8 GB cap.
MEDGEMMA_4B_SHARDS_GB = (4.96, 3.64)
MEDGEMMA_4B_ID = "google/medgemma-1.5-4b-it"


def ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def current_rss_bytes() -> int:
    """Current resident set of this process, from `ps` (kilobytes on macOS)."""
    output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return int(output.strip().split()[0]) * 1024


def disk_bytes(path: str | Path = ".") -> dict:
    usage = shutil.disk_usage(path)
    return {"total": usage.total, "used": usage.used, "free": usage.free}


def assert_process_budget(rss_bytes: int | None = None) -> None:
    rss = current_rss_bytes() if rss_bytes is None else rss_bytes
    if rss > PEAK_PROCESS_LIMIT_BYTES:
        raise MemoryError(
            f"Process RSS {rss / 1024**3:.2f} GiB exceeds the {PEAK_PROCESS_LIMIT_BYTES / 1024**3:.0f} GiB cap."
        )


def assert_model_id_allowed(model_id: str) -> None:
    """Refuse 27B weights and the 8.6 GB MedGemma 4B checkpoint. Does not download."""
    if not model_id or not str(model_id).strip():
        raise RuntimeError("No language-model id is configured, and none will be downloaded.")
    lowered = model_id.lower()
    if any(token in lowered for token in BLOCKED_27B):
        raise RuntimeError(
            f"Refusing to download or load {model_id}. "
            "A 27B checkpoint does not fit a 24 GB MacBook Air and is not used by this project."
        )
    if "medgemma-1.5-4b" in lowered or lowered.endswith("medgemma-4b-it"):
        total = sum(MEDGEMMA_4B_SHARDS_GB)
        raise RuntimeError(
            f"Refusing to download or load {model_id}. "
            f"Published weights are {MEDGEMMA_4B_SHARDS_GB[0]} GB + {MEDGEMMA_4B_SHARDS_GB[1]} GB "
            f"({total:.2f} GB), above the 8 GB optional-checkpoint limit. "
            "The ultrasound classifier does not use this model."
        )
    if "qwen2.5-7b" in lowered or "qwen2.5-14b" in lowered or "qwen2.5-32b" in lowered or "qwen2.5-72b" in lowered:
        raise RuntimeError(f"Refusing to download or load {model_id}. It is outside the 8 GB checkpoint budget.")


def hardware_snapshot(path: str | Path = ".") -> dict:
    memory = ram_bytes()
    rss = current_rss_bytes()
    disk = disk_bytes(path)
    cuda = False
    mps = False
    torch_version = None
    try:
        import torch

        torch_version = torch.__version__
        cuda = bool(torch.cuda.is_available())
        mps = bool(torch.backends.mps.is_available())
    except ImportError:
        pass
    return {
        "torch": torch_version,
        "cuda": cuda,
        "mps": mps,
        "cpu_count": os.cpu_count(),
        "ram_bytes": memory,
        "rss_bytes": rss,
        "disk_free_bytes": disk["free"],
        "optional_checkpoint_limit_bytes": OPTIONAL_CHECKPOINT_LIMIT_BYTES,
        "peak_process_limit_bytes": PEAK_PROCESS_LIMIT_BYTES,
        "medgemma_4b_published_gb": sum(MEDGEMMA_4B_SHARDS_GB),
        "medgemma_4b_loaded": False,
        "medgemma_27b_loaded": False,
    }
