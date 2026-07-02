"""Shared logging setup so every stage writes to console + a common log file."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "logs"

_CONFIGURED_LOGGERS: set[str] = set()


def get_logger(
    name: str,
    log_dir: str | Path = DEFAULT_LOG_DIR,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a logger that writes to both stdout and ``<log_dir>/<name>.log``.

    Safe to call repeatedly with the same name (handlers are only attached once).
    """
    logger = logging.getLogger(name)
    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    _CONFIGURED_LOGGERS.add(name)
    return logger
