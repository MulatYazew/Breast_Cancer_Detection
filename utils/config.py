"""YAML config loading with deep-merge override support.

The pipeline is config-driven: ``config.yaml`` at the repo root holds the full
default configuration (paths + hyperparameters for every stage). Files under
``configs/`` hold partial overrides (e.g. a quick-run smoke-test profile, or an
alternate U-Net encoder) that are deep-merged on top of the defaults.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    overrides: str | Path | list[str | Path] | None = None,
) -> dict[str, Any]:
    """Load the base config and apply zero or more override YAML files in order.

    Args:
        config_path: Path to the base config (defaults to repo-root ``config.yaml``).
        overrides: One or more paths to override YAML files (e.g.
            ``configs/quick_run.yaml``), applied in order so later files win.

    Returns:
        The fully merged configuration dictionary.
    """
    config = load_yaml(config_path)

    if overrides is None:
        override_paths: list[str | Path] = []
    elif isinstance(overrides, (str, Path)):
        override_paths = [overrides]
    else:
        override_paths = list(overrides)

    for override_path in override_paths:
        config = _deep_merge(config, load_yaml(override_path))

    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a dotted path key (e.g. 'paths.checkpoints') to an absolute Path.

    Relative paths in the config are interpreted as relative to the repo root.
    """
    node: Any = config
    for part in key.split("."):
        node = node[part]
    path = Path(node)
    return path if path.is_absolute() else (REPO_ROOT / path)
