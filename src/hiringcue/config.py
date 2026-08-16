"""Configuration loading.

Every value that affects what is measured lives in `configs/*.yaml`, not in
code. Changing a threshold, a model, a seed, or a label set is therefore a
visible diff in a versioned file rather than an edit buried in a function.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from . import paths


@lru_cache(maxsize=None)
def load(name: str) -> dict[str, Any]:
    """Load `configs/<name>.yaml`."""
    path: Path = paths.CONFIGS / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing config: {path}")
    with path.open() as handle:
        return yaml.safe_load(handle)


def study() -> dict[str, Any]:
    return load("study")


def models() -> dict[str, Any]:
    return load("models")


def stimuli() -> dict[str, Any]:
    return load("stimuli")


def gate_thresholds() -> dict[str, Any]:
    return load("gates")
