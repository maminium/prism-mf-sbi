from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = yaml.safe_load(p.read_text())
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected mapping in {p}, got {type(data).__name__}")
    return dict(data)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """
    Recursively merge mappings. override wins.
    """
    out: dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

