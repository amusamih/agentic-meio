"""Minimal ReplenishmentEnv import and package-root helpers."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path


def locate_package_root(
    *,
    module_name: str,
    explicit_root: Path | None = None,
) -> Path | None:
    """Return the discovered package root for a benchmark package."""

    if explicit_root is not None:
        return _normalize_package_root(module_name, explicit_root)
    spec = find_spec(module_name)
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent


def _normalize_package_root(module_name: str, candidate_root: Path) -> Path:
    resolved = candidate_root.resolve()
    if (resolved / "__init__.py").exists():
        return resolved
    nested = resolved / module_name
    if (nested / "__init__.py").exists():
        return nested
    return resolved


__all__ = ["locate_package_root"]
