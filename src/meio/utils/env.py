"""Small helpers for repo-local environment variable resolution."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


def load_env_value(key: str, *, env_file_path: Path = DEFAULT_ENV_FILE) -> str | None:
    """Resolve one setting from the process environment, then the repo .env file."""

    direct_value = os.getenv(key)
    if direct_value:
        return direct_value
    return load_env_value_from_file(env_file_path, key)


def load_env_value_from_file(path: Path, key: str) -> str | None:
    """Load one value from a simple KEY=value .env-style file."""

    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != key:
            continue
        parsed = value.strip().strip("\"'")
        return parsed or None
    return None


__all__ = ["DEFAULT_ENV_FILE", "load_env_value", "load_env_value_from_file"]
