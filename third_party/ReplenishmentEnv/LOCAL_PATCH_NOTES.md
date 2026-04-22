# Local ReplenishmentEnv Patch Notes

## Source

- Upstream repository: `https://github.com/VictorYXL/ReplenishmentEnv`
- Downloaded source archive: `https://github.com/VictorYXL/ReplenishmentEnv/archive/refs/heads/main.zip`
- Upstream archive hash recorded by the installed package metadata:
  `sha256=dde2c66502727eafb2799a037c183b6ef631f94e15d1e1f2ba7514a587ee4028`

## Why this local copy exists

The installed `replenishment==1.0` package in this environment is structurally incomplete.
Its installed `RECORD` contains `ReplenishmentEnv/__init__.py`, `config/`, and `data/`, but omits the `env/`, `wrapper/`, and `utility/` subpackages that `ReplenishmentEnv/__init__.py` imports.

## Minimal local patch

- `setup.py`
  - changed `find_packages(include=["ReplenishmentEnv"])`
  - to `find_packages(include=["ReplenishmentEnv", "ReplenishmentEnv.*"])`

That packaging defect caused the broken upstream install here.
No benchmark logic was changed in this local third-party copy.
