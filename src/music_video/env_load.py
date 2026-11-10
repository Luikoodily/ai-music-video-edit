"""Load optional music-video/.env into os.environ (no third-party dotenv).

Never prints secret values. Missing file is fine.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PACKAGE_ROOT / ".env"


def load_dotenv(path: Path | None = None, *, override: bool = False) -> Path | None:
    env_path = Path(path) if path else DEFAULT_ENV_PATH
    if not env_path.is_file():
        return None
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = val
    return env_path
