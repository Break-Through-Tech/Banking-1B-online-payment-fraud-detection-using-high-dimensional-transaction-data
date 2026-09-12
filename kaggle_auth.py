from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
KAGGLE_HOME_FILE = Path.home() / ".kaggle" / "kaggle.json"


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    return values


def get_kaggle_credentials() -> dict[str, str]:
    env_values = _read_env_file(ENV_FILE)

    username = env_values.get("KAGGLE_USERNAME") or os.getenv("KAGGLE_USERNAME")
    key = env_values.get("KAGGLE_KEY") or os.getenv("KAGGLE_KEY")

    if username and key:
        return {"username": username, "key": key}

    if KAGGLE_HOME_FILE.exists():
        with KAGGLE_HOME_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        username = data.get("username")
        key = data.get("key")
        if username and key:
            return {"username": username, "key": key}

    raise FileNotFoundError(
        "No Kaggle credentials found. Create a .env file from .env.example or keep your ~/.kaggle/kaggle.json file."
    )


if __name__ == "__main__":
    print(get_kaggle_credentials())
