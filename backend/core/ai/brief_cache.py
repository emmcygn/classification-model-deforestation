"""Cached policy briefs for keyless demo."""

import json
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


def load_cached_brief(region: str) -> dict | None:
    path = CACHE_DIR / f"{region}_brief.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_cached_brief(region: str, brief: dict) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{region}_brief.json"
    with open(path, "w") as f:
        json.dump(brief, f, indent=2)
    return path
