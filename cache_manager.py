"""
24-hour ranking cache — stores the full list of enriched Movie objects as JSON.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models import Movie

CACHE_PATH = Path(__file__).parent / "data" / "cache.json"
CACHE_TTL_H = 24


def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def load(cache_key: str) -> Optional[list[Movie]]:
    """Return cached movies if fresh, else None."""
    if not CACHE_PATH.exists():
        return None
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        entry = raw.get(cache_key)
        if not entry:
            return None
        age_h = (_now_ts() - entry["ts"]) / 3600
        if age_h > CACHE_TTL_H:
            return None
        return [Movie.model_validate(m) for m in entry["movies"]]
    except Exception:
        return None


def save(cache_key: str, movies: list[Movie]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    except Exception:
        raw = {}
    raw[cache_key] = {
        "ts":     _now_ts(),
        "movies": [m.model_dump() for m in movies],
    }
    CACHE_PATH.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")


def invalidate(cache_key: str | None = None) -> None:
    """Clear one key or the entire cache."""
    if not CACHE_PATH.exists():
        return
    if cache_key is None:
        CACHE_PATH.unlink()
        return
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        raw.pop(cache_key, None)
        CACHE_PATH.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def cache_key_for(platforms: list[str], with_allocine: bool, pages: int) -> str:
    return f"{'_'.join(sorted(platforms))}_ac{int(with_allocine)}_p{pages}"
