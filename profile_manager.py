"""
User profile management — local JSON storage.

Profile schema:
{
  "profiles": {
    "<name>": {
      "subscriptions": ["netflix", "disney"],
      "weights": {"imdb": 30, "tomatometer": 25, ...},
      "seen": {
        "<tmdb_id>": {"rating": 8.0, "wishlist": false}
      }
    }
  }
}
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from models import DEFAULT_WEIGHTS, STREAMING_PROVIDERS

PROFILES_PATH = Path(__file__).parent / "data" / "profiles.json"


def _load() -> dict:
    if PROFILES_PATH.exists():
        return json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    return {"profiles": {}}


def _save(data: dict) -> None:
    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILES_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Profile CRUD ──────────────────────────────────────────────

def list_profiles() -> list[str]:
    return list(_load()["profiles"].keys())


def get_profile(name: str) -> Optional[dict]:
    return _load()["profiles"].get(name)


def create_profile(name: str, subscriptions: list[str] | None = None) -> dict:
    data = _load()
    profile = {
        "subscriptions": subscriptions or list(STREAMING_PROVIDERS.keys()),
        "weights":       DEFAULT_WEIGHTS.copy(),
        "seen":          {},
    }
    data["profiles"][name] = profile
    _save(data)
    return profile


def update_profile(name: str, **kwargs) -> dict:
    """Update any top-level profile keys (subscriptions, weights)."""
    data = _load()
    if name not in data["profiles"]:
        raise KeyError(f"Profile '{name}' not found")
    data["profiles"][name].update(kwargs)
    _save(data)
    return data["profiles"][name]


def delete_profile(name: str) -> None:
    data = _load()
    data["profiles"].pop(name, None)
    _save(data)


# ── Seen / Wishlist ───────────────────────────────────────────

def mark_seen(profile_name: str, tmdb_id: int, rating: Optional[float] = None) -> None:
    data = _load()
    profile = data["profiles"][profile_name]
    entry = profile["seen"].get(str(tmdb_id), {"wishlist": False})
    entry["seen"]   = True
    entry["rating"] = rating
    profile["seen"][str(tmdb_id)] = entry
    _save(data)


def mark_wishlist(profile_name: str, tmdb_id: int) -> None:
    data = _load()
    profile = data["profiles"][profile_name]
    entry = profile["seen"].get(str(tmdb_id), {})
    entry["wishlist"] = not entry.get("wishlist", False)  # toggle
    profile["seen"][str(tmdb_id)] = entry
    _save(data)


def unmark_seen(profile_name: str, tmdb_id: int) -> None:
    data = _load()
    profile = data["profiles"][profile_name]
    entry = profile["seen"].get(str(tmdb_id), {})
    entry.pop("seen", None)
    entry.pop("rating", None)
    if entry:
        profile["seen"][str(tmdb_id)] = entry
    else:
        profile["seen"].pop(str(tmdb_id), None)
    _save(data)


def get_seen_ids(profile_name: str) -> set[int]:
    profile = get_profile(profile_name) or {}
    return {int(k) for k, v in profile.get("seen", {}).items() if v.get("seen")}


def get_wishlist_ids(profile_name: str) -> set[int]:
    profile = get_profile(profile_name) or {}
    return {int(k) for k, v in profile.get("seen", {}).items() if v.get("wishlist")}


def get_user_rating(profile_name: str, tmdb_id: int) -> Optional[float]:
    profile = get_profile(profile_name) or {}
    return profile.get("seen", {}).get(str(tmdb_id), {}).get("rating")
