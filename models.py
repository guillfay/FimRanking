from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


STREAMING_PROVIDERS: dict[str, dict] = {
    "netflix": {"id": 8, "name": "Netflix", "emoji": "🔴"},
    "disney": {"id": 337, "name": "Disney+", "emoji": "🔵"},
    "amazon": {"id": 119, "name": "Amazon Prime", "emoji": "🌐"},
    "canal": {"id": 35, "name": "Canal+", "emoji": "⬛️"},
    "apple": {"id": 350, "name": "Apple TV+", "emoji": "⚪"},
    "ocs": {"id": 56, "name": "OCS", "emoji": "🟠"},
}

SCORE_SOURCES = [
    "imdb",
    "tomatometer",
    "senscritique",
    "allocine_press",
    "allocine_audience",
    "tmdb",
]


DEFAULT_WEIGHTS: dict[str, int] = {
    "imdb": 25,
    "tomatometer": 20,
    "senscritique": 20,
    "allocine_press": 20,
    "allocine_audience": 5,
    "tmdb": 10,
}


class Scores(BaseModel):
    imdb: Optional[float] = None  # /10
    tomatometer: Optional[int] = None  # %
    rt_audience: Optional[int] = None  # %
    senscritique: Optional[float] = None  # /10
    allocine_press: Optional[float] = None  # /5
    allocine_audience: Optional[float] = None  # /5
    tmdb: Optional[float] = None  # /10
    imdb_id: Optional[str] = None


class Movie(BaseModel):
    tmdb_id: int
    title: str
    original_title: str
    year: int
    overview: str = ""
    platforms: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    runtime: Optional[int] = None  # minutes
    poster_path: Optional[str] = None  # TMDB path e.g. /abc123.jpg
    scores: Scores = Field(default_factory=Scores)
    composite_score: float = 0.0

    def __hash__(self):
        return hash(self.tmdb_id)

    def __eq__(self, o):
        return isinstance(o, Movie) and self.tmdb_id == o.tmdb_id


def compute_composite_score(scores: Scores, weights: dict[str, int]) -> float:
    """
    Weighted average /100.
    weights: dict source→int (0-100), active sources only (non-zero).
    Missing scores have their weight redistributed proportionally.
    """
    # Normalise /100 for each source
    raw: dict[str, float] = {}
    if scores.imdb is not None:
        raw["imdb"] = scores.imdb / 10 * 100
    if scores.tomatometer is not None:
        raw["tomatometer"] = float(scores.tomatometer)
    if scores.senscritique is not None:
        raw["senscritique"] = scores.senscritique / 10 * 100
    if scores.allocine_press is not None:
        raw["allocine_press"] = scores.allocine_press / 5 * 100
    if scores.allocine_audience is not None:
        raw["allocine_audience"] = scores.allocine_audience / 5 * 100
    if scores.tmdb is not None:
        raw["tmdb"] = scores.tmdb / 10 * 100

    candidates = [(raw[s], weights.get(s, 0)) for s in raw if weights.get(s, 0) > 0]
    if not candidates:
        return 0.0

    total_w = sum(w for _, w in candidates)
    return sum(v * w for v, w in candidates) / total_w
