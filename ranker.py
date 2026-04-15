import concurrent.futures
from typing import Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from clients import AllocineClient, OMDbClient, SensCritiqueClient, TMDBClient
from models import (
    Movie,
    Scores,
    DEFAULT_WEIGHTS,
    STREAMING_PROVIDERS,
    compute_composite_score,
)

console = Console()
_POOL_FACTOR = 2


def _enrich(
    movie: Movie,
    omdb: OMDbClient,
    sc: SensCritiqueClient,
    allocine: Optional[AllocineClient],
    weights: dict[str, int],
) -> Movie:
    omdb_data = (
        omdb.fetch(imdb_id=movie.scores.imdb_id)
        if movie.scores.imdb_id
        else omdb.fetch(title=movie.original_title or movie.title, year=movie.year)
    )
    movie.scores.imdb, movie.scores.tomatometer, movie.scores.rt_audience = (
        OMDbClient.parse(omdb_data)
    )
    movie.scores.senscritique = sc.get_score(
        movie.title, movie.original_title, movie.year
    )

    if allocine:
        movie.scores.allocine_press, movie.scores.allocine_audience = (
            allocine.get_scores(movie.title, movie.original_title, movie.year)
        )

    movie.composite_score = compute_composite_score(movie.scores, weights)
    return movie


def collect(platforms: list[str], tmdb: TMDBClient, pages: int = 3) -> list[Movie]:
    genres_map = tmdb.get_genres()
    movies: dict[int, Movie] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("", total=len(platforms))
        for key in platforms:
            prov = STREAMING_PROVIDERS[key]
            progress.update(task, description=f"{prov['emoji']} {prov['name']}...")
            for raw in tmdb.get_movies_by_provider(prov["id"], pages=pages):
                mid = raw["id"]
                if mid not in movies:
                    year = int((raw.get("release_date") or "0")[:4] or 0)
                    genres = [
                        genres_map.get(gid, "") for gid in raw.get("genre_ids", [])
                    ]
                    movies[mid] = Movie(
                        tmdb_id=mid,
                        title=raw.get("title") or raw.get("original_title", "?"),
                        original_title=raw.get("original_title", ""),
                        year=year,
                        overview=raw.get("overview", ""),
                        genres=[g for g in genres if g],
                        scores=Scores(tmdb=raw.get("vote_average")),
                        poster_path=raw.get("poster_path"),
                    )
                movies[mid].platforms.append(prov["name"])
            progress.advance(task)

    return list(movies.values())


def enrich_and_rank(
    candidates: list[Movie],
    tmdb: TMDBClient,
    omdb: OMDbClient,
    sc: SensCritiqueClient,
    allocine: AllocineClient,
    weights: dict[str, int] = DEFAULT_WEIGHTS,
    top_n: int = 50,
    workers: int = 8,
) -> list[Movie]:
    pool = sorted(candidates, key=lambda m: m.scores.tmdb or 0, reverse=True)
    pool = pool[: min(len(pool), int(top_n * _POOL_FACTOR))]

    # ── Batch fetch full details (runtime, countries, imdb_id) ────────────
    # Single /movie/{id} call replaces separate external_ids calls
    ids_needing_details = [
        m.tmdb_id for m in pool if not m.scores.imdb_id or m.runtime is None
    ]

    if ids_needing_details:
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]Fetching film details..."),
            BarColumn(),
            TextColumn(f"{{task.completed}}/{len(ids_needing_details)}"),
            console=console,
        ) as progress:
            task = progress.add_task("", total=len(ids_needing_details))

            movie_map = {m.tmdb_id: m for m in pool}

            def fetch_details(tmdb_id: int):
                data = tmdb._get(f"/movie/{tmdb_id}")
                progress.advance(task)
                return tmdb_id, data

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                for tmdb_id, data in ex.map(fetch_details, ids_needing_details):
                    if not data or tmdb_id not in movie_map:
                        continue
                    m = movie_map[tmdb_id]
                    # imdb_id
                    if not m.scores.imdb_id:
                        m.scores.imdb_id = data.get("imdb_id")
                    # runtime
                    if m.runtime is None:
                        m.runtime = data.get("runtime")
                    # countries
                    if not m.countries:
                        m.countries = [
                            c.get("name", "")
                            for c in data.get("production_countries", [])
                        ]
                    # detailed genres (overwrite the genre_ids-based list)
                    if not m.genres:
                        m.genres = [g.get("name", "") for g in data.get("genres", [])]

    # ── Parallel enrichment (OMDb + SC + optional Allociné) ──────────────
    enriched: list[Movie] = []
    enriched_ids: set[int] = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Enriching scores...", total=len(pool))

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_enrich, m, omdb, sc, allocine, weights): m
                for m in pool
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    movie = future.result()
                except Exception:
                    movie = futures[future]
                    movie.composite_score = compute_composite_score(
                        movie.scores, weights
                    )
                enriched.append(movie)
                enriched_ids.add(movie.tmdb_id)
                progress.advance(task)

    for m in candidates:
        if m.tmdb_id not in enriched_ids:
            m.composite_score = compute_composite_score(m.scores, weights)
            enriched.append(m)

    return sorted(enriched, key=lambda m: m.composite_score, reverse=True)[:top_n]
