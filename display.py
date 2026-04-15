import json
from typing import Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from models import Movie, STREAMING_PROVIDERS, DEFAULT_WEIGHTS

console = Console()


def _cell(value: Optional[float | int], suffix: str = "") -> str:
    return "[dim]—[/dim]" if value is None else f"{value}{suffix}"


def print_ranking(movies: list[Movie], with_allocine: bool = False) -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold white]🎬  TOP STREAMING FILMS[/bold white]",
            border_style="bright_yellow",
            padding=(0, 4),
        )
    )
    console.print()

    t = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold bright_cyan",
        border_style="dim white",
        row_styles=["", "dim"],
        expand=True,
        pad_edge=True,  # no outer padding
    )

    t.add_column("#", justify="right", style="bold yellow", width=3, no_wrap=True)
    t.add_column("Title", min_width=25, max_width=34, no_wrap=True)
    t.add_column("Year", justify="center", width=4, no_wrap=True)
    t.add_column("Plat.", justify="center", width=6, no_wrap=True)
    t.add_column("Score", justify="center", width=6, no_wrap=True)
    t.add_column("IMDb", justify="center", width=6, no_wrap=True)
    t.add_column("R🍅", justify="center", width=4, no_wrap=True)
    t.add_column("SC", justify="center", width=5, no_wrap=True)
    t.add_column("AC🗞", justify="center", width=5, no_wrap=True)
    t.add_column("AC👥", justify="center", width=5, no_wrap=True)
    t.add_column("TMDB", justify="center", width=5, no_wrap=True)
    t.add_column("Genres", min_width=12, max_width=24)

    for rank, m in enumerate(movies, 1):
        color = (
            "green"
            if m.composite_score >= 75
            else "yellow"
            if m.composite_score >= 55
            else "red"
        )
        platforms_icons = " ".join(
            v["emoji"]
            for k, v in STREAMING_PROVIDERS.items()
            if v["name"] in m.platforms
        )
        title = m.title if len(m.title) <= 33 else m.title[:32] + "…"

        row = [
            str(rank),
            title,
            str(m.year) if m.year else "—",
            platforms_icons,
            f"[{color}]{m.composite_score:.1f}[/{color}]",
            _cell(m.scores.imdb, "/10"),
            _cell(m.scores.tomatometer, "%"),
            _cell(m.scores.senscritique, "/10"),
            _cell(m.scores.allocine_press, "/5"),
            _cell(m.scores.allocine_audience, "/5"),
        ]
        row += [
            _cell(round(m.scores.tmdb, 1), "/10")
            if m.scores.tmdb is not None
            else _cell(None),
            ", ".join(m.genres[:2]),
        ]
        t.add_row(*row)

    console.print(t)

    weights_str = " · ".join(
        f"{k.replace('_', ' ').title()} {v}%"
        for k, v in DEFAULT_WEIGHTS.items()
        if v > 0
    )
    console.print(f"\n[dim]Score /100 · {weights_str}[/dim]")
    console.print("[dim]— : unavailable[/dim]\n")


def export_json(movies: list[Movie], path: str) -> None:
    data = [
        {
            "rank": rank,
            "tmdb_id": m.tmdb_id,
            "title": m.title,
            "original_title": m.original_title,
            "year": m.year,
            "platforms": sorted(set(m.platforms)),
            "genres": m.genres,
            "composite_score": round(m.composite_score, 2),
            "scores": m.scores.model_dump(),
        }
        for rank, m in enumerate(movies, 1)
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    console.print(f"[green]✓ JSON exported: {path}[/green]")
