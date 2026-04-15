import argparse
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from clients import OMDbClient, SensCritiqueClient, TMDBClient, AllocineClient
from display import export_json, print_ranking
from models import STREAMING_PROVIDERS
from ranker import collect, enrich_and_rank

load_dotenv()
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🎬 Streaming Movie Ranker — TMDB · OMDb · SensCritique · AlloCiné",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # all platforms
  python main.py -p netflix disney            # Netflix + Disney+ only
  python main.py -p amazon --top 20           # top 20 on Amazon
  python main.py --top 50 --export top.json
  python main.py --list-platforms

Env vars (or .env file):
  TMDB_API_KEY   https://www.themoviedb.org/settings/api  (free)
  OMDB_API_KEY   https://www.omdbapi.com/apikey.aspx       (free, 1000 req/day)
        """,
    )
    parser.add_argument(
        "-p",
        "--platforms",
        nargs="+",
        choices=list(STREAMING_PROVIDERS.keys()),
        metavar="PLATFORM",
    )
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument(
        "--pages", type=int, default=3, help="TMDB pages per provider (~20 films each)"
    )
    parser.add_argument("--region", default="FR")
    parser.add_argument("--export", metavar="FILE.json")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--list-platforms", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_platforms:
        console.print("\n[bold]Available platforms:[/bold]")
        for key, info in STREAMING_PROVIDERS.items():
            console.print(f"  {info['emoji']}  [cyan]{key:<10}[/cyan] {info['name']}")
        console.print()
        return

    tmdb_key = os.getenv("TMDB_API_KEY", "")
    omdb_key = os.getenv("OMDB_API_KEY", "")

    missing = []
    if not tmdb_key:
        missing.append("TMDB_API_KEY  →  https://www.themoviedb.org/settings/api")
    if not omdb_key and not args.fast:
        missing.append(
            "OMDB_API_KEY  →  https://www.omdbapi.com/apikey.aspx  (or use --fast)"
        )
    if missing:
        console.print(
            Panel(
                "\n".join(f"⚠  {m}" for m in missing),
                title="[yellow]Missing API keys[/yellow]",
                border_style="yellow",
            )
        )
        sys.exit(1)

    platforms = args.platforms or list(STREAMING_PROVIDERS.keys())

    console.print()
    console.print(
        Panel(
            f"[bold white]🎬  STREAMING MOVIE RANKER[/bold white]\n"
            f"[dim]Platforms: {' · '.join(STREAMING_PROVIDERS[p]['emoji'] + ' ' + STREAMING_PROVIDERS[p]['name'] for p in platforms)}\n"
            f"Top {args.top} · Region {args.region} · {'IMDb · Tomatometer · SensCritique · AlloCiné'}[/dim]",
            border_style="bright_cyan",
            padding=(0, 2),
        )
    )

    tmdb = TMDBClient(tmdb_key, region=args.region)
    omdb = OMDbClient(omdb_key)
    sc = SensCritiqueClient()
    allocine = AllocineClient()

    console.print()
    candidates = collect(platforms, tmdb, pages=args.pages)
    console.print(f"[green]✓[/green] {len(candidates)} unique films collected\n")

    movies = enrich_and_rank(
        candidates, tmdb, omdb, sc, allocine, top_n=args.top, workers=args.workers
    )

    print_ranking(movies)

    if args.export:
        export_json(movies, args.export)


if __name__ == "__main__":
    main()
