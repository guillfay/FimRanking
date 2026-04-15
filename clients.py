import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

TMDB_BASE = "https://api.themoviedb.org/3"
OMDB_BASE = "http://www.omdbapi.com"
SC_GQL = "https://apollo.senscritique.com"
ALLOCINE_BASE = "https://www.allocine.fr"

_SCRAPE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9",
    "Referer": "https://www.allocine.fr/",
}


class TMDBClient:
    def __init__(self, api_key: str, region: str = "FR", language: str = "fr-FR"):
        self.api_key = api_key
        self.region = region
        self.language = language
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "StreamingRanker/2.0"

    def _get(self, endpoint: str, params: dict = {}) -> dict:
        try:
            r = self.session.get(
                f"{TMDB_BASE}{endpoint}",
                params={"api_key": self.api_key, "language": self.language, **params},
                timeout=10,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            return {}

    def get_genres(self) -> dict[int, str]:
        data = self._get("/genre/movie/list")
        return {g["id"]: g["name"] for g in data.get("genres", [])}

    def get_movies_by_provider(self, provider_id: int, pages: int = 3) -> list[dict]:
        """
        Fetch movies for a provider. Uses append_to_response=external_ids is not
        available on /discover — we instead resolve imdb_ids in a separate batched step.
        """
        movies = []
        for page in range(1, pages + 1):
            data = self._get(
                "/discover/movie",
                {
                    "with_watch_providers": provider_id,
                    "watch_region": self.region,
                    "sort_by": "popularity.desc",
                    "page": page,
                    "vote_count.gte": 100,
                },
            )
            results = data.get("results", [])
            if not results:
                break
            movies.extend(results)
        return movies

    def get_external_ids(self, tmdb_id: int) -> dict:
        """Single film external IDs (imdb_id, etc.)."""
        return self._get(f"/movie/{tmdb_id}/external_ids")

    def get_external_ids_batch(
        self, tmdb_ids: list[int], workers: int = 10
    ) -> dict[int, str]:
        """
        Fetch imdb_ids for a list of TMDB IDs in parallel.
        Returns {tmdb_id: imdb_id}.
        Much faster than fetching one-by-one inside the enrichment loop.
        """
        import concurrent.futures

        results: dict[int, str] = {}

        def fetch_one(tmdb_id: int) -> tuple[int, str | None]:
            data = self.get_external_ids(tmdb_id)
            return tmdb_id, data.get("imdb_id")

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for tmdb_id, imdb_id in ex.map(fetch_one, tmdb_ids):
                if imdb_id:
                    results[tmdb_id] = imdb_id

        return results


class OMDbClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "StreamingRanker/2.0"
        self._cache: dict[str, dict] = {}

    def fetch(
        self, *, imdb_id: str = None, title: str = None, year: int = None
    ) -> dict:
        key = imdb_id or f"{title}_{year}"
        if key in self._cache:
            return self._cache[key]

        params = {"apikey": self.api_key, "tomatoes": "true"}
        if imdb_id:
            params["i"] = imdb_id
        else:
            params.update({"t": title, "y": year, "type": "movie"})

        try:
            r = self.session.get(OMDB_BASE, params=params, timeout=8)
            r.raise_for_status()
            data = r.json()
            self._cache[key] = data
            return data
        except requests.RequestException:
            return {}

    @staticmethod
    def parse(data: dict) -> tuple[Optional[float], Optional[int], Optional[int]]:
        """Returns (imdb /10, tomatometer %, rt_audience %)."""
        if not data or data.get("Response") == "False":
            return None, None, None

        imdb, tomato, audience = None, None, None

        raw = data.get("imdbRating", "N/A")
        if raw not in ("N/A", "", None):
            try:
                imdb = float(raw)
            except ValueError:
                pass

        for r in data.get("Ratings", []):
            if r.get("Source") == "Rotten Tomatoes":
                try:
                    tomato = int(r["Value"].replace("%", "").strip())
                except (ValueError, KeyError):
                    pass

        raw_aud = data.get("tomatoUserMeter", "N/A")
        if raw_aud not in ("N/A", "", None):
            try:
                audience = int(raw_aud)
            except ValueError:
                pass

        return imdb, tomato, audience


class SensCritiqueClient:
    """
    Internal GQL endpoint. Correct args (reverse-engineered):
      keywords (not query), searchByUniverse, universe int 1=Film, stats.ratingCount
    """

    _HEADERS = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Origin": "https://www.senscritique.com",
        "Referer": "https://www.senscritique.com/",
    }
    _QUERY = """
    query Search($kw: String!, $limit: Int) {
      searchByUniverse(keywords: $kw, universe: "Film", limit: $limit) {
        products {
          id title originalTitle yearOfProduction rating universe
          stats { ratingCount }
        }
      }
    }
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self._HEADERS)
        self._cache: dict[str, Optional[float]] = {}
        self._ready = False

    def _init(self):
        if self._ready:
            return
        try:
            self.session.options(SC_GQL, timeout=5)
        except requests.RequestException:
            pass
        self._ready = True

    def get_score(self, title: str, original_title: str, year: int) -> Optional[float]:
        self._init()
        for t in dict.fromkeys(filter(None, [title, original_title])):
            key = f"{t}_{year}"
            if key in self._cache:
                v = self._cache[key]
                if v is not None:
                    return v
                continue
            result = self._search(t, year)
            self._cache[f"{title}_{year}"] = result
            self._cache[f"{original_title}_{year}"] = result
            if result is not None:
                return result
        return None

    def _search(self, title: str, year: int) -> Optional[float]:
        try:
            resp = self.session.post(
                SC_GQL,
                json={"query": self._QUERY, "variables": {"kw": title, "limit": 5}},
                timeout=8,
            )
            resp.raise_for_status()
            products = (
                resp.json().get("data", {}).get("searchByUniverse", {}).get("products")
                or []
            )
            for item in products:
                if item.get("universe") != 1:
                    continue
                item_year = item.get("yearOfProduction")
                if item_year and abs(int(item_year) - year) > 1:
                    continue
                rating = item.get("rating")
                count = (item.get("stats") or {}).get("ratingCount", 0)
                if rating and count and int(count) >= 10:
                    return float(rating)
        except Exception:
            pass
        return None


class AllocineClient:
    """
    Two-step, no Cloudflare:
      1. /_/autocomplete/{query}  →  film ID (pure JSON, no bot protection)
      2. /film/fichefilm_gen_cfilm={id}.html  →  parse press + audience from .rating-item
    """

    _AUTOCOMPLETE = ALLOCINE_BASE + "/_/autocomplete/{query}"
    _FILM_PAGE = ALLOCINE_BASE + "/film/fichefilm_gen_cfilm={film_id}.html"
    _DELAY = 0.3

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(_SCRAPE_HEADERS)
        self._id_cache: dict[str, Optional[int]] = {}
        self._score_cache: dict[int, tuple[Optional[float], Optional[float]]] = {}

    def get_scores(
        self, title: str, original_title: str, year: int
    ) -> tuple[Optional[float], Optional[float]]:
        for t in dict.fromkeys(filter(None, [title, original_title])):
            film_id = self._get_film_id(t, year)
            if film_id:
                return self._get_film_scores(film_id)
        return None, None

    def _get_film_id(self, title: str, year: int) -> Optional[int]:
        key = f"{title}_{year}"
        if key in self._id_cache:
            return self._id_cache[key]
        try:
            r = self.session.get(
                self._AUTOCOMPLETE.format(query=requests.utils.quote(title)),
                timeout=8,
            )
            r.raise_for_status()
            for item in r.json().get("results", []):
                if item.get("entity_type") != "movie":
                    continue
                item_year = (item.get("data") or {}).get("year")
                if item_year and abs(int(item_year) - year) > 1:
                    continue
                film_id = int(item["entity_id"])
                self._id_cache[key] = film_id
                return film_id
        except Exception:
            pass
        self._id_cache[key] = None
        return None

    def _get_film_scores(self, film_id: int) -> tuple[Optional[float], Optional[float]]:
        if film_id in self._score_cache:
            return self._score_cache[film_id]

        time.sleep(self._DELAY)
        press, audience = None, None
        try:
            r = self.session.get(self._FILM_PAGE.format(film_id=film_id), timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for item in soup.select(".rating-item"):
                note_el = item.select_one(".stareval-note")
                if not note_el:
                    continue
                try:
                    score = float(note_el.get_text(strip=True).replace(",", "."))
                except ValueError:
                    continue
                item_text = item.get_text(" ", strip=True).lower()
                if "presse" in item_text:
                    press = score
                elif "spectateur" in item_text:
                    audience = score
            # Fallback: positional if labels not found
            if press is None and audience is None:
                scores = []
                for el in soup.select(".stareval-note"):
                    txt = el.get_text(strip=True)
                    if txt in ("", "--"):
                        continue
                    try:
                        scores.append(float(txt.replace(",", ".")))
                    except ValueError:
                        pass
                if len(scores) >= 2:
                    press, audience = scores[0], scores[1]
                elif scores:
                    audience = scores[0]
        except Exception:
            pass

        self._score_cache[film_id] = (press, audience)
        return press, audience
