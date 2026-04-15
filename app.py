"""
Streaming Movie Ranker — Streamlit UI
Run: streamlit run app.py
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from typing import Optional

import streamlit as st
from dotenv import load_dotenv


from cache_manager import (
    CACHE_PATH,
    cache_key_for,
    invalidate,
    load as cache_load,
    save as cache_save,
)
from clients import AllocineClient, OMDbClient, SensCritiqueClient, TMDBClient
from models import (
    DEFAULT_WEIGHTS,
    Movie,
    SCORE_SOURCES,
    STREAMING_PROVIDERS,
    compute_composite_score,
)
from profile_manager import (
    create_profile,
    delete_profile,
    get_profile,
    get_seen_ids,
    get_user_rating,
    get_wishlist_ids,
    list_profiles,
    mark_seen,
    mark_wishlist,
    unmark_seen,
    update_profile,
)
from ranker import collect, enrich_and_rank

load_dotenv()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Streaming Ranker",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  [data-testid="stSidebar"] { min-width: 260px; max-width: 280px; }
  .movie-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 2px; }
  .movie-meta  { font-size: 0.78rem; color: #94a3b8; margin-bottom: 4px; }
  .synopsis    { font-size: 0.82rem; color: #cbd5e1; margin: 6px 0 10px 0;
                 line-height: 1.5; }
  .platform-tag { background:#1e293b; border-radius:4px; padding:2px 7px;
                  font-size:0.72rem; margin-right:3px; }
  .rank-num   { font-size:1.8rem; font-weight:800; color:#facc15; line-height:1; }
  .score-num  { font-size:1.4rem; font-weight:700; line-height:1; }
  .score-bar-wrap { margin: 6px 0 4px 0; }
  .badge-seen { color:#4ade80; font-size:0.78rem; }
  .badge-wish { color:#fb923c; font-size:0.78rem; }
  div[data-testid="stExpander"] details { border:none !important; }
</style>
""",
    unsafe_allow_html=True,
)

TMDB_KEY = os.getenv("TMDB_API_KEY", "")
OMDB_KEY = os.getenv("OMDB_API_KEY", "")

SOURCE_LABELS = {
    "imdb": "IMDb ⭐",
    "tomatometer": "Tomatometer 🍅",
    "senscritique": "SensCritique 🇫🇷",
    "allocine_press": "Allociné Presse 🎬",
    "allocine_audience": "Allociné Spectateurs 🎬",
    "tmdb": "TMDB",
}
SOURCE_COLORS = {
    "imdb": "#f59e0b",
    "tomatometer": "#ef4444",
    "senscritique": "#3b82f6",
    "allocine_press": "#8b5cf6",
    "allocine_audience": "#a78bfa",
    "tmdb": "#6b7280",
}

ACTIVE_SOURCES = SCORE_SOURCES  # all sources always active (Allociné permanent)


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════


def _now() -> float:
    return datetime.now(timezone.utc).timestamp()


def _score_val(movie: Movie, source: str) -> Optional[float]:
    """Normalise source score to 0-100."""
    s = movie.scores
    table = {
        "imdb": (s.imdb, lambda v: v / 10 * 100),
        "tomatometer": (s.tomatometer, lambda v: float(v)),
        "senscritique": (s.senscritique, lambda v: v / 10 * 100),
        "allocine_press": (s.allocine_press, lambda v: v / 5 * 100),
        "allocine_audience": (s.allocine_audience, lambda v: v / 5 * 100),
        "tmdb": (s.tmdb, lambda v: v / 10 * 100),
    }
    raw, fn = table[source]
    return fn(raw) if raw is not None else None


def _fmt_score(movie: Movie, source: str) -> str:
    s = movie.scores
    if source == "imdb":
        return f"{s.imdb}/10" if s.imdb else "—"
    if source == "tomatometer":
        return f"{s.tomatometer}%" if s.tomatometer else "—"
    if source == "senscritique":
        return f"{s.senscritique}/10" if s.senscritique else "—"
    if source == "allocine_press":
        return f"{s.allocine_press}/5" if s.allocine_press else "—"
    if source == "allocine_audience":
        return f"{s.allocine_audience}/5" if s.allocine_audience else "—"
    if source == "tmdb":
        return f"{s.tmdb}/10" if s.tmdb else "—"
    return "—"


def _cache_age() -> str:
    if not CACHE_PATH.exists():
        return ""
    try:
        raw = json.loads(CACHE_PATH.read_text())
        ts_list = [v["ts"] for v in raw.values() if "ts" in v]
        if not ts_list:
            return ""
        age_m = int((_now() - max(ts_list)) / 60)
        return (
            f"Cache : {age_m}min"
            if age_m < 60
            else f"Cache : {age_m // 60}h{age_m % 60:02d}"
        )
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════
#  Vase communicants weight sliders
#  When one source goes up, the excess is taken from the others
#  proportionally to their current values.
# ══════════════════════════════════════════════════════════════


def _redistribute(
    weights: dict[str, int], changed: str, new_val: int
) -> dict[str, int]:
    """
    Set weights[changed] = new_val and redistribute the delta
    proportionally among the other sources so the total stays 100.
    """
    old_val = weights[changed]
    delta = new_val - old_val
    if delta == 0:
        return weights

    others = {k: v for k, v in weights.items() if k != changed}
    total_others = sum(others.values())

    result = weights.copy()
    result[changed] = new_val

    if total_others == 0:
        # Edge case: all weight was on the changed source — can't go higher
        result[changed] = old_val
        return result

    # Distribute -delta proportionally among others
    remaining = -delta
    keys = [k for k in others]
    for i, k in enumerate(keys):
        if i == len(keys) - 1:
            result[k] = max(0, result[k] + remaining)
        else:
            share = round(-delta * others[k] / total_others)
            new_k = max(0, result[k] + share)
            remaining -= new_k - result[k]
            result[k] = new_k

    # Ensure total == 100 (rounding fix)
    total = sum(result.values())
    if total != 100:
        # Apply correction to the largest non-changed source
        fix_key = max((k for k in result if k != changed), key=lambda k: result[k])
        result[fix_key] += 100 - total

    return result


def weight_sliders(weights: dict[str, int]) -> dict[str, int]:
    """
    Renders the segmented bar + one slider per source.
    Uses vase communicants logic — total always stays 100.
    """
    # Segmented colour bar
    bar = '<div style="display:flex;height:14px;border-radius:7px;overflow:hidden;margin:8px 0 12px 0">'
    for src in ACTIVE_SOURCES:
        w = weights.get(src, 0)
        if w:
            bar += f'<div style="width:{w}%;background:{SOURCE_COLORS[src]}" title="{SOURCE_LABELS[src]} {w}%"></div>'
    bar += "</div>"
    legend = "&nbsp;&nbsp;".join(
        f'<span style="color:{SOURCE_COLORS[s]};font-size:0.72rem">■ {SOURCE_LABELS[s]}&nbsp;{weights.get(s, 0)}%</span>'
        for s in ACTIVE_SOURCES
    )
    st.sidebar.markdown(bar + "<br>" + legend, unsafe_allow_html=True)
    st.sidebar.markdown("")

    new_weights = weights.copy()
    for src in ACTIVE_SOURCES:
        val = st.sidebar.slider(
            f"{SOURCE_LABELS[src]}",
            0,
            100,
            new_weights.get(src, 0),
            key=f"w_{src}",
            help=f"Actuellement {new_weights.get(src, 0)}%",
        )
        if val != new_weights[src]:
            new_weights = _redistribute(new_weights, src, val)
            # Force rerun so bar + other sliders update immediately
            st.rerun()

    return new_weights


# ══════════════════════════════════════════════════════════════
#  Movie card
# ══════════════════════════════════════════════════════════════


def movie_card(
    rank: int,
    movie: Movie,
    weights: dict[str, int],
    seen_ids: set[int],
    wish_ids: set[int],
    profile_name: str,
):
    is_seen = movie.tmdb_id in seen_ids
    is_wish = movie.tmdb_id in wish_ids
    user_rating = get_user_rating(profile_name, movie.tmdb_id)

    score_color = (
        "#4ade80"
        if movie.composite_score >= 75
        else "#facc15"
        if movie.composite_score >= 55
        else "#f87171"
    )

    # ── Segmented score bar ───────────────────────────────────
    bar = '<div class="score-bar-wrap"><div style="display:flex;height:8px;border-radius:4px;overflow:hidden">'
    for src in ACTIVE_SOURCES:
        val = _score_val(movie, src)
        w = weights.get(src, 0)
        if val is not None and w > 0:
            bar += (
                f'<div style="width:{w}%;background:{SOURCE_COLORS[src]};'
                f'opacity:{0.25 + val / 100 * 0.75:.2f};min-width:2px"></div>'
            )
    bar += "</div></div>"

    # ── Layout: poster | rank+info | scores | actions ──────────
    col_poster, col_info, col_scores, col_actions = st.columns([0.8, 3.5, 3, 1.4])

    with col_poster:
        if movie.poster_path:
            st.image(
                f"https://image.tmdb.org/t/p/w185{movie.poster_path}",
                use_container_width=True,
            )
        else:
            st.markdown(
                '<div style="background:#1e293b;border-radius:6px;height:120px;'
                "display:flex;align-items:center;justify-content:center;"
                'color:#475569;font-size:2rem">🎬</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="rank-num" style="text-align:center">#{rank}</div>'
            f'<div class="score-num" style="color:{score_color};text-align:center">'
            f"{movie.composite_score:.1f}</div>",
            unsafe_allow_html=True,
        )

    with col_info:
        platforms_html = " ".join(
            f'<span class="platform-tag">'
            f"{STREAMING_PROVIDERS[k]['emoji']} {STREAMING_PROVIDERS[k]['name']}</span>"
            for k, v in STREAMING_PROVIDERS.items()
            if v["name"] in movie.platforms
        )
        seen_badge = '<span class="badge-seen"> ✅ Vu</span>' if is_seen else ""
        wish_badge = '<span class="badge-wish"> 🔖 Wishlist</span>' if is_wish else ""
        if is_seen and user_rating:
            seen_badge = f'<span class="badge-seen"> ✅ Vu · ⭐ {user_rating}/10</span>'

        runtime_str = f"{movie.runtime} min" if movie.runtime else ""
        countries_str = ", ".join(movie.countries[:2]) if movie.countries else ""
        meta = " · ".join(
            filter(
                None,
                [
                    str(movie.year),
                    runtime_str,
                    countries_str,
                    " · ".join(movie.genres[:3]),
                ],
            )
        )
        st.markdown(
            f'<div class="movie-title">{movie.title}{seen_badge}{wish_badge}</div>'
            f'<div class="movie-meta">{meta}</div>'
            f"{platforms_html}{bar}",
            unsafe_allow_html=True,
        )
        # Synopsis hidden in expander
        if movie.overview:
            with st.expander("Synopsis"):
                st.write(movie.overview)

    with col_scores:
        # Score table: label + value, 2 columns
        left_sources = ACTIVE_SOURCES[:3]
        right_sources = ACTIVE_SOURCES[3:]
        sc_left, sc_right = st.columns(2)
        for src in left_sources:
            with sc_left:
                st.markdown(
                    f'<div style="font-size:0.68rem;color:{SOURCE_COLORS[src]}">'
                    f"{SOURCE_LABELS[src]}</div>"
                    f'<div style="font-size:0.9rem;font-weight:600;margin-bottom:6px">'
                    f"{_fmt_score(movie, src)}</div>",
                    unsafe_allow_html=True,
                )
        for src in right_sources:
            with sc_right:
                st.markdown(
                    f'<div style="font-size:0.68rem;color:{SOURCE_COLORS[src]}">'
                    f"{SOURCE_LABELS[src]}</div>"
                    f'<div style="font-size:0.9rem;font-weight:600;margin-bottom:6px">'
                    f"{_fmt_score(movie, src)}</div>",
                    unsafe_allow_html=True,
                )

    with col_actions:
        if is_seen:
            if st.button(
                "✅ Vu", key=f"unsee_{movie.tmdb_id}", help="Cliquer pour retirer"
            ):
                unmark_seen(profile_name, movie.tmdb_id)
                st.rerun()
            new_r = st.number_input(
                "Ma note /10",
                0.0,
                10.0,
                float(user_rating or 0),
                step=0.5,
                key=f"rate_{movie.tmdb_id}",
            )
            if new_r != (user_rating or 0):
                mark_seen(profile_name, movie.tmdb_id, new_r or None)
                st.rerun()
        else:
            if st.button("Marquer vu", key=f"see_{movie.tmdb_id}"):
                mark_seen(profile_name, movie.tmdb_id)
                st.rerun()
        if st.button(
            "🔖 Retirer" if is_wish else "🔖 Wishlist",
            key=f"wish_{movie.tmdb_id}",
        ):
            mark_wishlist(profile_name, movie.tmdb_id)
            st.rerun()

    st.divider()


# ══════════════════════════════════════════════════════════════
#  Profile page  (full-page modal-like experience)
# ══════════════════════════════════════════════════════════════


def profile_page():
    st.title("👤 Gestion des profils")

    profiles = list_profiles()

    col_list, col_detail = st.columns([1, 2])

    with col_list:
        st.subheader("Profils")
        if not profiles:
            st.info("Aucun profil existant.")
        else:
            selected = st.radio("Sélectionner", profiles, label_visibility="collapsed")

        st.markdown("---")
        st.subheader("Créer un profil")
        new_name = st.text_input("Nom")
        if st.button("➕ Créer", disabled=not new_name):
            if new_name in profiles:
                st.error("Ce nom existe déjà.")
            else:
                create_profile(new_name)
                st.success(f"Profil « {new_name} » créé !")
                st.rerun()

    with col_detail:
        if not profiles:
            st.info("👈 Créez votre premier profil.")
            return

        profile = get_profile(selected) or {}
        st.subheader(f"⚙️ Profil : {selected}")

        # ── Subscriptions ─────────────────────────────────────
        st.markdown("**📺 Abonnements streaming**")
        saved_subs = profile.get("subscriptions", list(STREAMING_PROVIDERS.keys()))
        new_subs = []
        sub_cols = st.columns(3)
        for i, (key, info) in enumerate(STREAMING_PROVIDERS.items()):
            with sub_cols[i % 3]:
                checked = st.checkbox(
                    f"{info['emoji']} {info['name']}",
                    value=key in saved_subs,
                    key=f"sub_{selected}_{key}",
                )
                if checked:
                    new_subs.append(key)
        if new_subs != saved_subs:
            update_profile(selected, subscriptions=new_subs)
            st.rerun()

        st.markdown("---")

        # ── Film count ────────────────────────────────────────
        st.markdown("**🎞 Quantité de films à analyser par plateforme**")
        saved_pages = profile.get("pages", 3)
        pages_labels = {
            1: "~20 films",
            2: "~40 films",
            3: "~60 films",
            5: "~100 films",
            8: "~160 films",
        }
        pages_val = st.select_slider(
            "Films analysés par plateforme",
            options=[1, 2, 3, 5, 8],
            value=saved_pages,
            format_func=lambda v: pages_labels.get(v, f"~{v * 20} films"),
            label_visibility="collapsed",
        )
        st.caption(
            "Plus il y a de films analysés, plus le classement est complet, mais plus c'est long à charger."
        )
        if pages_val != saved_pages:
            update_profile(selected, pages=pages_val)

        st.markdown("---")

        # ── Delete ────────────────────────────────────────────
        st.markdown("**🗑 Zone dangereuse**")
        if st.button(f"Supprimer le profil « {selected} »", type="secondary"):
            delete_profile(selected)
            st.success("Profil supprimé.")
            st.rerun()


# ══════════════════════════════════════════════════════════════
#  Filters
# ══════════════════════════════════════════════════════════════


def filters_panel(movies: list[Movie]) -> dict:
    with st.expander("🔍 Filtres", expanded=False):
        # Row 1 — content filters
        c1, c2, c3, c4, c5 = st.columns(5)

        all_genres = sorted({g for m in movies for g in m.genres})
        all_countries = sorted({c for m in movies for c in m.countries})
        years = [m.year for m in movies if m.year]
        runtimes = [m.runtime for m in movies if m.runtime]

        with c1:
            genres = st.multiselect("Genre", all_genres, key="f_genre")
        with c2:
            countries = st.multiselect("Pays", all_countries, key="f_country")
        with c3:
            year_range = (
                st.slider(
                    "Année",
                    min(years),
                    max(years),
                    (min(years), max(years)),
                    key="f_year",
                )
                if years
                else (0, 9999)
            )
        with c4:
            rt_range = (
                st.slider(
                    "Durée (min)",
                    min(runtimes),
                    max(runtimes),
                    (min(runtimes), max(runtimes)),
                    key="f_rt",
                )
                if runtimes
                else (0, 9999)
            )
        with c5:
            score_min = st.slider("Score min", 0, 100, 0, key="f_score")

        # Row 2 — status filters
        st.markdown("**Statut**")
        cs1, cs2, cs3, _ = st.columns([1, 1, 1, 3])
        with cs1:
            seen_only = st.toggle("✅ Vus seulement", key="f_seen_only")
        with cs2:
            wish_only = st.toggle("🔖 Wishlist seulement", key="f_wish_only")
        with cs3:
            hide_seen = st.toggle("🙈 Masquer vus", key="f_hide_seen")

        if seen_only or wish_only:
            hide_seen = False

    return dict(
        genres=genres,
        countries=countries,
        year_range=year_range,
        rt_range=rt_range,
        score_min=score_min,
        seen_only=seen_only,
        wish_only=wish_only,
        hide_seen=hide_seen,
    )


def apply_filters(
    movies: list[Movie],
    f: dict,
    seen_ids: set[int],
    wish_ids: set[int],
) -> list[Movie]:
    out = []
    for m in movies:
        if f["seen_only"] and m.tmdb_id not in seen_ids:
            continue
        if f["wish_only"] and m.tmdb_id not in wish_ids:
            continue
        if f["hide_seen"] and m.tmdb_id in seen_ids:
            continue
        if f["genres"] and not any(g in m.genres for g in f["genres"]):
            continue
        if f["countries"] and not any(c in m.countries for c in f["countries"]):
            continue
        if m.year and not (f["year_range"][0] <= m.year <= f["year_range"][1]):
            continue
        if m.runtime and not (f["rt_range"][0] <= m.runtime <= f["rt_range"][1]):
            continue
        if m.composite_score < f["score_min"]:
            continue
        out.append(m)
    return out


# ══════════════════════════════════════════════════════════════
#  Sidebar (ranking page)
# ══════════════════════════════════════════════════════════════


def ranking_sidebar() -> tuple[Optional[str], dict, list[str], int]:
    """Returns (profile_name, weights, platforms, pages)."""

    profiles = list_profiles()

    # Profile selector
    st.sidebar.markdown("### 👤 Profil actif")
    if not profiles:
        st.sidebar.warning("Aucun profil.")
        st.sidebar.page_link("app.py", label="→ Créer un profil", icon="➕")
        return None, DEFAULT_WEIGHTS.copy(), [], 3

    profile_name = st.sidebar.selectbox(
        "Profil", profiles, label_visibility="collapsed"
    )
    profile = get_profile(profile_name) or {}

    st.sidebar.divider()

    # Quick info
    subs = profile.get("subscriptions", list(STREAMING_PROVIDERS.keys()))
    pages = profile.get("pages", 3)
    st.sidebar.caption(
        "📺 "
        + " ".join(STREAMING_PROVIDERS[k]["emoji"] for k in subs)
        + f"  ·  ~{pages * 20} films/plateforme"
    )

    st.sidebar.divider()

    # Weights
    st.sidebar.markdown("### ⚖️ Poids des notes")
    saved_w = profile.get("weights", DEFAULT_WEIGHTS.copy())
    # Ensure all sources present and total = 100
    for s in ACTIVE_SOURCES:
        if s not in saved_w:
            saved_w[s] = DEFAULT_WEIGHTS.get(s, 0)
    # Fix total if needed
    total = sum(saved_w.values())
    if total != 100 and total > 0:
        diff = 100 - total
        saved_w[max(saved_w, key=saved_w.get)] += diff

    new_weights = weight_sliders(saved_w)
    if new_weights != saved_w:
        update_profile(profile_name, weights=new_weights)

    return profile_name, new_weights, subs, pages


# ══════════════════════════════════════════════════════════════
#  Main ranking page
# ══════════════════════════════════════════════════════════════


def ranking_page():
    profile_name, weights, platforms, pages = ranking_sidebar()

    if profile_name is None:
        st.info("👈 Créez un profil pour commencer.")
        return

    # Header
    c_title, c_cache, c_refresh = st.columns([4, 2, 1])
    with c_title:
        st.title("🎬 Top Films en Streaming")
    with c_cache:
        st.caption(_cache_age())
    with c_refresh:
        if st.button("🔄 Actualiser"):
            ck = cache_key_for(platforms, True, pages)
            invalidate(ck)
            st.rerun()

    if not platforms:
        st.warning("Aucune plateforme sélectionnée — modifiez votre profil.")
        return

    if not TMDB_KEY:
        st.error("TMDB_API_KEY manquante dans le fichier .env")
        return

    # Load or build cache
    ck = cache_key_for(platforms, True, pages)
    movies = cache_load(ck)

    if movies is None:
        with st.spinner(
            "Collecte et enrichissement des films… (première fois uniquement)"
        ):
            tmdb = TMDBClient(TMDB_KEY)
            omdb = OMDbClient(OMDB_KEY)
            sc = SensCritiqueClient()
            allocine = AllocineClient()

            candidates = collect(platforms, tmdb, pages=pages)
            movies = enrich_and_rank(
                candidates,
                tmdb,
                omdb,
                sc,
                weights=weights,
                allocine=allocine,
                top_n=max(50, pages * len(platforms) * 3),
                workers=8,
            )
            cache_save(ck, movies)

    # Recompute scores with current weights (instant, no API calls)
    for m in movies:
        m.composite_score = compute_composite_score(m.scores, weights)
    movies = sorted(movies, key=lambda m: m.composite_score, reverse=True)

    seen_ids = get_seen_ids(profile_name)
    wish_ids = get_wishlist_ids(profile_name)

    # Filters
    f = filters_panel(movies)
    filtered = apply_filters(movies, f, seen_ids, wish_ids)

    # Tabs
    tab_rank, tab_wish, tab_seen = st.tabs(
        [
            f"🏆 Classement ({len(filtered)}/{len(movies)})",
            f"🔖 Wishlist ({len(wish_ids)})",
            f"✅ Vus ({len(seen_ids)})",
        ]
    )

    with tab_rank:
        if not filtered:
            st.info("Aucun film ne correspond aux filtres actifs.")
        for rank, movie in enumerate(filtered[:50], 1):
            movie_card(rank, movie, weights, seen_ids, wish_ids, profile_name)

    with tab_wish:
        wish_movies = [m for m in movies if m.tmdb_id in wish_ids]
        if not wish_movies:
            st.info("Votre wishlist est vide.")
        for rank, m in enumerate(wish_movies, 1):
            movie_card(rank, m, weights, seen_ids, wish_ids, profile_name)

    with tab_seen:
        seen_movies = [m for m in movies if m.tmdb_id in seen_ids]
        if not seen_movies:
            st.info("Aucun film marqué comme vu.")
        for rank, m in enumerate(seen_movies, 1):
            movie_card(rank, m, weights, seen_ids, wish_ids, profile_name)


# ══════════════════════════════════════════════════════════════
#  Router
# ══════════════════════════════════════════════════════════════


def main():
    # Simple page routing via sidebar radio
    st.sidebar.markdown("## 🎬 Streaming Ranker")
    page = st.sidebar.radio(
        "Navigation",
        ["🏆 Classement", "👤 Profils"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    if page == "🏆 Classement":
        ranking_page()
    else:
        profile_page()


if __name__ == "__main__":
    main()
