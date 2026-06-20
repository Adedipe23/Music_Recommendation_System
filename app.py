from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from config import EVALUATION_REPORT_PATH, MODEL_PATH
from src.data_loader import load_interactions
from src.lastfm_api import LastFMClient, LastFMError
from src.recommender import HybridMusicRecommender
from src.utils import (
    compact_number,
    lastfm_artist_url,
    lastfm_track_url,
    safe_html,
    youtube_artist_url,
    youtube_track_url,
)

load_dotenv()
st.set_page_config(page_title="Resonance", page_icon="♫", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
:root { --green:#1ed760; --panel:#181818; --soft:#242424; --muted:#a7a7a7; }
[data-testid="stAppViewContainer"] { background:radial-gradient(circle at 70% -20%,#193626 0,transparent 35%),#0b0b0b; color:#fff; }
[data-testid="stSidebar"] { background:#050505; border-right:1px solid #202020; }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1280px; padding-top:2.2rem; }
h1,h2,h3 { letter-spacing:-.035em; }
.eyebrow { color:var(--green); font-size:.75rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
.hero { padding:2rem 2.2rem; border:1px solid #2c2c2c; border-radius:24px; background:linear-gradient(135deg,rgba(30,215,96,.15),rgba(24,24,24,.86) 55%); margin-bottom:1.5rem; }
.hero h1 { font-size:clamp(2.15rem,4vw,4rem); line-height:1; margin:.4rem 0 1rem; max-width:900px; }
.hero p { color:#c6c6c6; font-size:1.05rem; max-width:700px; }
.metric-strip { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:1rem 0 2rem; }
.metric { background:var(--panel); border:1px solid #292929; border-radius:16px; padding:1rem 1.1rem; }
.metric small { color:var(--muted); display:block; margin-bottom:.3rem; }
.metric strong { font-size:1.45rem; }
.artist-card,.song-card { display:grid; grid-template-columns:86px 1fr auto; gap:18px; align-items:center; background:linear-gradient(115deg,#191919,#121212); border:1px solid #2b2b2b; border-radius:18px; padding:14px; margin:0 0 13px; box-shadow:0 12px 35px rgba(0,0,0,.18); transition:.2s ease; }
.artist-card:hover,.song-card:hover { transform:translateY(-2px); border-color:#444; background:#1d1d1d; }
.cover { width:86px; height:86px; object-fit:cover; border-radius:13px; background:linear-gradient(145deg,#285d3b,#111); }
.placeholder { width:86px; height:86px; border-radius:13px; background:radial-gradient(circle at 25% 20%,#38d878,#193c28 48%,#0d140f); display:grid; place-items:center; font-size:2.15rem; font-weight:800; color:#fff; box-shadow:inset 0 0 0 1px rgba(255,255,255,.08); }
.card-title { font-size:1.22rem; font-weight:800; margin-bottom:3px; }
.card-subtitle { color:var(--muted); font-size:.88rem; }
.rank { color:var(--green); font-size:.72rem; font-weight:900; letter-spacing:.1em; }
.reason { margin-top:8px; color:#d4d4d4; font-size:.9rem; }
.scores { display:flex; flex-wrap:wrap; gap:7px; margin-top:10px; }
.pill { border:1px solid #343434; background:#232323; border-radius:999px; padding:4px 8px; color:#bdbdbd; font-size:.72rem; }
.pill b { color:#fff; }
.card-actions { display:flex; flex-direction:column; gap:8px; min-width:145px; }
.action { text-decoration:none!important; text-align:center; border-radius:999px; padding:8px 13px; font-size:.76rem; font-weight:800; color:white!important; border:1px solid #525252; }
.action.primary { background:var(--green); color:#07150c!important; border-color:var(--green); }
.action:hover { border-color:white; }
.notice { background:#171717; border:1px solid #303030; border-left:3px solid var(--green); border-radius:12px; padding:1rem; color:#cacaca; }
.dataset-badge { background:#111d15; border:1px solid #21452d; border-radius:12px; padding:.8rem .9rem; color:#d6f8e1; font-size:.84rem; line-height:1.35; }
.dataset-badge b { color:var(--green); display:block; font-size:.68rem; letter-spacing:.11em; text-transform:uppercase; margin-bottom:.2rem; }
div.stButton > button, div.stButton > button:focus, div.stButton > button:active { border-radius:999px; min-height:44px; font-weight:800; background:var(--green)!important; color:#07150c!important; border:0!important; box-shadow:none!important; }
div.stButton > button:hover { background:#35e576!important; color:#07150c!important; }
@media(max-width:760px){.metric-strip{grid-template-columns:repeat(2,1fr)}.artist-card,.song-card{grid-template-columns:70px 1fr}.cover,.placeholder{width:70px;height:70px}.card-actions{grid-column:1/-1;flex-direction:row}.action{flex:1}.scores{display:none}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model_and_data() -> tuple[HybridMusicRecommender, pd.DataFrame, dict[str, str]]:
    frame, source = load_interactions()
    if MODEL_PATH.exists():
        try:
            model = HybridMusicRecommender.load_model(MODEL_PATH)
            return model, frame, source
        except Exception:
            pass
    model = HybridMusicRecommender().fit(frame)
    model.save_model()
    return model, frame, source


@st.cache_data(ttl=3600, show_spinner=False)
def artist_metadata(name: str) -> dict[str, str | None]:
    try:
        return LastFMClient().get_artist_info(name)
    except LastFMError:
        return {"name": name, "url": None, "image": None}


@st.cache_data(ttl=1800, show_spinner=False)
def top_tracks(name: str) -> list[dict]:
    return LastFMClient().get_top_tracks(name)


@st.cache_data(ttl=1800, show_spinner=False)
def similar_tracks(artist: str, track: str, limit: int) -> list[dict]:
    return LastFMClient().get_similar_tracks(artist, track, limit)


def metric_strip(summary: dict[str, int]) -> None:
    labels = [("Listeners", summary["users"]), ("Artists", summary["artists"]), ("Interactions", summary["interactions"]), ("Total plays", summary["total_plays"])]
    markup = "".join(f'<div class="metric"><small>{label}</small><strong>{compact_number(value)}</strong></div>' for label, value in labels)
    st.markdown(f'<div class="metric-strip">{markup}</div>', unsafe_allow_html=True)


def image_markup(url: str | None, label: str) -> str:
    if url:
        return f'<img class="cover" src="{safe_html(url)}" alt="{safe_html(label)} cover" loading="lazy">'
    return '<div class="placeholder" aria-label="No image available">♫</div>'


def artist_card(row: pd.Series) -> str:
    name = str(row.artist_name)
    meta = artist_metadata(name) if os.getenv("LASTFM_API_KEY") else {"image": None, "url": None}
    lastfm_url = meta.get("url") or lastfm_artist_url(name)
    return f'''<div class="artist-card">
      {image_markup(meta.get("image"), name)}
      <div><div class="rank">#{int(row['rank'])} RECOMMENDED ARTIST</div><div class="card-title">{safe_html(name)}</div>
      <div class="card-subtitle">{compact_number(row.total_play_count)} plays · {compact_number(row.unique_listeners)} listeners</div>
      <div class="reason">{safe_html(row.reason)}</div>
      <div class="scores"><span class="pill">Hybrid <b>{row.hybrid_score:.3f}</b></span><span class="pill">Collaborative <b>{row.collaborative_score:.3f}</b></span><span class="pill">Content <b>{row.content_score:.3f}</b></span><span class="pill">Popularity <b>{row.popularity_score:.3f}</b></span></div></div>
      <div class="card-actions"><a class="action primary" href="{lastfm_url}" target="_blank" rel="noopener">Open on Last.fm</a><a class="action" href="{youtube_artist_url(name)}" target="_blank" rel="noopener">Search on YouTube</a></div>
    </div>'''


def song_card(track: dict, rank: int) -> str:
    name, artist = str(track.get("name", "Unknown track")), str(track.get("artist", "Unknown artist"))
    lastfm = track.get("url") or lastfm_track_url(artist, name)
    match = float(track.get("match", 0) or 0)
    return f'''<div class="song-card">{image_markup(track.get("image"), name)}
      <div><div class="rank">TRACK #{rank}</div><div class="card-title">{safe_html(name)}</div><div class="card-subtitle">{safe_html(artist)}</div>
      <div class="scores"><span class="pill">Match <b>{match:.0%}</b></span><span class="pill">Listeners <b>{compact_number(track.get('listeners'))}</b></span><span class="pill">Plays <b>{compact_number(track.get('playcount'))}</b></span></div></div>
      <div class="card-actions"><a class="action primary" href="{lastfm}" target="_blank" rel="noopener">Open on Last.fm</a><a class="action" href="{youtube_track_url(artist, name)}" target="_blank" rel="noopener">Watch on YouTube</a></div></div>'''


model, interactions, source = load_model_and_data()
summary = model.get_dataset_summary()

if "custom_model" in st.session_state:
    model = st.session_state.custom_model
    interactions = st.session_state.custom_data
    source = st.session_state.custom_source
    summary = model.get_dataset_summary()

with st.sidebar:
    st.markdown("## ♫ Resonance")
    page = st.radio("Navigate", ["Artist Recommender", "Song Recommender", "Dataset Insights", "Evaluation Results", "About"], label_visibility="collapsed")
    st.divider()
    st.markdown(
        f'<div class="dataset-badge"><b>Dataset</b>{safe_html(source["label"])}<br>'
        f'<span style="color:#8eaa97">{summary["users"]:,} listeners · {summary["artists"]:,} artists</span></div>',
        unsafe_allow_html=True,
    )

if page == "Artist Recommender":
    st.markdown('<section class="hero"><div class="eyebrow">Artist recommender</div><h1>Resonance: Hybrid Music Recommendation System</h1><p>Discover artists through real listening patterns.</p></section>', unsafe_allow_html=True)
    names = sorted(model.metadata.artist_name.astype(str).tolist())
    c1, c2 = st.columns([3, 1])
    selected = c1.selectbox("Search or select an artist", names, index=names.index("Tems") if "Tems" in names else 0, placeholder="Search artists…")
    top_k = c2.selectbox("Top K", [5, 10, 20], index=1)
    if st.button("Recommend artists", type="primary", use_container_width=True):
        st.session_state["artist_results"] = model.recommend_artists(selected, top_k)
        st.session_state["artist_seed"] = selected
    if st.session_state.get("artist_seed") == selected and "artist_results" in st.session_state:
        stats = model.get_artist_stats(selected)
        st.markdown(f"### Because you chose {safe_html(selected)}")
        st.caption(f"{compact_number(stats['total_play_count'])} plays · {compact_number(stats['unique_listeners'])} listeners in this dataset")
        for _, recommendation in st.session_state.artist_results.iterrows():
            st.markdown(artist_card(recommendation), unsafe_allow_html=True)
        with st.expander("View score table"):
            score_columns = ["rank", "artist_name", "hybrid_score", "collaborative_score", "content_score", "popularity_score", "total_play_count", "unique_listeners", "reason"]
            st.dataframe(st.session_state.artist_results[score_columns], hide_index=True, use_container_width=True)

elif page == "Song Recommender":
    st.markdown('<div class="eyebrow">Powered by Last.fm</div>', unsafe_allow_html=True)
    st.title("Turn one track into a listening trail")
    if not LastFMClient().enabled:
        st.markdown('<div class="notice"><b>Song recommendation requires a Last.fm API key.</b><br>Add it to <code>.env</code> to enable this feature.</div>', unsafe_allow_html=True)
    else:
        artist_query = st.text_input("Artist name", value="Tems", placeholder="Type an artist name")
        if artist_query:
            try:
                tracks = top_tracks(artist_query.strip())
                if not tracks:
                    st.warning("No tracks were found for that artist.")
                else:
                    selected_track = st.selectbox("Choose a track", [track["name"] for track in tracks])
                    count = st.selectbox("Number of similar songs", [5, 10, 20], index=1)
                    if st.button("Recommend songs", type="primary", use_container_width=True):
                        with st.spinner("Following the signal…"):
                            st.session_state["song_results"] = similar_tracks(artist_query.strip(), selected_track, count)
                            st.session_state["song_seed"] = (artist_query.strip(), selected_track)
                    if st.session_state.get("song_seed") == (artist_query.strip(), selected_track):
                        st.markdown(f"### Songs like {safe_html(selected_track)}")
                        for rank, track in enumerate(st.session_state.get("song_results", []), 1):
                            st.markdown(song_card(track, rank), unsafe_allow_html=True)
            except LastFMError as exc:
                st.error(str(exc))

elif page == "Dataset Insights":
    st.markdown('<div class="eyebrow">Dataset insights</div>', unsafe_allow_html=True)
    st.title("What people are listening to")
    metric_strip(summary)
    artist_stats = model.metadata.sort_values("total_play_count", ascending=False).head(15)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(artist_stats, x="total_play_count", y="artist_name", orientation="h", title="Top artists by play count", template="plotly_dark").update_layout(yaxis={"categoryorder":"total ascending"}), use_container_width=True)
    listener_stats = model.metadata.sort_values("unique_listeners", ascending=False).head(15)
    c2.plotly_chart(px.bar(listener_stats, x="unique_listeners", y="artist_name", orientation="h", title="Top artists by unique listeners", template="plotly_dark").update_layout(yaxis={"categoryorder":"total ascending"}), use_container_width=True)
    st.plotly_chart(px.histogram(interactions, x="play_count", nbins=40, title="Distribution of play counts", template="plotly_dark", log_y=True), use_container_width=True)
    st.caption(f"Active dataset: {source['label']}")

elif page == "Evaluation Results":
    st.markdown('<div class="eyebrow">Offline evaluation</div>', unsafe_allow_html=True)
    st.title("How well does Resonance recover taste?")
    if EVALUATION_REPORT_PATH.exists():
        st.markdown(EVALUATION_REPORT_PATH.read_text(encoding="utf-8"))
    else:
        st.info("Run `python -m src.evaluation` to generate the evaluation report for the active dataset.")
    st.caption("Implicit-feedback metrics are directional: listening does not always mean preference, and missing plays are not necessarily dislikes.")

else:
    st.markdown('<div class="eyebrow">About the project</div>', unsafe_allow_html=True)
    st.title("Built for explainable discovery")
    st.markdown("""
Resonance is an artist-first recommendation system because the offline Last.fm dataset records **user–artist play counts**, not individual song histories. Artist recommendations therefore work locally and immediately. Song discovery is an optional second phase powered by live Last.fm metadata.

The hybrid ranker combines audience overlap, a lightweight content profile, and popularity. It is deliberately laptop-friendly: similarities are computed from sparse matrices only when requested rather than stored as a dense all-pairs catalog.

**Current limitations:** artist names are only a baseline content signal; play counts can overstate passive listening; and demo data cannot represent the long tail of a production catalog.

**Where it can go next:** Last.fm tags, richer artist embeddings, time-aware preference decay, personalized user profiles, diversity reranking, and album artwork from a dedicated metadata provider.
""")
    st.divider()
    with st.expander("Advanced Dataset Settings"):
        st.caption("Optional: temporarily use a compatible CSV or Parquet file for this browser session.")
        upload = st.file_uploader("Choose a listening-history file", type=["csv", "parquet"], help="Required fields: user_id, artist_id, artist_name, play_count")
        if upload and st.button("Load advanced dataset"):
            try:
                frame, upload_source = load_interactions(upload)
                st.session_state["custom_model"] = HybridMusicRecommender().fit(frame)
                st.session_state["custom_data"] = frame
                st.session_state["custom_source"] = upload_source
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(f"That dataset could not be loaded: {exc}")
