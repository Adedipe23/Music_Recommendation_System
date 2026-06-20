# Resonance: Hybrid Music Recommendation System

Resonance is a Spotify-inspired Streamlit app for explainable artist and song discovery. Pick an artist, choose how many results you want, and explore rich recommendation cards with score breakdowns and direct Last.fm and YouTube links.

## Run with Docker

```bash
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501). The built-in demo dataset is generated automatically, so no upload or API key is needed for artist recommendations.

To enable song recommendations, copy `.env.example` to `.env`, add a [Last.fm API key](https://www.last.fm/api/account/create), and restart Compose.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python -m src.evaluation
streamlit run app.py
```

## Recommendation model

The final score combines three normalized signals:

```text
0.6 × collaborative score + 0.3 × content score + 0.1 × popularity score
```

- **Collaborative:** cosine similarity over a sparse artist–listener matrix with `log1p(play_count)` interaction strength.
- **Content:** character n-gram TF-IDF over artist names, designed as a no-metadata baseline.
- **Popularity:** total play count and unique-listener reach, used as a stable fallback.

The app computes query-to-catalog similarity on demand and avoids a dense all-pairs matrix.

## Data

The loader looks in `data/` for `play_counts.csv`, `play_counts.parquet`, `lastfm.csv`, `lastfm.parquet`, or another compatible CSV/Parquet file. Required fields are:

| Field | Meaning |
|---|---|
| `user_id` | Listener identifier |
| `artist_id` | Artist identifier |
| `artist_name` | Display name |
| `play_count` | Positive implicit-feedback strength |

If no compatible local file exists, the app uses `data/demo_listening_history.csv`. It contains overlapping Afrobeats, pop, hip-hop, R&B, West African, and UK rap listener cohorts rather than random independent rows.

An optional uploader lives under **Advanced Dataset Settings**; it is not part of the normal discovery flow.

## Project map

- `app.py` — five-page Streamlit product UI with Artist Recommender as the landing page
- `train.py` — model training and persistence
- `src/recommender.py` — hybrid recommender
- `src/data_loader.py` — real-data discovery and demo generation
- `src/lastfm_api.py` — fault-tolerant song metadata client
- `src/evaluation.py` — hold-one-out implicit-feedback metrics
- `reports/evaluation_results.md` — latest evaluation snapshot

## Notes and limitations

Artist recommendation is Phase 1 because the referenced Last.fm dataset is artist-level. Live song similarity is Phase 2 and requires Last.fm. Name-only TF-IDF is intentionally modest; production versions should add tags, embeddings, recency, diversity, and richer artwork metadata.
