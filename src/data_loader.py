from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import DATA_DIR, DEMO_DATA_PATH, RANDOM_STATE

REQUIRED_COLUMNS = ["user_id", "artist_id", "artist_name", "play_count"]
ALIASES = {
    "user": "user_id", "userid": "user_id", "userID": "user_id",
    "artist": "artist_name", "name": "artist_name",
    "artistid": "artist_id", "artistID": "artist_id",
    "plays": "play_count", "weight": "play_count", "playcount": "play_count",
}

ARTIST_GROUPS = {
    "afrobeats": ["Tems", "Wizkid", "Burna Boy", "Davido", "Ayra Starr", "Rema", "Asake", "Tyla", "Fireboy DML", "Omah Lay", "Joeboy", "CKay", "Kizz Daniel", "Simi", "Tiwa Savage"],
    "pop": ["Ariana Grande", "Taylor Swift", "Billie Eilish", "Ed Sheeran", "Dua Lipa", "Justin Bieber", "Bruno Mars", "Olivia Rodrigo", "Miley Cyrus", "Lana Del Rey", "Adele", "Coldplay"],
    "hiphop": ["Drake", "Kendrick Lamar", "Travis Scott", "Post Malone", "Doja Cat", "Nicki Minaj", "Future", "Metro Boomin", "Central Cee", "Stormzy", "Dave", "J Hus"],
    "rnb": ["SZA", "The Weeknd", "Beyonce", "Rihanna", "Khalid", "Chris Brown", "Tems", "Amaarae", "Tyla", "Ariana Grande"],
    "west_africa": ["Sarkodie", "Shatta Wale", "Stonebwoy", "Black Sherif", "Amaarae", "Burna Boy", "Wizkid", "Tiwa Savage", "Davido", "Asake"],
    "uk_rap": ["Central Cee", "Stormzy", "Dave", "J Hus", "Black Sherif", "Drake", "Future", "Skepta", "Little Simz", "Headie One"],
}


def generate_demo_dataset(path: Path = DEMO_DATA_PATH, users: int = 180) -> pd.DataFrame:
    """Create overlapping listener cohorts, not independent random interactions."""
    rng = np.random.default_rng(RANDOM_STATE)
    all_artists = list(dict.fromkeys(a for group in ARTIST_GROUPS.values() for a in group))
    artist_ids = {name: f"artist_{i:03d}" for i, name in enumerate(all_artists, 1)}
    group_names = list(ARTIST_GROUPS)
    rows: list[dict[str, Any]] = []

    for user_index in range(1, users + 1):
        primary = group_names[(user_index - 1) % len(group_names)]
        secondary = group_names[(user_index * 3 + 1) % len(group_names)]
        primary_choices = rng.choice(ARTIST_GROUPS[primary], size=8, replace=False)
        secondary_pool = [a for a in ARTIST_GROUPS[secondary] if a not in primary_choices]
        secondary_choices = rng.choice(secondary_pool, size=min(3, len(secondary_pool)), replace=False)
        for rank, artist in enumerate([*primary_choices, *secondary_choices]):
            affinity = 1.75 if rank < len(primary_choices) else 0.9
            plays = max(1, int(rng.lognormal(mean=3.25 + affinity / 3, sigma=0.65)))
            rows.append({"user_id": f"user_{user_index:03d}", "artist_id": artist_ids[artist], "artist_name": artist, "play_count": plays})

    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: ALIASES.get(column, ALIASES.get(str(column).lower(), str(column).lower())) for column in frame.columns}
    frame = frame.rename(columns=renamed)
    if "artist_id" not in frame and "artist_name" in frame:
        frame["artist_id"] = frame["artist_name"].astype(str).str.lower().str.replace(r"\W+", "_", regex=True)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    return frame[REQUIRED_COLUMNS].copy()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def load_interactions(uploaded_file: Any | None = None) -> tuple[pd.DataFrame, dict[str, str]]:
    if uploaded_file is not None:
        suffix = Path(getattr(uploaded_file, "name", "upload.csv")).suffix.lower()
        frame = pd.read_parquet(uploaded_file) if suffix == ".parquet" else pd.read_csv(uploaded_file)
        name = getattr(uploaded_file, "name", "uploaded dataset")
        return _normalize_columns(frame), {"kind": "upload", "label": f"Advanced upload · {name}"}

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    preferred = [DATA_DIR / name for name in ("play_counts.csv", "play_counts.parquet", "lastfm.csv", "lastfm.parquet")]
    candidates = [p for p in preferred if p.exists()]
    candidates += [p for p in sorted(DATA_DIR.glob("*.csv")) + sorted(DATA_DIR.glob("*.parquet")) if p not in candidates and p != DEMO_DATA_PATH]
    for path in candidates:
        try:
            return _normalize_columns(_read(path)), {"kind": "local", "label": f"Last.fm local file · {path.name}"}
        except (ValueError, OSError, pd.errors.ParserError):
            continue

    frame = _read(DEMO_DATA_PATH) if DEMO_DATA_PATH.exists() else generate_demo_dataset()
    return _normalize_columns(frame), {"kind": "demo", "label": "Demo listening history"}


if __name__ == "__main__":
    demo = generate_demo_dataset()
    print(f"Generated {len(demo):,} interactions for {demo.user_id.nunique()} users and {demo.artist_id.nunique()} artists.")
