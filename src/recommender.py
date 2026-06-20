from __future__ import annotations

import difflib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import COLLABORATIVE_WEIGHT, CONTENT_WEIGHT, MODEL_PATH, POPULARITY_WEIGHT
from src.preprocessing import ProcessedData, preprocess_interactions


class HybridMusicRecommender:
    def __init__(
        self,
        collaborative_weight: float = COLLABORATIVE_WEIGHT,
        content_weight: float = CONTENT_WEIGHT,
        popularity_weight: float = POPULARITY_WEIGHT,
    ) -> None:
        total = collaborative_weight + content_weight + popularity_weight
        if total <= 0:
            raise ValueError("At least one recommendation weight must be positive.")
        self.collaborative_weight = collaborative_weight / total
        self.content_weight = content_weight / total
        self.popularity_weight = popularity_weight / total
        self.processed: ProcessedData | None = None
        self.metadata: pd.DataFrame | None = None
        self.content_matrix = None
        self.vectorizer: TfidfVectorizer | None = None
        self.name_to_index: dict[str, int] = {}

    def fit(self, interactions_df: pd.DataFrame) -> "HybridMusicRecommender":
        self.processed = preprocess_interactions(interactions_df)
        self.metadata = self.processed.metadata.sort_values("artist_idx").reset_index(drop=True)
        self.name_to_index = {
            str(row.artist_name).casefold(): int(row.artist_idx)
            for row in self.metadata.itertuples()
        }
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True)
        self.content_matrix = self.vectorizer.fit_transform(self.metadata.artist_name.fillna(""))
        max_plays = float(self.metadata.total_play_count.max())
        max_listeners = float(self.metadata.unique_listeners.max())
        self.metadata["popularity_score"] = (
            0.65 * np.log1p(self.metadata.total_play_count) / max(np.log1p(max_plays), 1)
            + 0.35 * self.metadata.unique_listeners / max(max_listeners, 1)
        )
        return self

    def _require_fit(self) -> None:
        if self.processed is None or self.metadata is None:
            raise RuntimeError("Fit the recommender before requesting recommendations.")

    def _resolve_artist(self, artist_name: str) -> int:
        self._require_fit()
        key = artist_name.strip().casefold()
        if key in self.name_to_index:
            return self.name_to_index[key]
        matches = self.search_artists(artist_name, limit=5)
        suggestion = f" Did you mean: {', '.join(matches)}?" if matches else ""
        raise ValueError(f"Artist '{artist_name}' was not found.{suggestion}")

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        finite = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        low, high = finite.min(initial=0), finite.max(initial=0)
        return (finite - low) / (high - low) if high > low else np.zeros_like(finite)

    def _get_collaborative_scores(self, artist_idx: int) -> np.ndarray:
        self._require_fit()
        query = self.processed.artist_user_matrix[artist_idx]
        return cosine_similarity(query, self.processed.artist_user_matrix).ravel()

    def _get_content_scores(self, artist_idx: int) -> np.ndarray:
        self._require_fit()
        return cosine_similarity(self.content_matrix[artist_idx], self.content_matrix).ravel()

    def _get_popularity_scores(self) -> np.ndarray:
        self._require_fit()
        return self.metadata.popularity_score.to_numpy(dtype=float)

    def recommend_artists(self, artist_name: str, top_k: int = 10) -> pd.DataFrame:
        artist_idx = self._resolve_artist(artist_name)
        collaborative = self._normalize_scores(self._get_collaborative_scores(artist_idx))
        content = self._normalize_scores(self._get_content_scores(artist_idx))
        popularity = self._normalize_scores(self._get_popularity_scores())
        hybrid = (
            self.collaborative_weight * collaborative
            + self.content_weight * content
            + self.popularity_weight * popularity
        )
        hybrid[artist_idx] = -1
        count = min(max(int(top_k), 1), len(hybrid) - 1)
        chosen = np.argsort(hybrid)[::-1][:count]
        result = self.metadata.set_index("artist_idx").loc[chosen].copy()
        result["hybrid_score"] = hybrid[chosen]
        result["collaborative_score"] = collaborative[chosen]
        result["content_score"] = content[chosen]
        result["popularity_score"] = popularity[chosen]
        result["reason"] = [self._reason(c, t, p, artist_name) for c, t, p in zip(collaborative[chosen], content[chosen], popularity[chosen])]
        result.insert(0, "rank", range(1, len(result) + 1))
        return result.reset_index(drop=True)[[
            "rank", "artist_name", "hybrid_score", "collaborative_score", "content_score",
            "popularity_score", "total_play_count", "unique_listeners", "reason",
        ]]

    @staticmethod
    def _reason(collaborative: float, content: float, popularity: float, seed: str) -> str:
        if collaborative >= max(content, popularity):
            return f"Shares strong listener patterns with {seed}"
        if content >= popularity:
            return f"Similar artist-name profile with listeners related to {seed}"
        return "Popular among listeners with related taste"

    def search_artists(self, query: str, limit: int = 10) -> list[str]:
        self._require_fit()
        names = self.metadata.artist_name.astype(str).tolist()
        term = query.strip().casefold()
        contains = [name for name in names if term in name.casefold()]
        fuzzy = difflib.get_close_matches(query, names, n=limit, cutoff=0.35)
        return list(dict.fromkeys([*contains, *fuzzy]))[:limit]

    def get_artist_stats(self, artist_name: str) -> dict[str, object]:
        idx = self._resolve_artist(artist_name)
        row = self.metadata[self.metadata.artist_idx == idx].iloc[0]
        return row.to_dict()

    def get_dataset_summary(self) -> dict[str, int]:
        self._require_fit()
        return {
            "users": int(self.processed.interactions.user_id.nunique()),
            "artists": int(self.processed.interactions.artist_id.nunique()),
            "interactions": int(len(self.processed.interactions)),
            "total_plays": int(self.processed.interactions.play_count.sum()),
        }

    def get_popular_artists(self, top_k: int = 10) -> pd.DataFrame:
        self._require_fit()
        return self.metadata.nlargest(top_k, "popularity_score").reset_index(drop=True)

    def save_model(self, path: str | Path = MODEL_PATH) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load_model(cls, path: str | Path = MODEL_PATH) -> "HybridMusicRecommender":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("Saved object is not a HybridMusicRecommender.")
        return model

