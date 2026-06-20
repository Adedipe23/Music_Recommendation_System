from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import EVALUATION_REPORT_PATH, RANDOM_STATE
from src.data_loader import load_interactions
from src.recommender import HybridMusicRecommender


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    return len(set(recommended[:k]) & relevant) / max(k, 1)


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    return len(set(recommended[:k]) & relevant) / max(len(relevant), 1)


def average_precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    hits, score = 0, 0.0
    for rank, item in enumerate(recommended[:k], 1):
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / max(min(len(relevant), k), 1)


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / np.log2(rank + 1) for rank, item in enumerate(recommended[:k], 1) if item in relevant)
    ideal = sum(1 / np.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return float(dcg / ideal) if ideal else 0.0


def catalog_coverage(all_recommendations: list[list[str]], catalog: set[str]) -> float:
    recommended = set().union(*(set(items) for items in all_recommendations)) if all_recommendations else set()
    return len(recommended & catalog) / max(len(catalog), 1)


def train_test_split_implicit(frame: pd.DataFrame, random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    held_out_indices = []
    for _, group in frame.groupby("user_id"):
        if group.artist_id.nunique() >= 3:
            held_out_indices.append(rng.choice(group.index.to_numpy()))
    test = frame.loc[held_out_indices].copy()
    train = frame.drop(index=held_out_indices).copy()
    return train, test


def evaluate_recommender(frame: pd.DataFrame, k: int = 10, sample_users: int = 100) -> dict[str, float]:
    train, test = train_test_split_implicit(frame)
    model = HybridMusicRecommender().fit(train)
    eligible = test.user_id.unique()[:sample_users]
    metrics = {"precision": [], "recall": [], "map": [], "ndcg": []}
    recommendation_lists: list[list[str]] = []
    for user_id in eligible:
        history = train[train.user_id == user_id].sort_values("play_count", ascending=False)
        if history.empty:
            continue
        seed = history.iloc[0].artist_name
        relevant = set(test.loc[test.user_id == user_id, "artist_id"].astype(str))
        try:
            recommendations = model.recommend_artists(seed, k)
        except ValueError:
            continue
        names = recommendations.artist_name.tolist()
        ids = [str(model.metadata.loc[model.metadata.artist_name == name, "artist_id"].iloc[0]) for name in names]
        recommendation_lists.append(ids)
        metrics["precision"].append(precision_at_k(ids, relevant, k))
        metrics["recall"].append(recall_at_k(ids, relevant, k))
        metrics["map"].append(average_precision_at_k(ids, relevant, k))
        metrics["ndcg"].append(ndcg_at_k(ids, relevant, k))
    catalog = set(frame.artist_id.astype(str))
    return {
        "Precision@K": float(np.mean(metrics["precision"])) if metrics["precision"] else 0.0,
        "Recall@K": float(np.mean(metrics["recall"])) if metrics["recall"] else 0.0,
        "MAP@K": float(np.mean(metrics["map"])) if metrics["map"] else 0.0,
        "NDCG@K": float(np.mean(metrics["ndcg"])) if metrics["ndcg"] else 0.0,
        "Coverage": catalog_coverage(recommendation_lists, catalog),
        "Evaluated users": float(len(recommendation_lists)),
    }


def save_evaluation_report(metrics: dict[str, float], path: str | Path = EVALUATION_REPORT_PATH, k: int = 10) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"| {name} | {value:.4f} |" if name != "Evaluated users" else f"| {name} | {int(value)} |" for name, value in metrics.items())
    target.write_text(
        f"# Evaluation Results\n\nHold-one-out evaluation using a recommendation cutoff of K={k}. "
        "Each user's strongest remaining artist is used as the discovery seed.\n\n"
        f"| Metric | Result |\n|---|---:|\n{rows}\n\n"
        "Results are measured on the active dataset and may change when a real Last.fm dataset is added.\n",
        encoding="utf-8",
    )


def main() -> None:
    frame, _ = load_interactions()
    metrics = evaluate_recommender(frame)
    save_evaluation_report(metrics)
    print(metrics)


if __name__ == "__main__":
    main()

