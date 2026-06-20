from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from config import MAX_ARTISTS, MIN_ARTIST_PLAY_COUNT, MIN_USER_INTERACTIONS, RANDOM_STATE, SAMPLE_USERS


@dataclass
class ProcessedData:
    interactions: pd.DataFrame
    user_artist_matrix: csr_matrix
    artist_user_matrix: csr_matrix
    metadata: pd.DataFrame
    user_encoder: LabelEncoder
    artist_encoder: LabelEncoder


def preprocess_interactions(
    frame: pd.DataFrame,
    min_user_interactions: int = MIN_USER_INTERACTIONS,
    min_artist_play_count: int = MIN_ARTIST_PLAY_COUNT,
) -> ProcessedData:
    data = frame[["user_id", "artist_id", "artist_name", "play_count"]].copy()
    data = data.dropna(subset=["user_id", "artist_id", "artist_name", "play_count"])
    data["play_count"] = pd.to_numeric(data["play_count"], errors="coerce")
    data = data.dropna(subset=["play_count"])
    data = data[data.play_count > 0]
    data = data.groupby(["user_id", "artist_id", "artist_name"], as_index=False).play_count.sum()

    eligible_users = data.groupby("user_id").artist_id.nunique()
    data = data[data.user_id.isin(eligible_users[eligible_users >= min_user_interactions].index)]
    artist_plays = data.groupby("artist_id").play_count.sum()
    data = data[data.artist_id.isin(artist_plays[artist_plays >= min_artist_play_count].index)]

    if data.user_id.nunique() > SAMPLE_USERS:
        users = pd.Series(data.user_id.unique()).sample(SAMPLE_USERS, random_state=RANDOM_STATE)
        data = data[data.user_id.isin(users)]
    if data.artist_id.nunique() > MAX_ARTISTS:
        keep = data.groupby("artist_id").play_count.sum().nlargest(MAX_ARTISTS).index
        data = data[data.artist_id.isin(keep)]
    if data.empty:
        raise ValueError("No interactions remain after preprocessing.")

    user_encoder, artist_encoder = LabelEncoder(), LabelEncoder()
    data["user_idx"] = user_encoder.fit_transform(data.user_id)
    data["artist_idx"] = artist_encoder.fit_transform(data.artist_id)
    data["strength"] = np.log1p(data.play_count)
    matrix = csr_matrix(
        (data.strength, (data.user_idx, data.artist_idx)),
        shape=(len(user_encoder.classes_), len(artist_encoder.classes_)),
    )
    metadata = data.groupby(["artist_idx", "artist_id", "artist_name"], as_index=False).agg(
        total_play_count=("play_count", "sum"), unique_listeners=("user_id", "nunique")
    ).sort_values("artist_idx")
    return ProcessedData(data, matrix, matrix.T.tocsr(), metadata, user_encoder, artist_encoder)
