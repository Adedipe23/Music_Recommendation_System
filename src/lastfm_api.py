from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

from config import LASTFM_API_BASE_URL
from src.utils import image_from_lastfm

load_dotenv()


class LastFMError(RuntimeError):
    pass


class LastFMClient:
    def __init__(self, api_key: str | None = None, timeout: int = 10) -> None:
        self.api_key = api_key or os.getenv("LASTFM_API_KEY")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _request(self, method: str, **params: Any) -> dict[str, Any]:
        if not self.api_key:
            raise LastFMError("A Last.fm API key is required.")
        payload = {"method": method, "api_key": self.api_key, "format": "json", **params}
        try:
            response = requests.get(LASTFM_API_BASE_URL, params=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise LastFMError(f"Last.fm could not be reached: {exc}") from exc
        if "error" in data:
            raise LastFMError(data.get("message", "Last.fm returned an error."))
        return data

    def get_top_tracks(self, artist: str, limit: int = 30) -> list[dict[str, Any]]:
        tracks = self._request("artist.getTopTracks", artist=artist, limit=limit).get("toptracks", {}).get("track", [])
        return [self._normalize_track(track) for track in tracks]

    def get_similar_tracks(self, artist: str, track: str, limit: int = 20) -> list[dict[str, Any]]:
        tracks = self._request("track.getSimilar", artist=artist, track=track, limit=limit, autocorrect=1).get("similartracks", {}).get("track", [])
        return [self._normalize_track(item) for item in tracks]

    def get_artist_info(self, artist: str) -> dict[str, Any]:
        item = self._request("artist.getInfo", artist=artist, autocorrect=1).get("artist", {})
        return {"name": item.get("name", artist), "url": item.get("url"), "image": image_from_lastfm(item.get("image"))}

    @staticmethod
    def _normalize_track(track: dict[str, Any]) -> dict[str, Any]:
        artist = track.get("artist", {})
        artist_name = artist.get("name", artist.get("#text", "")) if isinstance(artist, dict) else str(artist)
        return {
            "name": track.get("name", "Unknown track"),
            "artist": artist_name,
            "match": float(track.get("match", 0) or 0),
            "listeners": int(float(track.get("listeners", 0) or 0)),
            "playcount": int(float(track.get("playcount", 0) or 0)),
            "url": track.get("url"),
            "image": image_from_lastfm(track.get("image")),
        }

