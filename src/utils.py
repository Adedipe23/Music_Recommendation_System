from __future__ import annotations

import html
import re
from urllib.parse import quote_plus


def youtube_artist_url(artist_name: str) -> str:
    query = quote_plus(f"{artist_name} official music video")
    return f"https://www.youtube.com/results?search_query={query}"


def youtube_track_url(artist_name: str, track_name: str) -> str:
    query = quote_plus(f"{artist_name} {track_name} official music video")
    return f"https://www.youtube.com/results?search_query={query}"


def lastfm_artist_url(artist_name: str) -> str:
    return f"https://www.last.fm/music/{quote_plus(artist_name)}"


def lastfm_track_url(artist_name: str, track_name: str) -> str:
    return f"https://www.last.fm/music/{quote_plus(artist_name)}/_/{quote_plus(track_name)}"


def safe_html(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def compact_number(value: object) -> str:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return "—"
    for suffix, divisor in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(number) >= divisor:
            return f"{number / divisor:.1f}{suffix}".replace(".0", "")
    return f"{number:,.0f}"


def image_from_lastfm(images: object) -> str | None:
    if not isinstance(images, list):
        return None
    for preferred in ("extralarge", "large", "medium", "small"):
        for image in images:
            if isinstance(image, dict) and image.get("size") == preferred:
                url = image.get("#text")
                if url and re.match(r"^https?://", str(url)):
                    return str(url)
    return None
