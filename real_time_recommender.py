import os
from typing import List, Dict

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Set your key in environment: YOUTUBE_API_KEY
API_KEY = os.getenv("YOUTUBE_API_KEY", "YOUR_YOUTUBE_API_KEY")


def youtube_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Return YouTube video recommendations for a query.

    Returns an empty list if configuration is missing or the API call fails.
    """
    if not query or not str(query).strip():
        return []

    if not API_KEY or API_KEY == "YOUR_YOUTUBE_API_KEY":
        return []

    try:
        youtube = build("youtube", "v3", developerKey=API_KEY)
        request = youtube.search().list(
            part="snippet",
            q=str(query).strip(),
            type="video",
            maxResults=max_results,
        )
        response = request.execute()
    except HttpError:
        return []
    except Exception:
        return []

    results: List[Dict[str, str]] = []

    for item in response.get("items", []):
        snippet = item.get("snippet", {})
        video_id = item.get("id", {}).get("videoId")
        title = snippet.get("title")

        if not title or not video_id:
            continue

        results.append({
            "title": title,
            "link": f"https://www.youtube.com/watch?v={video_id}",
        })

    return results
