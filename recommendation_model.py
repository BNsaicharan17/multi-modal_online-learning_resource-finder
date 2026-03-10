"""
Optimized recommendation model with fuzzy matching, source filtering, and caching.
Module 3: Learning Recommendation Model
"""

import pandas as pd
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_resources() -> pd.DataFrame:
    """Load resources CSV once and cache it."""
    return pd.read_csv("data/resources.csv")


def get_all_sources() -> list:
    """Return all unique source platforms."""
    df = _load_resources()
    return sorted(df["source"].dropna().unique().tolist())


def get_all_topics() -> list:
    """Return all unique topics."""
    df = _load_resources()
    return sorted(df["topic"].dropna().unique().tolist())


def recommend(
    topic: str,
    level: str | None = None,
    sources: list | None = None,
) -> pd.DataFrame:
    """Return resources matching the given topic.

    Uses case-insensitive substring matching so partial matches
    (e.g., "machine" → "Machine Learning") still surface results.

    Args:
        topic: The predicted or user-supplied topic string.
        level: Optional difficulty filter (Beginner / Intermediate / Advanced).
        sources: Optional list of source platforms to filter by.

    Returns:
        DataFrame of matching resources, sorted by level.
    """
    df = _load_resources()

    if not topic or not topic.strip():
        return df.head(5)

    topic_lower = topic.strip().lower()

    # Fuzzy match: case-insensitive substring search on topic column
    mask = df["topic"].str.lower().str.contains(topic_lower, na=False)

    # If no direct match, try matching individual words
    if not mask.any():
        words = topic_lower.split()
        for word in words:
            if len(word) > 2:
                mask = mask | df["topic"].str.lower().str.contains(word, na=False)

    results = df[mask]

    # Filter by source platform
    if sources and len(sources) > 0:
        results = results[results["source"].isin(sources)]

    # Filter by level
    if level and level != "All":
        level_filtered = results[results["level"].str.lower() == level.strip().lower()]
        if not level_filtered.empty:
            results = level_filtered

    # Sort by level: Beginner → Intermediate → Advanced
    level_order = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
    results = results.copy()
    results["_sort"] = results["level"].map(level_order).fillna(3)
    results = results.sort_values("_sort").drop(columns=["_sort"])

    return results