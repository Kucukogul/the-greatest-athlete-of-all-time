from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_athlete_df() -> pd.DataFrame:
    return pd.DataFrame({
        "athlete_id": ["a1", "a2", "a3", "a4", "a5"],
        "name": ["Michael Jordan", "LeBron James", "Kobe Bryant", "Magic Johnson", "Larry Bird"],
        "sport": ["basketball"] * 5,
        "season": [1996, 2012, 2006, 1987, 1986],
        "ppg": [30.1, 27.1, 35.4, 19.3, 28.7],
        "rpg": [6.6, 7.9, 5.3, 6.3, 9.8],
        "apg": [5.3, 7.2, 4.7, 11.2, 6.8],
        "ppg_normalized": [95.0, 80.0, 100.0, 55.0, 88.0],
        "rpg_normalized": [60.0, 72.0, 45.0, 58.0, 95.0],
        "apg_normalized": [48.0, 65.0, 43.0, 100.0, 62.0],
        "per_normalized": [90.0, 85.0, 82.0, 75.0, 78.0],
    })


# ── Tennis fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def tennis_config_path() -> Path:
    return Path(__file__).parent.parent / "configs" / "scoring_tennis.yaml"


@pytest.fixture
def big3_raw_df() -> pd.DataFrame:
    """Career stats for Federer, Nadal, Djokovic. Source: tennis_big3_raw.json."""
    fixture_path = Path(__file__).parent / "fixtures" / "tennis_big3_raw.json"
    return pd.read_json(fixture_path)


@pytest.fixture
def minimal_tennis_df() -> pd.DataFrame:
    """Single-player DataFrame for isolation tests — uses Djokovic as baseline."""
    return pd.DataFrame([{
        "athlete_id": "djk_001", "name": "Novak Djokovic", "sport": "tennis",
        "era": "Big 3 Era",
        "grand_slams": 24, "weeks_at_no1": 428, "masters_titles": 40,
        "career_wins": 1105, "career_matches": 1293,
        "yearend_no1_count": 8,
        "finals_won": 99, "finals_played": 128,
        "h2h_top10_wins": 340, "h2h_top10_total": 430,
        "hard_win_pct": 0.860, "clay_win_pct": 0.830, "grass_win_pct": 0.840,
        "years_in_top5": 20,
    }])
