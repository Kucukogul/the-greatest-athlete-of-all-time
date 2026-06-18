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


# ── NBA fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def nba_data_dir() -> Path:
    return Path(__file__).parent.parent / "data" / "raw"


@pytest.fixture(scope="session")
def nba_config_dir() -> Path:
    return Path(__file__).parent.parent / "configs"


@pytest.fixture(scope="session")
def nba_result(nba_data_dir, nba_config_dir):
    """Full NBA GOAT ranking — computed once per test session (scope=session)."""
    from src.pipelines.nba_runner import NBARunner
    if not (nba_data_dir / "nba_all_seasons_raw.csv").exists():
        pytest.skip("nba_all_seasons_raw.csv not present — skipping NBA integration tests")
    runner = NBARunner(nba_data_dir, nba_config_dir)
    return runner.run()


@pytest.fixture(scope="session")
def nba_snapshot() -> dict:
    """Expected GOAT scores loaded from versioned fixture file."""
    import json
    path = Path(__file__).parent / "fixtures" / "nba_goat_snapshot_v1.json"
    with open(path) as f:
        return json.load(f)


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


# ── Swimming fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def swimming_config_path() -> Path:
    return Path(__file__).parent.parent / "configs" / "scoring_swimming.yaml"


@pytest.fixture
def swimming_pipeline_df() -> pd.DataFrame:
    """Three-athlete pipeline output (career_years + olympic_gold_total already derived).

    One athlete per era (Modern / Amateur / Pre-Modern) to exercise era-adjustment logic.
    Pre-Modern athlete has wc_gold_individual = 0 — structural absence, not failure.
    """
    return pd.DataFrame([
        {
            "athlete_id": "sw_t01", "name": "Alpha Modern", "nationality": "USA",
            "era": "Modern", "birth_year": 1985, "career_start": 2000, "career_end": 2016,
            "career_years": 17, "olympic_games_count": 5,
            "olympic_gold_individual": 13, "olympic_gold_relay": 10, "olympic_gold_total": 23,
            "olympic_silver_individual": 1, "olympic_silver_relay": 2,
            "olympic_bronze_individual": 2, "olympic_bronze_relay": 0,
            "wc_gold_individual": 17, "wc_gold_relay": 9,
            "wc_silver_individual": 5, "wc_bronze_individual": 3,
            "world_records": 39, "events_dominated": 5,
        },
        {
            "athlete_id": "sw_t02", "name": "Beta Amateur", "nationality": "AUS",
            "era": "Amateur", "birth_year": 1960, "career_start": 1978, "career_end": 1992,
            "career_years": 15, "olympic_games_count": 3,
            "olympic_gold_individual": 3, "olympic_gold_relay": 1, "olympic_gold_total": 4,
            "olympic_silver_individual": 0, "olympic_silver_relay": 0,
            "olympic_bronze_individual": 0, "olympic_bronze_relay": 0,
            "wc_gold_individual": 4, "wc_gold_relay": 0,
            "wc_silver_individual": 2, "wc_bronze_individual": 1,
            "world_records": 6, "events_dominated": 2,
        },
        {
            "athlete_id": "sw_t03", "name": "Gamma PreModern", "nationality": "USA",
            "era": "Pre-Modern", "birth_year": 1950, "career_start": 1968, "career_end": 1972,
            "career_years": 5, "olympic_games_count": 2,
            "olympic_gold_individual": 4, "olympic_gold_relay": 5, "olympic_gold_total": 9,
            "olympic_silver_individual": 1, "olympic_silver_relay": 0,
            "olympic_bronze_individual": 1, "olympic_bronze_relay": 0,
            "wc_gold_individual": 0, "wc_gold_relay": 0,
            "wc_silver_individual": 0, "wc_bronze_individual": 0,
            "world_records": 9, "events_dominated": 4,
        },
    ])
