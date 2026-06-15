from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.pipelines.nba_pipeline import NBAPipeline


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _row(
    player_id: str = "p1",
    name: str = "Alice",
    year: int = 2000,
    tm: str = "LAL",
    g: float = 70.0,
    mp: float = 1500.0,
    bpm: float | None = 5.0,
    vorp: float | None = 3.0,
    ws: float = 8.0,
    ws48: float | None = 0.180,
    ts_pct: float | None = 0.55,
    fta: float = 200.0,
    fga: float = 400.0,
) -> dict:
    return {
        "Year": year,
        "Player": name,
        "player_id": player_id,
        "Tm": tm,
        "G": g,
        "MP": mp,
        "BPM": bpm,
        "VORP": vorp,
        "WS": ws,
        "WS/48": ws48,
        "TS%": ts_pct,
        "FTA": fta,
        "FGA": fga,
    }


def _pipe() -> NBAPipeline:
    """Pipeline instance that never touches the filesystem (internal-method tests only)."""
    return NBAPipeline(data_dir=Path("."), config_dir=Path("."))


def _seasons(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ── TestDedup ─────────────────────────────────────────────────────────────────

class TestDedup:
    def test_removes_per_team_rows_when_aggregate_exists(self):
        seasons = _seasons(
            _row("p1", "Alice", 2000, "2TM", g=80, mp=2000),
            _row("p1", "Alice", 2000, "LAL", g=50, mp=1200),
            _row("p1", "Alice", 2000, "NYK", g=30, mp=800),
            _row("p2", "Bob",   2000, "LAL", g=75, mp=1800),
        )
        result = _pipe()._dedup_multi_team(seasons)
        assert len(result) == 2
        assert result[result["player_id"] == "p1"].iloc[0]["Tm"] == "2TM"
        assert result[result["player_id"] == "p2"].iloc[0]["Tm"] == "LAL"

    def test_single_team_players_unaffected(self):
        seasons = _seasons(
            _row("p1", "Alice", 2000, "LAL"),
            _row("p1", "Alice", 2001, "LAL"),
            _row("p1", "Alice", 2002, "LAL"),
        )
        result = _pipe()._dedup_multi_team(seasons)
        assert len(result) == 3

    @pytest.mark.parametrize("label", ["TOT", "2TM", "3TM"])
    def test_all_aggregate_labels_handled(self, label: str):
        seasons = _seasons(
            _row("p1", "Alice", 2000, label, g=80),
            _row("p1", "Alice", 2000, "LAL",  g=50),
            _row("p1", "Alice", 2000, "NYK",  g=30),
        )
        result = _pipe()._dedup_multi_team(seasons)
        assert len(result) == 1
        assert result.iloc[0]["Tm"] == label

    def test_player_in_different_years_handled_independently(self):
        """Year 2000: changed teams (has 2TM). Year 2001: stayed (no 2TM)."""
        seasons = _seasons(
            _row("p1", "Alice", 2000, "2TM", g=80),
            _row("p1", "Alice", 2000, "LAL", g=50),
            _row("p1", "Alice", 2001, "BOS", g=75),  # single team — keep
        )
        result = _pipe()._dedup_multi_team(seasons)
        assert len(result) == 2
        tms = set(result["Tm"])
        assert tms == {"2TM", "BOS"}


# ── TestNameCollisionDetection ────────────────────────────────────────────────

class TestNameCollisionDetection:
    def test_logs_warning_for_multiple_names(self, caplog):
        seasons = _seasons(
            _row("p1", "Vit Krejci",  2022, "OKC"),
            _row("p1", "Vít Krejčí", 2023, "OKC"),
        )
        with caplog.at_level(logging.WARNING, logger="src.pipelines.nba_pipeline"):
            _pipe()._detect_name_collisions(seasons)
        assert any("p1" in m for m in caplog.messages)

    def test_silent_when_names_consistent(self, caplog):
        seasons = _seasons(
            _row("p1", "Alice", 2000, "LAL"),
            _row("p1", "Alice", 2001, "LAL"),
        )
        with caplog.at_level(logging.WARNING, logger="src.pipelines.nba_pipeline"):
            _pipe()._detect_name_collisions(seasons)
        assert not any("collision" in m.lower() for m in caplog.messages)


# ── TestAggregateCareerRecord ─────────────────────────────────────────────────

class TestAggregateCareerRecord:
    def test_one_row_per_player_id(self):
        seasons = _seasons(
            _row("p1", "Alice", 2000),
            _row("p1", "Alice", 2001),
            _row("p2", "Bob",   2000),
        )
        result = _pipe()._aggregate_careers(seasons)
        assert len(result) == 2
        assert set(result["player_id"]) == {"p1", "p2"}

    def test_total_games_summed(self):
        seasons = _seasons(
            _row("p1", year=2000, g=70),
            _row("p1", year=2001, g=82),
            _row("p1", year=2002, g=60),
        )
        career = _pipe()._aggregate_careers(seasons)
        assert career.iloc[0]["total_games"] == 212

    def test_career_vorp_none_when_all_nan(self):
        """Pre-advanced era players have no VORP data — should return None, not 0."""
        seasons = _seasons(
            _row("p1", year=1962, vorp=None),
            _row("p1", year=1963, vorp=None),
        )
        career = _pipe()._aggregate_careers(seasons)
        assert career.iloc[0]["career_vorp"] is None

    def test_career_vorp_sums_correctly(self):
        seasons = _seasons(
            _row("p1", year=2000, vorp=5.0),
            _row("p1", year=2001, vorp=3.0),
            _row("p1", year=2002, vorp=4.5),
        )
        career = _pipe()._aggregate_careers(seasons)
        assert career.iloc[0]["career_vorp"] == pytest.approx(12.5)

    def test_career_bpm_is_mp_weighted(self):
        """High-MP season should dominate the BPM average."""
        seasons = _seasons(
            _row("p1", year=2000, bpm=10.0, mp=3000.0),
            _row("p1", year=2001, bpm=2.0,  mp=100.0),
        )
        career = _pipe()._aggregate_careers(seasons)
        expected = np.average([10.0, 2.0], weights=[3000.0, 100.0])
        assert career.iloc[0]["career_bpm"] == pytest.approx(expected, abs=1e-3)

    def test_career_bpm_none_when_metric_all_nan(self):
        seasons = _seasons(
            _row("p1", year=1960, bpm=None, mp=1500),
            _row("p1", year=1961, bpm=None, mp=1800),
        )
        career = _pipe()._aggregate_careers(seasons)
        assert career.iloc[0]["career_bpm"] is None

    def test_qualifying_seasons_counts_g_ge_40(self):
        seasons = _seasons(
            _row("p1", year=2000, g=50),   # qualifies
            _row("p1", year=2001, g=39),   # does NOT qualify
            _row("p1", year=2002, g=40),   # qualifies
            _row("p1", year=2003, g=82),   # qualifies
        )
        career = _pipe()._aggregate_careers(seasons)
        assert career.iloc[0]["qualifying_seasons"] == 3

    def test_years_active_computed_correctly(self):
        seasons = _seasons(
            _row("p1", year=1985, g=70),
            _row("p1", year=1990, g=70),
            _row("p1", year=1998, g=70),
        )
        career = _pipe()._aggregate_careers(seasons)
        assert career.iloc[0]["years_active"] == 14   # 1998 − 1985 + 1

    def test_name_most_frequent_chosen(self):
        """player_id 'p1' has two name variants — most common should be selected."""
        seasons = _seasons(
            _row("p1", "Vít Krejčí", 2022),
            _row("p1", "Vit Krejci", 2023),
            _row("p1", "Vit Krejci", 2024),  # "Vit Krejci" appears twice
        )
        career = _pipe()._aggregate_careers(seasons)
        assert career.iloc[0]["name"] == "Vit Krejci"


# ── TestApplyCareerThreshold ──────────────────────────────────────────────────

class TestApplyCareerThreshold:
    def _career_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"player_id": "p1", "total_games": 600, "qualifying_seasons": 8},
            {"player_id": "p2", "total_games": 200, "qualifying_seasons": 8},   # below min_games
            {"player_id": "p3", "total_games": 600, "qualifying_seasons": 3},   # below min_seasons
            {"player_id": "p4", "total_games": 100, "qualifying_seasons": 2},   # below both
        ])

    def test_eliminates_below_min_career_games(self):
        result = _pipe()._apply_career_threshold(self._career_df(), min_career_games=400, min_qualifying_seasons=5)
        assert "p2" not in result["player_id"].values

    def test_eliminates_below_min_qualifying_seasons(self):
        result = _pipe()._apply_career_threshold(self._career_df(), min_career_games=400, min_qualifying_seasons=5)
        assert "p3" not in result["player_id"].values

    def test_qualifying_players_survive(self):
        result = _pipe()._apply_career_threshold(self._career_df(), min_career_games=400, min_qualifying_seasons=5)
        assert list(result["player_id"]) == ["p1"]

    def test_seçenek_c_name_collision_eliminated(self):
        """Two different 'George Johnson' players sharing a name — only the one with
        enough career games (real GOAT candidate) should survive."""
        career = pd.DataFrame([
            {"player_id": "johnsge01", "total_games": 900, "qualifying_seasons": 10},
            {"player_id": "johnsge02", "total_games": 85,  "qualifying_seasons": 1},
        ])
        result = _pipe()._apply_career_threshold(career, min_career_games=400, min_qualifying_seasons=5)
        assert list(result["player_id"]) == ["johnsge01"]


# ── TestPeak3Consecutive (static method) ─────────────────────────────────────

class TestPeak3Consecutive:
    def _group(self, years: list[int], values: list[float | None], metric: str = "BPM") -> pd.DataFrame:
        return pd.DataFrame({"Year": years, metric: values})

    def test_best_3window_chosen(self):
        """BPM sequence 8, 12, 11, 10, 6 — best window is 12+11+10 = 11.0 avg (years 2001–03)."""
        g = self._group([2000, 2001, 2002, 2003, 2004], [8.0, 12.0, 11.0, 10.0, 6.0])
        avg, year = NBAPipeline._peak_3_consecutive(g, "BPM")
        assert avg == pytest.approx(11.0)
        assert year == 2001

    def test_returns_none_when_metric_all_nan(self):
        g = self._group([1960, 1961, 1962], [None, None, None])
        avg, year = NBAPipeline._peak_3_consecutive(g, "BPM")
        assert avg is None
        assert year is None

    def test_fallback_to_mean_for_fewer_than_3_seasons(self):
        """Only 2 qualifying seasons — fall back to mean of available."""
        g = self._group([2000, 2001], [6.0, 10.0])
        avg, year = NBAPipeline._peak_3_consecutive(g, "BPM")
        assert avg == pytest.approx(8.0)

    def test_single_season_fallback(self):
        g = self._group([2000], [9.5])
        avg, year = NBAPipeline._peak_3_consecutive(g, "BPM")
        assert avg == pytest.approx(9.5)
        assert year == 2000

    def test_nan_seasons_excluded_from_window(self):
        """NaN seasons are excluded before windowing, so the window finds the real best."""
        g = self._group([2000, 2001, 2002, 2003], [None, 10.0, 11.0, 9.0])
        avg, year = NBAPipeline._peak_3_consecutive(g, "BPM")
        # Only 3 valid: [10, 11, 9] → avg 10.0
        assert avg == pytest.approx(10.0)

    def test_works_with_ws48(self):
        g = pd.DataFrame({"Year": [1995, 1996, 1997], "WS/48": [0.25, 0.31, 0.28]})
        avg, year = NBAPipeline._peak_3_consecutive(g, "WS/48")
        assert avg == pytest.approx(pytest.approx((0.25 + 0.31 + 0.28) / 3))
        assert year == 1995


# ── TestComputePeakMetrics ────────────────────────────────────────────────────

class TestComputePeakMetrics:
    def test_peak_vorp_is_best_single_season(self):
        seasons = _seasons(
            _row("p1", year=2000, g=70, vorp=5.0),
            _row("p1", year=2001, g=70, vorp=9.5),  # best
            _row("p1", year=2002, g=70, vorp=7.0),
        )
        peak = _pipe()._compute_peak_metrics(seasons)
        assert peak.iloc[0]["peak_vorp_season"] == pytest.approx(9.5)

    def test_only_qualifying_seasons_used_for_peak(self):
        """Season with G < 40 must be excluded from peak computation."""
        seasons = _seasons(
            _row("p1", year=2000, g=10, bpm=20.0),   # G < 40 — exclude
            _row("p1", year=2001, g=70, bpm=8.0),    # valid
            _row("p1", year=2002, g=75, bpm=9.0),    # valid
            _row("p1", year=2003, g=80, bpm=7.0),    # valid
        )
        peak = _pipe()._compute_peak_metrics(seasons)
        # If g<40 season included, peak would be 20 — must NOT be
        assert peak.iloc[0]["peak_bpm"] == pytest.approx((8.0 + 9.0 + 7.0) / 3)

    def test_pre_advanced_players_have_none_peak_bpm(self):
        """Wilt/Russell era: no BPM in dataset → peak_bpm must be None."""
        seasons = _seasons(
            _row("p1", year=1962, g=72, bpm=None, ws48=0.29),
            _row("p1", year=1963, g=80, bpm=None, ws48=0.28),
            _row("p1", year=1964, g=78, bpm=None, ws48=0.30),
        )
        peak = _pipe()._compute_peak_metrics(seasons)
        assert peak.iloc[0]["peak_bpm"] is None
        assert peak.iloc[0]["peak_ws48"] is not None


# ── TestJoinAchievements ──────────────────────────────────────────────────────

class TestJoinAchievements:
    _ACHIEVEMENTS = {
        "p1": {
            "championships": 6,
            "mvp_awards": 5,
            "finals_mvp": 6,
            "all_nba_1st_team": 10,
            "all_nba_2nd_team": 1,
            "all_star_selections": 14,
            "dpoy_awards": 1,
            "all_defensive_1st_team": 9,
        }
    }

    def _career(self, player_ids: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"player_id": player_ids})

    def test_known_player_gets_correct_values(self):
        result = _pipe()._join_achievements(self._career(["p1"]), self._ACHIEVEMENTS)
        row = result.iloc[0]
        assert row["championships"] == 6
        assert row["mvp_awards"] == 5
        assert row["all_defensive_1st_team"] == 9

    def test_unknown_player_defaults_to_zero(self):
        result = _pipe()._join_achievements(self._career(["p99"]), self._ACHIEVEMENTS)
        row = result.iloc[0]
        assert row["championships"] == 0
        assert row["mvp_awards"] == 0
        assert row["all_star_selections"] == 0

    def test_all_achievement_columns_present(self):
        result = _pipe()._join_achievements(self._career(["p1", "p99"]), self._ACHIEVEMENTS)
        expected_cols = {
            "championships", "mvp_awards", "finals_mvp", "all_nba_1st_team",
            "all_nba_2nd_team", "all_star_selections", "dpoy_awards", "all_defensive_1st_team",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_achievement_columns_are_int_dtype(self):
        result = _pipe()._join_achievements(self._career(["p1", "p99"]), self._ACHIEVEMENTS)
        for col in ["championships", "mvp_awards", "finals_mvp", "all_nba_1st_team"]:
            assert result[col].dtype in (np.int64, np.int32, int)


# ── TestAssignEra ─────────────────────────────────────────────────────────────

class TestAssignEra:
    def _run_era(self, median_year: int) -> tuple[str, int]:
        """Helper: create a player whose median qualifying year == median_year."""
        player_seasons = _seasons(
            _row("p1", year=median_year, g=50),
        )
        career = pd.DataFrame([{"player_id": "p1"}])
        result = _pipe()._assign_era(career, player_seasons)
        row = result.iloc[0]
        return row["era"], row["era_encoded"]

    def test_pre_advanced_era(self):
        era, encoded = self._run_era(1962)
        assert era == "pre_advanced"
        assert encoded == 0

    def test_pre_3pt_era(self):
        era, encoded = self._run_era(1977)
        assert era == "pre_3pt"
        assert encoded == 1

    def test_modern_era(self):
        era, encoded = self._run_era(1995)
        assert era == "modern"
        assert encoded == 2

    def test_analytics_era(self):
        era, encoded = self._run_era(2018)
        assert era == "analytics"
        assert encoded == 3


# ── TestMpWeightedMean (static) ───────────────────────────────────────────────

class TestMpWeightedMean:
    def test_correct_weighted_value(self):
        df = pd.DataFrame({"MP": [3000.0, 100.0], "BPM": [10.0, 2.0]})
        result = NBAPipeline._mp_weighted_mean(df, "BPM")
        expected = np.average([10.0, 2.0], weights=[3000.0, 100.0])
        assert result == pytest.approx(expected, abs=1e-6)

    def test_returns_none_when_metric_all_nan(self):
        df = pd.DataFrame({"MP": [1500.0, 1800.0], "BPM": [np.nan, np.nan]})
        assert NBAPipeline._mp_weighted_mean(df, "BPM") is None

    def test_nan_rows_excluded_from_weighting(self):
        """Season with NaN BPM must not influence the weighted mean."""
        df = pd.DataFrame({
            "MP":  [1000.0, np.nan, 2000.0],
            "BPM": [5.0,    np.nan, 9.0],
        })
        result = NBAPipeline._mp_weighted_mean(df, "BPM")
        expected = np.average([5.0, 9.0], weights=[1000.0, 2000.0])
        assert result == pytest.approx(expected, abs=1e-6)


# ── TestShotWeightedTs (static) ───────────────────────────────────────────────

class TestShotWeightedTs:
    def test_correct_weighted_value(self):
        """Season with more shots (FTA+FGA) should carry more weight."""
        df = pd.DataFrame({
            "TS%": [0.50, 0.65],
            "FTA": [100.0, 500.0],
            "FGA": [300.0, 700.0],
        })
        # weights: [400, 1200]
        expected = np.average([0.50, 0.65], weights=[400.0, 1200.0])
        result = NBAPipeline._shot_weighted_ts(df)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_returns_none_when_ts_all_nan(self):
        df = pd.DataFrame({
            "TS%": [np.nan, np.nan],
            "FTA": [200.0, 300.0],
            "FGA": [400.0, 500.0],
        })
        assert NBAPipeline._shot_weighted_ts(df) is None

    def test_high_volume_seasons_dominate(self):
        """A 0.70 TS% season with 10× more shots should pull the result much closer to 0.70."""
        df = pd.DataFrame({
            "TS%": [0.40, 0.70],
            "FTA": [10.0,  100.0],
            "FGA": [20.0, 200.0],
        })
        result = NBAPipeline._shot_weighted_ts(df)
        assert result > 0.65


# ── TestPipelineRun (integration) ────────────────────────────────────────────

class TestPipelineRun:
    """End-to-end smoke tests using minimal synthetic CSV + YAML in tmp_path."""

    def _write_csv(self, path: Path) -> None:
        rows = []
        # Player A — 8 good seasons, has achievements
        for yr in range(2000, 2008):
            rows.append(_row("pA", "PlayerA", yr, "LAL", g=75, mp=2000, bpm=8.0, vorp=6.0, ws=12.0, ws48=0.220))
        # Player B — 8 seasons but weak metrics (still qualifies at low threshold)
        for yr in range(1990, 1998):
            rows.append(_row("pB", "PlayerB", yr, "BOS", g=60, mp=1400, bpm=2.0, vorp=1.0, ws=5.0, ws48=0.110))
        pd.DataFrame(rows).to_csv(path, index=False)

    def _write_yaml(self, path: Path) -> None:
        data = {
            "players": {
                "pA": {
                    "championships": 3, "mvp_awards": 2, "finals_mvp": 2,
                    "all_nba_1st_team": 5, "all_nba_2nd_team": 2,
                    "all_star_selections": 8, "dpoy_awards": 0,
                    "all_defensive_1st_team": 2,
                }
            }
        }
        with open(path, "w") as f:
            yaml.dump(data, f)

    @pytest.fixture
    def pipeline(self, tmp_path: Path) -> tuple[NBAPipeline, pd.DataFrame]:
        csv_path = tmp_path / "nba_all_seasons_raw.csv"
        yml_path = tmp_path / "nba_achievements.yaml"
        self._write_csv(csv_path)
        self._write_yaml(yml_path)
        pipe = NBAPipeline(data_dir=tmp_path, config_dir=tmp_path)
        df = pipe.run(min_career_games=200, min_qualifying_seasons=3)
        return pipe, df

    def test_output_has_required_columns(self, pipeline):
        _, df = pipeline
        required = {
            "player_id", "name", "era", "era_encoded",
            "career_vorp", "career_ws", "career_bpm", "career_ws48",
            "peak_bpm", "peak_ws48", "peak_vorp_season",
            "championships", "mvp_awards", "finals_mvp",
            "total_games", "qualifying_seasons", "years_active",
        }
        assert required.issubset(set(df.columns))

    def test_no_duplicate_player_ids(self, pipeline):
        _, df = pipeline
        assert df["player_id"].nunique() == len(df)

    def test_known_player_achievements_joined(self, pipeline):
        _, df = pipeline
        row = df[df["player_id"] == "pA"].iloc[0]
        assert row["championships"] == 3
        assert row["mvp_awards"] == 2

    def test_unknown_player_achievements_default_to_zero(self, pipeline):
        _, df = pipeline
        row = df[df["player_id"] == "pB"].iloc[0]
        assert row["championships"] == 0
        assert row["mvp_awards"] == 0
