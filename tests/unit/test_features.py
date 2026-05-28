import pandas as pd
import pytest

from src.features.engineering import FeatureEngineer


def make_athlete_df() -> pd.DataFrame:
    return pd.DataFrame({
        "athlete_id": ["a1", "a2", "a3"],
        "ppg": [30.0, 20.0, 10.0],
        "rpg": [12.0, 8.0, 4.0],
    })


class TestFeatureEngineer:
    def test_normalize_adds_normalized_columns(self):
        df = make_athlete_df()
        result = FeatureEngineer(["ppg", "rpg"]).normalize(df)
        assert "ppg_normalized" in result.columns
        assert "rpg_normalized" in result.columns

    def test_normalize_min_is_zero(self):
        df = make_athlete_df()
        result = FeatureEngineer(["ppg"]).normalize(df)
        assert result["ppg_normalized"].min() == pytest.approx(0.0)

    def test_normalize_max_is_100(self):
        df = make_athlete_df()
        result = FeatureEngineer(["ppg"]).normalize(df)
        assert result["ppg_normalized"].max() == pytest.approx(100.0)

    def test_original_columns_are_not_mutated(self):
        df = make_athlete_df()
        result = FeatureEngineer(["ppg"]).normalize(df)
        assert list(result["ppg"]) == [30.0, 20.0, 10.0]

    def test_era_adjust_scales_by_era_mean(self):
        df = pd.DataFrame({
            "season": ["1990", "1990", "2010"],
            "ppg": [30.0, 20.0, 25.0],
        })
        era_means = {"1990": 2.0, "2010": 1.0}
        result = FeatureEngineer(["ppg"]).era_adjust(df, "season", era_means)
        assert result["ppg"].iloc[0] == pytest.approx(15.0)
        assert result["ppg"].iloc[2] == pytest.approx(25.0)
