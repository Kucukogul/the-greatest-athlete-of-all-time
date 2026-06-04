import pandas as pd
import pytest

from src.features.engineering import FeatureEngineer, TennisSurfaceFeatures


def make_athlete_df() -> pd.DataFrame:
    return pd.DataFrame({
        "athlete_id": ["a1", "a2", "a3"],
        "ppg": [30.0, 20.0, 10.0],
        "rpg": [12.0, 8.0, 4.0],
    })


def make_tennis_df() -> pd.DataFrame:
    return pd.DataFrame({
        "name": ["Djokovic", "Nadal", "Sampras"],
        "era": ["Big 3 Era", "Big 3 Era", "Modern Era"],
        "hard_win_pct": [0.84, 0.77, 0.80],
        "clay_win_pct": [0.80, 0.90, 0.63],
        "grass_win_pct": [0.86, 0.78, 0.84],
        "surface_versatility_normalized": [92.9, 81.9, 71.3],
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


class TestTennisSurfaceFeatures:
    def test_surface_flexibility_is_product(self):
        df = make_tennis_df()
        result = TennisSurfaceFeatures.add_surface_flexibility(df)
        expected = df["clay_win_pct"] * df["grass_win_pct"]
        pd.testing.assert_series_equal(
            result["surface_flexibility"], expected, check_names=False
        )

    def test_surface_gap_is_absolute_difference(self):
        df = make_tennis_df()
        result = TennisSurfaceFeatures.add_surface_gap(df)
        expected = (df["clay_win_pct"] - df["grass_win_pct"]).abs()
        pd.testing.assert_series_equal(
            result["surface_gap"], expected, check_names=False
        )

    def test_surface_floor_is_minimum_across_surfaces(self):
        df = make_tennis_df()
        result = TennisSurfaceFeatures.add_surface_floor(df)
        expected = df[["hard_win_pct", "clay_win_pct", "grass_win_pct"]].min(axis=1)
        pd.testing.assert_series_equal(
            result["surface_floor"], expected, check_names=False
        )

    def test_era_encoding_big3(self):
        df = make_tennis_df()
        result = TennisSurfaceFeatures.add_era_encoding(df)
        assert result.loc[0, "era_encoded"] == 2

    def test_era_encoding_modern(self):
        df = make_tennis_df()
        result = TennisSurfaceFeatures.add_era_encoding(df)
        assert result.loc[2, "era_encoded"] == 1

    def test_era_encoding_open(self):
        df = pd.DataFrame({"era": ["Open Era"]})
        result = TennisSurfaceFeatures.add_era_encoding(df)
        assert result.loc[0, "era_encoded"] == 0

    def test_era_interactions_raises_without_era_encoded(self):
        df = make_tennis_df()
        with pytest.raises(ValueError, match="era_encoded"):
            TennisSurfaceFeatures.add_era_interactions(df)

    def test_era_interactions_clay_sampras(self):
        # Sampras: Modern Era (1), clay=0.63 → clay_era_interaction=0.63
        df = TennisSurfaceFeatures.add_era_encoding(make_tennis_df())
        result = TennisSurfaceFeatures.add_era_interactions(df)
        assert result.loc[2, "clay_era_interaction"] == pytest.approx(0.63)

    def test_era_interactions_clay_djokovic(self):
        # Djokovic: Big 3 Era (2), clay=0.80 → clay_era_interaction=1.60
        df = TennisSurfaceFeatures.add_era_encoding(make_tennis_df())
        result = TennisSurfaceFeatures.add_era_interactions(df)
        assert result.loc[0, "clay_era_interaction"] == pytest.approx(1.60)

    def test_era_interactions_open_era_is_zero(self):
        # Open Era (0): all era interaction terms must be 0
        df = pd.DataFrame({
            "era": ["Open Era"],
            "hard_win_pct": [0.70],
            "clay_win_pct": [0.68],
            "grass_win_pct": [0.72],
            "surface_versatility_normalized": [85.0],
        })
        df = TennisSurfaceFeatures.add_era_encoding(df)
        result = TennisSurfaceFeatures.add_era_interactions(df)
        assert result.loc[0, "hard_era_interaction"] == pytest.approx(0.0)
        assert result.loc[0, "clay_era_interaction"] == pytest.approx(0.0)
        assert result.loc[0, "grass_era_interaction"] == pytest.approx(0.0)
        assert result.loc[0, "versatility_era_interaction"] == pytest.approx(0.0)

    def test_build_all_adds_expected_columns(self):
        df = make_tennis_df()
        result = TennisSurfaceFeatures.build_all(df)
        expected = {
            "surface_flexibility",
            "surface_gap",
            "surface_floor",
            "era_encoded",
            "hard_era_interaction",
            "clay_era_interaction",
            "grass_era_interaction",
            "versatility_era_interaction",
        }
        assert expected.issubset(result.columns)

    def test_build_all_does_not_mutate_input(self):
        df = make_tennis_df()
        original_clay = df["clay_win_pct"].copy()
        TennisSurfaceFeatures.build_all(df)
        pd.testing.assert_series_equal(df["clay_win_pct"], original_clay)
