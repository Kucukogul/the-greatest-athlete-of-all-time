from __future__ import annotations

import pandas as pd
import pytest

from src.normalize.tennis_normalizer import TennisNormalizer, _safe_ratio

_EXPECTED_OUTPUT_COLUMNS = [
    "grand_slam_normalized",
    "weeks_no1_normalized",
    "masters_titles_normalized",
    "career_win_rate_normalized",
    "yearend_no1_normalized",
    "finals_win_rate_normalized",
    "h2h_top10_normalized",
    "surface_versatility_normalized",
    "longevity_normalized",
]


# ── Validation ────────────────────────────────────────────────────────────────

class TestTennisNormalizerValidation:
    def test_empty_dataframe_raises_value_error(self, tennis_config_path):
        normalizer = TennisNormalizer(tennis_config_path)
        with pytest.raises(ValueError, match="empty"):
            normalizer.normalize(pd.DataFrame())

    def test_missing_column_raises_value_error(self, tennis_config_path, big3_raw_df):
        normalizer = TennisNormalizer(tennis_config_path)
        incomplete = big3_raw_df.drop(columns=["grand_slams"])
        with pytest.raises(ValueError, match="Missing required raw columns"):
            normalizer.normalize(incomplete)

    def test_duplicate_athlete_id_raises_value_error(self, tennis_config_path, big3_raw_df):
        normalizer = TennisNormalizer(tennis_config_path)
        duplicated = pd.concat([big3_raw_df, big3_raw_df.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="Duplicate athlete_id"):
            normalizer.normalize(duplicated)


# ── Derived metrics ───────────────────────────────────────────────────────────

class TestDerivedMetrics:
    def test_career_win_rate_is_wins_over_matches(self, tennis_config_path, big3_raw_df):
        normalizer = TennisNormalizer(tennis_config_path)
        result = normalizer.normalize(big3_raw_df, era_adjust=False)
        federer = result[result["athlete_id"] == "fed_001"].iloc[0]
        expected = 1251 / 1526
        assert federer["career_win_rate"] == pytest.approx(expected, rel=1e-4)

    def test_finals_win_rate_is_won_over_played(self, tennis_config_path, big3_raw_df):
        normalizer = TennisNormalizer(tennis_config_path)
        result = normalizer.normalize(big3_raw_df, era_adjust=False)
        nadal = result[result["athlete_id"] == "nad_001"].iloc[0]
        expected = 92 / 134
        assert nadal["finals_win_rate"] == pytest.approx(expected, rel=1e-4)

    def test_h2h_top10_is_wins_over_total(self, tennis_config_path, big3_raw_df):
        normalizer = TennisNormalizer(tennis_config_path)
        result = normalizer.normalize(big3_raw_df, era_adjust=False)
        djokovic = result[result["athlete_id"] == "djk_001"].iloc[0]
        expected = djokovic["h2h_top10_wins"] / djokovic["h2h_top10_total"]
        assert djokovic["h2h_top10"] == pytest.approx(expected, rel=1e-4)

    def test_safe_ratio_returns_zero_when_denominator_is_zero(self):
        num = pd.Series([10.0, 20.0])
        den = pd.Series([0.0, 4.0])
        result = _safe_ratio(num, den)
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(5.0)


# ── Surface versatility ───────────────────────────────────────────────────────

class TestSurfaceVersatility:
    def test_equal_surface_rates_return_midpoint_in_single_player_dataset(
        self, tennis_config_path
    ):
        # Single-player edge case: surface_versatility intermediate score = 100.0
        # (std=0 → inverse_std logic returns 100.0).
        # normalize_series(minmax) on a constant series returns 50.0 by design —
        # same contract verified in TestNormalizer.test_constant_series_returns_50.
        normalizer = TennisNormalizer(tennis_config_path)
        df = pd.DataFrame([
            {
                "athlete_id": "p1", "name": "Perfect", "sport": "tennis",
                "era": "Big 3 Era",
                "grand_slams": 10, "weeks_at_no1": 100, "masters_titles": 10,
                "career_wins": 500, "career_matches": 600,
                "yearend_no1_count": 3, "finals_won": 40, "finals_played": 50,
                "h2h_top10_wins": 30, "h2h_top10_total": 50,
                "hard_win_pct": 0.80, "clay_win_pct": 0.80, "grass_win_pct": 0.80,
                "years_in_top5": 10,
            }
        ])
        result = normalizer.normalize(df, era_adjust=False)
        assert result["surface_versatility_normalized"].iloc[0] == pytest.approx(50.0)

    def test_clay_specialist_has_lower_versatility_than_allrounder(
        self, tennis_config_path, big3_raw_df
    ):
        normalizer = TennisNormalizer(tennis_config_path)
        result = normalizer.normalize(big3_raw_df, era_adjust=False)
        nadal_v = result.loc[result["athlete_id"] == "nad_001", "surface_versatility_normalized"].iloc[0]
        djokovic_v = result.loc[result["athlete_id"] == "djk_001", "surface_versatility_normalized"].iloc[0]
        assert djokovic_v > nadal_v


# ── Era adjustment ────────────────────────────────────────────────────────────

class TestEraAdjustment:
    def test_era_adjust_false_preserves_raw_grand_slams(
        self, tennis_config_path, big3_raw_df
    ):
        normalizer = TennisNormalizer(tennis_config_path)
        result = normalizer.normalize(big3_raw_df, era_adjust=False)
        federer = result[result["athlete_id"] == "fed_001"].iloc[0]
        assert federer["grand_slams"] == pytest.approx(20.0)

    def test_big3_era_adjustment_factor_is_one(self, tennis_config_path, big3_raw_df):
        # All Big 3 are in the reference era — their raw counts must be unchanged.
        normalizer = TennisNormalizer(tennis_config_path)
        result_adjusted = normalizer.normalize(big3_raw_df.copy(), era_adjust=True)
        result_raw = normalizer.normalize(big3_raw_df.copy(), era_adjust=False)
        for athlete_id in ["fed_001", "nad_001", "djk_001"]:
            adj = result_adjusted.loc[result_adjusted["athlete_id"] == athlete_id, "grand_slams"].iloc[0]
            raw = result_raw.loc[result_raw["athlete_id"] == athlete_id, "grand_slams"].iloc[0]
            assert adj == pytest.approx(raw, rel=1e-6)

    def test_open_era_player_grand_slams_are_boosted(self, tennis_config_path, big3_raw_df):
        # Inject a synthetic Open Era player and confirm their GS count is scaled up.
        normalizer = TennisNormalizer(tennis_config_path)
        laver_row = big3_raw_df.iloc[0].copy()
        laver_row["athlete_id"] = "laver_001"
        laver_row["name"] = "Rod Laver"
        laver_row["era"] = "Open Era"
        laver_row["grand_slams"] = 11
        df_with_laver = pd.concat(
            [big3_raw_df, pd.DataFrame([laver_row])], ignore_index=True
        )
        result = normalizer.normalize(df_with_laver, era_adjust=True)
        laver_raw_gs = 11.0
        laver_adjusted_gs = result.loc[result["athlete_id"] == "laver_001", "grand_slams"].iloc[0]
        assert laver_adjusted_gs > laver_raw_gs


# ── Output columns ────────────────────────────────────────────────────────────

class TestOutputColumns:
    def test_all_expected_columns_are_present(self, tennis_config_path, big3_raw_df):
        normalizer = TennisNormalizer(tennis_config_path)
        result = normalizer.normalize(big3_raw_df)
        for col in _EXPECTED_OUTPUT_COLUMNS:
            assert col in result.columns, f"Missing output column: {col}"

    def test_all_normalized_values_are_in_zero_to_100_range(
        self, tennis_config_path, big3_raw_df
    ):
        normalizer = TennisNormalizer(tennis_config_path)
        result = normalizer.normalize(big3_raw_df)
        for col in _EXPECTED_OUTPUT_COLUMNS:
            assert result[col].between(0.0, 100.0).all(), (
                f"{col} has values outside [0, 100]: {result[col].tolist()}"
            )

    def test_raw_columns_are_preserved_in_output(self, tennis_config_path, big3_raw_df):
        normalizer = TennisNormalizer(tennis_config_path)
        result = normalizer.normalize(big3_raw_df)
        assert "grand_slams" in result.columns
        assert "name" in result.columns

    def test_output_columns_method_matches_metric_map(self, tennis_config_path):
        normalizer = TennisNormalizer(tennis_config_path)
        assert set(normalizer.output_columns()) == set(_EXPECTED_OUTPUT_COLUMNS)
