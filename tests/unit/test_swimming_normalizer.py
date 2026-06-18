"""Unit tests for SwimmingNormalizer — uses swimming_pipeline_df and swimming_config_path fixtures (conftest.py)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.normalize.swimming_normalizer import SwimmingNormalizer, _METRIC_MAP

_EXPECTED_OUTPUT_COLUMNS = list(_METRIC_MAP.keys())


# ── TestValidation ────────────────────────────────────────────────────────────

class TestValidation:
    def test_empty_dataframe_raises(self, swimming_config_path):
        norm = SwimmingNormalizer(swimming_config_path)
        with pytest.raises(ValueError, match="empty"):
            norm.normalize(pd.DataFrame())

    def test_missing_column_raises(self, swimming_config_path, swimming_pipeline_df):
        norm = SwimmingNormalizer(swimming_config_path)
        incomplete = swimming_pipeline_df.drop(columns=["world_records"])
        with pytest.raises(ValueError, match="Missing required raw columns"):
            norm.normalize(incomplete)

    def test_duplicate_athlete_id_raises(self, swimming_config_path, swimming_pipeline_df):
        norm = SwimmingNormalizer(swimming_config_path)
        dup = pd.concat(
            [swimming_pipeline_df, swimming_pipeline_df.iloc[[0]]], ignore_index=True
        )
        with pytest.raises(ValueError, match="Duplicate athlete_id"):
            norm.normalize(dup)


# ── TestEraAdjustment ─────────────────────────────────────────────────────────

class TestEraAdjustment:
    def test_pre_modern_wc_gold_individual_is_imputed(
        self, swimming_config_path, swimming_pipeline_df
    ):
        """Pre-Modern athlete's wc_gold_individual must be replaced with imputed value."""
        norm = SwimmingNormalizer(swimming_config_path)
        result = norm.normalize(swimming_pipeline_df, era_adjust=True)
        pre_modern = result[result["era"] == "Pre-Modern"].iloc[0]
        # sw_t03: olympic_gold_individual=4, ratio=2.89 → estimated = 11.56
        assert pre_modern["wc_gold_individual"] == pytest.approx(11.56, rel=1e-4)

    def test_pre_modern_raw_zero_becomes_positive(
        self, swimming_config_path, swimming_pipeline_df
    ):
        """Ensure the adjustment actually changes a 0 to something positive."""
        raw_zero = swimming_pipeline_df[swimming_pipeline_df["era"] == "Pre-Modern"][
            "wc_gold_individual"
        ].iloc[0]
        assert raw_zero == 0.0

        norm = SwimmingNormalizer(swimming_config_path)
        result = norm.normalize(swimming_pipeline_df, era_adjust=True)
        adjusted = result[result["era"] == "Pre-Modern"]["wc_gold_individual"].iloc[0]
        assert adjusted > 0.0

    def test_modern_athlete_wc_gold_unchanged(
        self, swimming_config_path, swimming_pipeline_df
    ):
        """Modern era athletes must not have their wc_gold_individual modified."""
        raw = swimming_pipeline_df[swimming_pipeline_df["era"] == "Modern"][
            "wc_gold_individual"
        ].iloc[0]
        norm = SwimmingNormalizer(swimming_config_path)
        result = norm.normalize(swimming_pipeline_df, era_adjust=True)
        adjusted = result[result["era"] == "Modern"]["wc_gold_individual"].iloc[0]
        assert adjusted == pytest.approx(float(raw), rel=1e-6)

    def test_amateur_athlete_wc_gold_unchanged(
        self, swimming_config_path, swimming_pipeline_df
    ):
        """Amateur era athletes must not have their wc_gold_individual modified."""
        raw = swimming_pipeline_df[swimming_pipeline_df["era"] == "Amateur"][
            "wc_gold_individual"
        ].iloc[0]
        norm = SwimmingNormalizer(swimming_config_path)
        result = norm.normalize(swimming_pipeline_df, era_adjust=True)
        adjusted = result[result["era"] == "Amateur"]["wc_gold_individual"].iloc[0]
        assert adjusted == pytest.approx(float(raw), rel=1e-6)

    def test_era_adjust_false_preserves_pre_modern_zero(
        self, swimming_config_path, swimming_pipeline_df
    ):
        """When era_adjust=False, Pre-Modern wc_gold_individual stays 0."""
        norm = SwimmingNormalizer(swimming_config_path)
        result = norm.normalize(swimming_pipeline_df, era_adjust=False)
        pre_modern_wc = result[result["era"] == "Pre-Modern"]["wc_gold_individual"].iloc[0]
        assert pre_modern_wc == pytest.approx(0.0)


# ── TestOutputColumns ─────────────────────────────────────────────────────────

class TestOutputColumns:
    def test_all_normalized_columns_present(
        self, swimming_config_path, swimming_pipeline_df
    ):
        norm = SwimmingNormalizer(swimming_config_path)
        result = norm.normalize(swimming_pipeline_df)
        for col in _EXPECTED_OUTPUT_COLUMNS:
            assert col in result.columns, f"Missing normalized column: {col}"

    def test_all_normalized_values_in_zero_to_100(
        self, swimming_config_path, swimming_pipeline_df
    ):
        norm = SwimmingNormalizer(swimming_config_path)
        result = norm.normalize(swimming_pipeline_df)
        for col in _EXPECTED_OUTPUT_COLUMNS:
            assert result[col].between(0.0, 100.0).all(), (
                f"{col} has values outside [0, 100]: {result[col].tolist()}"
            )

    def test_raw_columns_are_preserved(
        self, swimming_config_path, swimming_pipeline_df
    ):
        norm = SwimmingNormalizer(swimming_config_path)
        result = norm.normalize(swimming_pipeline_df)
        assert "name" in result.columns
        assert "era" in result.columns
        assert "career_years" in result.columns

    def test_output_columns_method_matches_metric_map(self, swimming_config_path):
        norm = SwimmingNormalizer(swimming_config_path)
        assert set(norm.output_columns()) == set(_EXPECTED_OUTPUT_COLUMNS)


# ── TestNormalizationContracts ────────────────────────────────────────────────

class TestNormalizationContracts:
    def test_leader_in_each_metric_gets_100(
        self, swimming_config_path, swimming_pipeline_df
    ):
        """MinMax guarantee: the population maximum normalizes to 100."""
        norm = SwimmingNormalizer(swimming_config_path)
        result = norm.normalize(swimming_pipeline_df, era_adjust=False)
        for col in _EXPECTED_OUTPUT_COLUMNS:
            assert result[col].max() == pytest.approx(100.0, abs=1e-9), (
                f"{col}: max is not 100.0"
            )

    def test_trailer_in_each_metric_gets_0(
        self, swimming_config_path, swimming_pipeline_df
    ):
        """MinMax guarantee: the population minimum normalizes to 0."""
        norm = SwimmingNormalizer(swimming_config_path)
        result = norm.normalize(swimming_pipeline_df, era_adjust=False)
        for col in _EXPECTED_OUTPUT_COLUMNS:
            assert result[col].min() == pytest.approx(0.0, abs=1e-9), (
                f"{col}: min is not 0.0"
            )

    def test_constant_column_returns_50(self, swimming_config_path, swimming_pipeline_df):
        """normalize_series contract: constant input → 50.0 (not 0 or NaN)."""
        norm = SwimmingNormalizer(swimming_config_path)
        flat_df = swimming_pipeline_df.copy()
        flat_df["world_records"] = 5  # all same value
        result = norm.normalize(flat_df, era_adjust=False)
        assert result["world_records_normalized"].eq(50.0).all()

    def test_normalize_does_not_mutate_input(
        self, swimming_config_path, swimming_pipeline_df
    ):
        norm = SwimmingNormalizer(swimming_config_path)
        original_wc = swimming_pipeline_df["wc_gold_individual"].copy()
        norm.normalize(swimming_pipeline_df, era_adjust=True)
        pd.testing.assert_series_equal(
            swimming_pipeline_df["wc_gold_individual"], original_wc
        )
