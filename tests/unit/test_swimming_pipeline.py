"""Unit tests for SwimmingPipeline. Real-data tests skip if swimming_athletes_raw.csv is absent."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipelines.swimming_pipeline import SwimmingPipeline

# ── Helpers ───────────────────────────────────────────────────────────────────

def _pipe() -> SwimmingPipeline:
    """Pipeline instance pointing nowhere — for calling internal methods only."""
    return SwimmingPipeline(Path("."))


def _raw_row(**overrides) -> dict:
    base = {
        "athlete_id": "sw001",
        "name": "Test Swimmer",
        "nationality": "USA",
        "era": "Modern",
        "birth_year": 1985,
        "career_start": 2000,
        "career_end": 2016,
        "olympic_gold_individual": 5,
        "olympic_gold_relay": 2,
        "olympic_silver_individual": 1,
        "olympic_silver_relay": 1,
        "olympic_bronze_individual": 0,
        "olympic_bronze_relay": 0,
        "olympic_games_count": 4,
        "wc_gold_individual": 8,
        "wc_gold_relay": 3,
        "wc_silver_individual": 2,
        "wc_bronze_individual": 1,
        "world_records": 10,
        "events_dominated": 3,
    }
    base.update(overrides)
    return base


def _df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ── TestValidation ────────────────────────────────────────────────────────────

class TestValidation:
    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _pipe()._validate(pd.DataFrame())

    def test_missing_required_column_raises(self):
        df = _df(_raw_row()).drop(columns=["world_records"])
        with pytest.raises(ValueError, match="Missing required columns"):
            _pipe()._validate(df)

    def test_duplicate_athlete_id_raises(self):
        df = _df(_raw_row(), _raw_row())
        with pytest.raises(ValueError, match="Duplicate athlete_id"):
            _pipe()._validate(df)

    def test_invalid_era_label_raises(self):
        df = _df(_raw_row(era="Ancient"))
        with pytest.raises(ValueError, match="Unknown era labels"):
            _pipe()._validate(df)

    def test_negative_numeric_column_raises(self):
        df = _df(_raw_row(world_records=-1))
        with pytest.raises(ValueError, match="Negative values in 'world_records'"):
            _pipe()._validate(df)

    def test_career_end_before_start_raises(self):
        df = _df(_raw_row(career_start=2010, career_end=2005))
        with pytest.raises(ValueError, match="career_end < career_start"):
            _pipe()._validate(df)

    def test_events_dominated_exceeds_og_individual_raises(self):
        df = _df(_raw_row(olympic_gold_individual=2, events_dominated=3))
        with pytest.raises(ValueError, match="events_dominated exceeds olympic_gold_individual"):
            _pipe()._validate(df)

    def test_valid_row_does_not_raise(self):
        df = _df(_raw_row())
        _pipe()._validate(df)  # must not raise

    def test_all_valid_eras_accepted(self):
        df = _df(
            _raw_row(athlete_id="a1", era="Modern"),
            _raw_row(athlete_id="a2", era="Amateur"),
            _raw_row(athlete_id="a3", era="Pre-Modern"),
        )
        _pipe()._validate(df)


# ── TestDerivedFeatures ───────────────────────────────────────────────────────

class TestDerivedFeatures:
    def test_career_years_is_end_minus_start_plus_one(self):
        df = _df(_raw_row(career_start=2000, career_end=2016))
        result = _pipe()._derive(df)
        assert result["career_years"].iloc[0] == 17

    def test_career_years_single_year(self):
        df = _df(_raw_row(career_start=1972, career_end=1972))
        result = _pipe()._derive(df)
        assert result["career_years"].iloc[0] == 1

    def test_olympic_gold_total_is_individual_plus_relay(self):
        df = _df(_raw_row(olympic_gold_individual=13, olympic_gold_relay=10))
        result = _pipe()._derive(df)
        assert result["olympic_gold_total"].iloc[0] == 23

    def test_olympic_gold_total_relay_only(self):
        df = _df(_raw_row(olympic_gold_individual=0, olympic_gold_relay=4))
        result = _pipe()._derive(df)
        assert result["olympic_gold_total"].iloc[0] == 4

    def test_derive_does_not_modify_original(self):
        df = _df(_raw_row())
        original_cols = set(df.columns)
        _pipe()._derive(df)
        assert set(df.columns) == original_cols  # original untouched


# ── TestColumnOrder ───────────────────────────────────────────────────────────

class TestColumnOrder:
    def test_athlete_id_is_first_column(self):
        df = _df(_raw_row())
        result = _pipe()._reorder(_pipe()._derive(df))
        assert result.columns[0] == "athlete_id"

    def test_career_years_precedes_olympic_gold_individual(self):
        df = _df(_raw_row())
        result = _pipe()._reorder(_pipe()._derive(df))
        cols = list(result.columns)
        assert cols.index("career_years") < cols.index("olympic_gold_individual")

    def test_olympic_gold_total_follows_relay(self):
        df = _df(_raw_row())
        result = _pipe()._reorder(_pipe()._derive(df))
        cols = list(result.columns)
        assert cols.index("olympic_gold_relay") < cols.index("olympic_gold_total")


# ── TestRealData ──────────────────────────────────────────────────────────────

class TestRealData:
    _RAW_PATH = Path(__file__).parents[2] / "data" / "raw"

    def _skip_if_absent(self):
        if not (self._RAW_PATH / "swimming_athletes_raw.csv").exists():
            pytest.skip("swimming_athletes_raw.csv not present")

    def test_run_returns_35_rows(self):
        self._skip_if_absent()
        df = SwimmingPipeline(self._RAW_PATH).run()
        assert len(df) == 35

    def test_run_output_has_career_years_column(self):
        self._skip_if_absent()
        df = SwimmingPipeline(self._RAW_PATH).run()
        assert "career_years" in df.columns

    def test_run_output_has_olympic_gold_total_column(self):
        self._skip_if_absent()
        df = SwimmingPipeline(self._RAW_PATH).run()
        assert "olympic_gold_total" in df.columns

    def test_phelps_career_years_is_17(self):
        self._skip_if_absent()
        df = SwimmingPipeline(self._RAW_PATH).run()
        phelps = df[df["athlete_id"] == "sw001"].iloc[0]
        assert phelps["career_years"] == 17

    def test_phelps_olympic_gold_total_is_23(self):
        self._skip_if_absent()
        df = SwimmingPipeline(self._RAW_PATH).run()
        phelps = df[df["athlete_id"] == "sw001"].iloc[0]
        assert phelps["olympic_gold_total"] == 23

    def test_no_negative_values_in_numeric_columns(self):
        self._skip_if_absent()
        df = SwimmingPipeline(self._RAW_PATH).run()
        numeric = df.select_dtypes("number").columns
        assert (df[numeric] >= 0).all().all()
