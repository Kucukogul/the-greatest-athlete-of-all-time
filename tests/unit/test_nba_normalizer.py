"""Unit tests for NBANormalizer.

All tests use synthetic DataFrames — no file I/O, no real NBA data.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.normalize.nba_normalizer import (
    NBANormalizer,
    _ACHIEVEMENT_MAP,
    _CAREER_MAP,
    _EFFICIENCY_MAP,
    _LAYER_MAPS,
    _LONGEVITY_MAP,
    _PEAK_MAP,
    _REQUIRED_COLS,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_row(**overrides) -> dict:
    """Minimal valid player career row with sensible defaults."""
    row = {
        # identity
        "player_id":    "jorda001",
        "name":         "Michael Jordan",
        "era":          "modern",
        "era_encoded":  2,
        # peak layer
        "peak_bpm":          12.0,
        "peak_ws48":         0.30,
        "peak_vorp_season":  9.8,
        # career layer
        "career_vorp":   115.4,
        "career_ws":     214.0,
        "career_bpm":    9.2,
        # efficiency layer
        "career_ws48":   0.250,
        "career_ts_pct": 0.535,
        # achievement layer
        "championships":       6,
        "mvp_awards":          5,
        "finals_mvp":          6,
        "all_nba_1st_team":   10,
        "all_nba_2nd_team":    1,
        "all_star_selections": 14,
        "dpoy_awards":         1,
        "all_defensive_1st_team": 9,
        # longevity layer
        "years_active": 19,
        # pipeline extras (not used by normalizer but common in pipeline output)
        "total_games":        1072,
        "qualifying_seasons":   13,
    }
    row.update(overrides)
    return row


def _player(player_id: str, **metrics) -> dict:
    """Build a player row with a unique player_id, merging provided metrics."""
    return _base_row(player_id=player_id, name=player_id, **metrics)


def _df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ── TestValidation ─────────────────────────────────────────────────────────────

class TestValidation:
    def test_empty_dataframe_raises(self):
        n = NBANormalizer()
        with pytest.raises(ValueError, match="empty"):
            n.normalize(pd.DataFrame())

    def test_missing_column_raises(self):
        df = _df(_player("a"))
        df = df.drop(columns=["peak_bpm"])
        n = NBANormalizer()
        with pytest.raises(ValueError, match="Missing required columns"):
            n.normalize(df)

    def test_missing_player_id_raises(self):
        df = _df(_player("a"))
        df = df.drop(columns=["player_id"])
        n = NBANormalizer()
        with pytest.raises(ValueError, match="Missing required columns"):
            n.normalize(df)

    def test_missing_era_raises(self):
        df = _df(_player("a"))
        df = df.drop(columns=["era"])
        n = NBANormalizer()
        with pytest.raises(ValueError, match="Missing required columns"):
            n.normalize(df)

    def test_duplicate_player_id_raises(self):
        df = _df(_player("dup"), _player("dup"))
        n = NBANormalizer()
        with pytest.raises(ValueError, match="Duplicate player_id"):
            n.normalize(df)

    def test_valid_single_player_passes(self):
        df = _df(_player("a"))
        n = NBANormalizer()
        result = n.normalize(df)
        assert len(result) == 1


# ── TestOutputColumns ─────────────────────────────────────────────────────────

class TestOutputColumns:
    def test_output_columns_returns_list_of_strings(self):
        cols = NBANormalizer.output_columns()
        assert isinstance(cols, list)
        assert all(isinstance(c, str) for c in cols)

    def test_output_columns_match_all_layer_map_keys(self):
        expected = [col for m in _LAYER_MAPS.values() for col in m.keys()]
        assert NBANormalizer.output_columns() == expected

    def test_all_output_columns_present_after_normalize(self):
        df = _df(_player("a"), _player("b"))
        result = NBANormalizer().normalize(df)
        for col in NBANormalizer.output_columns():
            assert col in result.columns, f"Missing column: {col}"


# ── TestLayerColumns ──────────────────────────────────────────────────────────

class TestLayerColumns:
    def test_peak_layer(self):
        assert NBANormalizer.layer_columns("peak") == list(_PEAK_MAP.keys())

    def test_career_layer(self):
        assert NBANormalizer.layer_columns("career") == list(_CAREER_MAP.keys())

    def test_efficiency_layer(self):
        assert NBANormalizer.layer_columns("efficiency") == list(_EFFICIENCY_MAP.keys())

    def test_achievement_layer(self):
        assert NBANormalizer.layer_columns("achievement") == list(_ACHIEVEMENT_MAP.keys())

    def test_longevity_layer(self):
        assert NBANormalizer.layer_columns("longevity") == list(_LONGEVITY_MAP.keys())

    def test_unknown_layer_raises(self):
        with pytest.raises(ValueError, match="Unknown layer"):
            NBANormalizer.layer_columns("nonexistent")

    def test_layers_method_returns_all_keys(self):
        assert NBANormalizer.layers() == list(_LAYER_MAPS.keys())


# ── TestMinMaxNormalization ───────────────────────────────────────────────────

class TestMinMaxNormalization:
    """Core invariants: min→0, max→100, others in-between, immutable input."""

    def test_min_player_gets_zero(self):
        df = _df(
            _player("low",  peak_bpm=1.0),
            _player("mid",  peak_bpm=5.0),
            _player("high", peak_bpm=9.0),
        )
        result = NBANormalizer().normalize(df)
        low = result.loc[result["player_id"] == "low", "peak_bpm_normalized"].iloc[0]
        assert low == pytest.approx(0.0)

    def test_max_player_gets_100(self):
        df = _df(
            _player("low",  peak_bpm=1.0),
            _player("mid",  peak_bpm=5.0),
            _player("high", peak_bpm=9.0),
        )
        result = NBANormalizer().normalize(df)
        high = result.loc[result["player_id"] == "high", "peak_bpm_normalized"].iloc[0]
        assert high == pytest.approx(100.0)

    def test_midpoint_player_is_between(self):
        df = _df(
            _player("low",  peak_bpm=0.0),
            _player("high", peak_bpm=10.0),
            _player("mid",  peak_bpm=4.0),
        )
        result = NBANormalizer().normalize(df)
        mid = result.loc[result["player_id"] == "mid", "peak_bpm_normalized"].iloc[0]
        assert mid == pytest.approx(40.0)

    def test_all_equal_metrics_get_50(self):
        """When all players have the same value, normalize_series returns 50.0."""
        df = _df(
            _player("a", peak_bpm=8.0),
            _player("b", peak_bpm=8.0),
            _player("c", peak_bpm=8.0),
        )
        result = NBANormalizer().normalize(df)
        vals = result["peak_bpm_normalized"]
        assert (vals == 50.0).all()

    def test_original_dataframe_not_mutated(self):
        df = _df(_player("a"), _player("b"))
        original_cols = set(df.columns)
        NBANormalizer().normalize(df)
        assert set(df.columns) == original_cols

    def test_normalized_range_zero_to_100(self):
        df = _df(
            _player("a", career_ws=50.0),
            _player("b", career_ws=150.0),
            _player("c", career_ws=250.0),
        )
        result = NBANormalizer().normalize(df)
        vals = result["career_ws_normalized"]
        assert vals.min() >= 0.0
        assert vals.max() <= 100.0


# ── TestNanHandling ───────────────────────────────────────────────────────────

class TestNanHandling:
    """Pre-advanced era players have NaN for BPM/VORP — must propagate through normalization."""

    def test_nan_peak_bpm_preserved(self):
        df = _df(
            _player("modern",     peak_bpm=10.0),
            _player("pre_adv",    peak_bpm=float("nan")),
        )
        result = NBANormalizer().normalize(df)
        pre_bpm = result.loc[result["player_id"] == "pre_adv", "peak_bpm_normalized"].iloc[0]
        assert math.isnan(pre_bpm)

    def test_nan_does_not_bias_minmax(self):
        """NaN rows must be excluded from min/max computation."""
        df = _df(
            _player("a", peak_bpm=0.0),
            _player("b", peak_bpm=10.0),
            _player("nan_player", peak_bpm=float("nan")),
        )
        result = NBANormalizer().normalize(df)
        a_score = result.loc[result["player_id"] == "a", "peak_bpm_normalized"].iloc[0]
        b_score = result.loc[result["player_id"] == "b", "peak_bpm_normalized"].iloc[0]
        assert a_score == pytest.approx(0.0)
        assert b_score == pytest.approx(100.0)

    def test_nan_career_vorp_preserved(self):
        df = _df(
            _player("modern", career_vorp=100.0),
            _player("early",  career_vorp=float("nan")),
        )
        result = NBANormalizer().normalize(df)
        early_vorp = result.loc[result["player_id"] == "early", "career_vorp_normalized"].iloc[0]
        assert math.isnan(early_vorp)

    def test_nan_peak_vorp_preserved(self):
        df = _df(
            _player("a", peak_vorp_season=9.0),
            _player("b", peak_vorp_season=float("nan")),
        )
        result = NBANormalizer().normalize(df)
        b_vorp = result.loc[result["player_id"] == "b", "peak_vorp_normalized"].iloc[0]
        assert math.isnan(b_vorp)

    def test_nan_career_bpm_preserved(self):
        df = _df(
            _player("a", career_bpm=5.0),
            _player("b", career_bpm=float("nan")),
        )
        result = NBANormalizer().normalize(df)
        b_bpm = result.loc[result["player_id"] == "b", "career_bpm_normalized"].iloc[0]
        assert math.isnan(b_bpm)

    def test_non_bpm_columns_unaffected_by_nan_rows(self):
        """WS normalization should be unaffected when a row has NaN BPM."""
        df = _df(
            _player("a", career_ws=100.0, peak_bpm=float("nan")),
            _player("b", career_ws=200.0, peak_bpm=float("nan")),
        )
        result = NBANormalizer().normalize(df)
        ws_a = result.loc[result["player_id"] == "a", "career_ws_normalized"].iloc[0]
        ws_b = result.loc[result["player_id"] == "b", "career_ws_normalized"].iloc[0]
        assert ws_a == pytest.approx(0.0)
        assert ws_b == pytest.approx(100.0)

    def test_all_nan_bpm_column_remains_all_nan(self):
        df = _df(
            _player("a", peak_bpm=float("nan")),
            _player("b", peak_bpm=float("nan")),
        )
        result = NBANormalizer().normalize(df)
        assert result["peak_bpm_normalized"].isna().all()

    def test_nan_debug_log_emitted(self, caplog):
        import logging
        df = _df(
            _player("modern", peak_bpm=8.0),
            _player("early",  peak_bpm=float("nan")),
        )
        with caplog.at_level(logging.DEBUG, logger="src.normalize.nba_normalizer"):
            NBANormalizer().normalize(df)
        assert any("NaN peak_bpm_normalized" in r.message for r in caplog.records)


# ── TestAchievementNormalization ──────────────────────────────────────────────

class TestAchievementNormalization:
    def test_zero_championships_gets_zero(self):
        df = _df(
            _player("a", championships=0),
            _player("b", championships=6),
        )
        result = NBANormalizer().normalize(df)
        a_score = result.loc[result["player_id"] == "a", "championships_normalized"].iloc[0]
        assert a_score == pytest.approx(0.0)

    def test_max_championships_gets_100(self):
        df = _df(
            _player("a", championships=0),
            _player("b", championships=11),
        )
        result = NBANormalizer().normalize(df)
        b_score = result.loc[result["player_id"] == "b", "championships_normalized"].iloc[0]
        assert b_score == pytest.approx(100.0)

    def test_all_achievement_columns_normalized(self):
        df = _df(
            _player("a", championships=0, mvp_awards=0, finals_mvp=0,
                    all_nba_1st_team=0, all_nba_2nd_team=0,
                    all_star_selections=0, dpoy_awards=0,
                    all_defensive_1st_team=0),
            _player("b", championships=6, mvp_awards=5, finals_mvp=6,
                    all_nba_1st_team=10, all_nba_2nd_team=3,
                    all_star_selections=14, dpoy_awards=1,
                    all_defensive_1st_team=9),
        )
        result = NBANormalizer().normalize(df)
        for col in _ACHIEVEMENT_MAP:
            assert col in result.columns
            assert result[col].notna().any(), f"{col} should have non-NaN values"


# ── TestMetricLayerIndependence ────────────────────────────────────────────────

class TestMetricLayerIndependence:
    """Normalizing one layer must not affect another."""

    def test_peak_layer_normalized_independently_of_career(self):
        """Two players with same peak but different career stats → same peak normalized."""
        df = _df(
            _player("a", peak_bpm=5.0, career_bpm=1.0),
            _player("b", peak_bpm=5.0, career_bpm=9.0),
        )
        result = NBANormalizer().normalize(df)
        a_peak = result.loc[result["player_id"] == "a", "peak_bpm_normalized"].iloc[0]
        b_peak = result.loc[result["player_id"] == "b", "peak_bpm_normalized"].iloc[0]
        assert a_peak == pytest.approx(50.0)
        assert b_peak == pytest.approx(50.0)

    def test_efficiency_and_longevity_are_independent(self):
        df = _df(
            _player("a", career_ts_pct=0.50, years_active=5),
            _player("b", career_ts_pct=0.60, years_active=20),
        )
        result = NBANormalizer().normalize(df)
        a_ts = result.loc[result["player_id"] == "a", "career_ts_normalized"].iloc[0]
        a_lon = result.loc[result["player_id"] == "a", "longevity_normalized"].iloc[0]
        assert a_ts == pytest.approx(0.0)
        assert a_lon == pytest.approx(0.0)


# ── TestRequiredCols ─────────────────────────────────────────────────────────

class TestRequiredCols:
    def test_required_cols_contains_all_source_cols(self):
        expected = frozenset(
            {src for m in _LAYER_MAPS.values() for src in m.values()}
            | {"player_id", "era"}
        )
        assert _REQUIRED_COLS == expected

    def test_each_source_column_is_required(self):
        for layer_map in _LAYER_MAPS.values():
            for src_col in layer_map.values():
                assert src_col in _REQUIRED_COLS, f"{src_col} must be in _REQUIRED_COLS"
