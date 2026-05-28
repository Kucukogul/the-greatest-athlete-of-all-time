import pandas as pd
import pytest

from src.scoring.normalizer import normalize_series, percentile_rank
from src.scoring.scorer import AthleteScorer, ScoringConfig


VALID_CONFIG = ScoringConfig(weights={"ppg_normalized": 0.5, "per_normalized": 0.5})


def make_df(**kwargs: list) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


class TestScoringConfig:
    def test_raises_on_weights_not_summing_to_one(self):
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            ScoringConfig(weights={"a": 0.3, "b": 0.3})

    def test_accepts_valid_weights(self):
        config = ScoringConfig(weights={"a": 0.6, "b": 0.4})
        assert config.weights["a"] == 0.6

    def test_floating_point_tolerance(self):
        # 0.1 + 0.2 + 0.7 = 1.0000000000000002 in IEEE 754
        ScoringConfig(weights={"a": 0.1, "b": 0.2, "c": 0.7})


class TestAthleteScorer:
    def test_score_returns_series(self):
        df = make_df(ppg_normalized=[80.0, 60.0, 40.0], per_normalized=[70.0, 50.0, 30.0])
        scores = AthleteScorer(VALID_CONFIG).score(df)
        assert isinstance(scores, pd.Series)
        assert len(scores) == 3

    def test_empty_dataframe_returns_empty_series(self):
        df = pd.DataFrame(columns=["ppg_normalized", "per_normalized"])
        scores = AthleteScorer(VALID_CONFIG).score(df)
        assert scores.empty

    def test_missing_column_raises_value_error(self):
        df = make_df(ppg_normalized=[80.0])
        with pytest.raises(ValueError, match="missing required metric columns"):
            AthleteScorer(VALID_CONFIG).score(df)

    def test_score_breakdown_contains_total(self):
        df = make_df(ppg_normalized=[80.0], per_normalized=[60.0])
        breakdown = AthleteScorer(VALID_CONFIG).score_breakdown(df)
        assert "total" in breakdown.columns

    def test_higher_stats_yield_higher_score(self):
        df = make_df(ppg_normalized=[90.0, 10.0], per_normalized=[90.0, 10.0])
        scores = AthleteScorer(VALID_CONFIG).score(df)
        assert scores.iloc[0] > scores.iloc[1]

    def test_rank_based_strategy(self):
        config = ScoringConfig(weights={"ppg_normalized": 1.0}, strategy="rank_based")
        df = make_df(ppg_normalized=[10.0, 50.0, 90.0])
        scores = AthleteScorer(config).score(df)
        assert scores.iloc[2] > scores.iloc[0]

    def test_score_uses_fixture(self, sample_athlete_df):
        config = ScoringConfig(
            weights={"ppg_normalized": 0.3, "rpg_normalized": 0.2, "apg_normalized": 0.2, "per_normalized": 0.3}
        )
        scores = AthleteScorer(config).score(sample_athlete_df)
        assert len(scores) == 5
        assert scores.between(0, 100).all()


class TestNormalizer:
    def test_minmax_min_is_zero(self):
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        normalized = normalize_series(s, "minmax")
        assert normalized.min() == pytest.approx(0.0)

    def test_minmax_max_is_100(self):
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        normalized = normalize_series(s, "minmax")
        assert normalized.max() == pytest.approx(100.0)

    def test_constant_series_returns_50(self):
        s = pd.Series([5.0, 5.0, 5.0])
        result = normalize_series(s, "minmax")
        assert all(result == 50.0)

    def test_percentile_rank_max_is_100(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        ranks = percentile_rank(s)
        assert ranks.max() == pytest.approx(100.0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_series(pd.Series([1.0, 2.0]), method="unknown")
