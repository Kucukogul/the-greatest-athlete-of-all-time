from __future__ import annotations

import glob
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_TOUR_LEVELS: frozenset[str] = frozenset({"G", "M", "A", "F", "D"})

_RANKING_GLOBS = [
    "atp_rankings_70s.csv",
    "atp_rankings_80s.csv",
    "atp_rankings_90s.csv",
    "atp_rankings_00s.csv",
    "atp_rankings_10s.csv",
    "atp_rankings_20s.csv",
    "atp_rankings_current.csv",
]


class TennisPipeline:
    """
    Raw ATP CSV data → one career-stats row per player.

    Output schema matches TennisNormalizer._REQUIRED_RAW_COLUMNS exactly.
    Pipeline is a pure transformation: no side effects, no file writes.

    Usage:
        df = TennisPipeline(data_dir).run(player_ids=[103819, 104745, 104925])
        df = TennisPipeline(data_dir).run(min_matches=200)  # all qualifying players
    """

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)

    def run(
        self,
        player_ids: list[int] | None = None,
        min_matches: int = 200,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per player, ready for TennisNormalizer.

        Args:
            player_ids:  Explicit list of ATP player IDs. If None, selects all
                         players with >= min_matches career tour-level matches.
            min_matches: Minimum career matches threshold when player_ids is None.
        """
        logger.info("Loading match data from %s", self._data_dir)
        matches = self._load_matches()
        rankings = self._load_rankings()
        players = self._load_players()

        if player_ids is None:
            player_ids = self._qualifying_players(matches, min_matches)
            logger.info("Selected %d qualifying players", len(player_ids))

        records = []
        for pid in player_ids:
            record = self._compute_career_stats(pid, matches, rankings, players)
            if record is not None:
                records.append(record)

        result = pd.DataFrame(records)
        logger.info("Pipeline complete — %d player records produced", len(result))
        return result

    def _load_matches(self) -> pd.DataFrame:
        pattern = str(self._data_dir / "atp_matches_[0-9]*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No match files found in {self._data_dir}")

        chunks = []
        for f in files:
            df = pd.read_csv(
                f,
                usecols=[
                    "tourney_id", "tourney_name", "surface", "tourney_level",
                    "tourney_date", "round",
                    "winner_id", "winner_name",
                    "loser_id", "loser_name",
                    "winner_rank", "loser_rank",
                ],
                low_memory=False,
            )
            df = df[df["tourney_level"].isin(_TOUR_LEVELS)]
            chunks.append(df)

        matches = pd.concat(chunks, ignore_index=True)
        matches["year"] = matches["tourney_date"].astype(str).str[:4].astype(int)
        matches["surface"] = matches["surface"].str.strip()
        return matches

    def _load_rankings(self) -> pd.DataFrame:
        chunks = []
        for filename in _RANKING_GLOBS:
            path = self._data_dir / filename
            if path.exists():
                df = pd.read_csv(path, usecols=["ranking_date", "rank", "player"])
                chunks.append(df)
        rankings = pd.concat(chunks, ignore_index=True)
        rankings["ranking_date"] = pd.to_datetime(
            rankings["ranking_date"].astype(str), format="%Y%m%d"
        )
        rankings["year"] = rankings["ranking_date"].dt.year
        return rankings

    def _load_players(self) -> pd.DataFrame:
        path = self._data_dir / "atp_players.csv"
        return pd.read_csv(
            path,
            usecols=["player_id", "name_first", "name_last"],
        )

    def _qualifying_players(self, matches: pd.DataFrame, min_matches: int) -> list[int]:
        wins = matches["winner_id"].value_counts()
        losses = matches["loser_id"].value_counts()
        total = wins.add(losses, fill_value=0)
        return list(total[total >= min_matches].index.astype(int))

    def _compute_career_stats(
        self,
        player_id: int,
        matches: pd.DataFrame,
        rankings: pd.DataFrame,
        players: pd.DataFrame,
    ) -> dict | None:
        wins_df = matches[matches["winner_id"] == player_id]
        losses_df = matches[matches["loser_id"] == player_id]

        career_wins = len(wins_df)
        career_matches = career_wins + len(losses_df)
        if career_matches == 0:
            return None

        player_row = players[players["player_id"] == player_id]
        if player_row.empty:
            return None
        name = f"{player_row.iloc[0]['name_first']} {player_row.iloc[0]['name_last']}"

        player_rankings = rankings[rankings["player"] == player_id]
        era = self._assign_era(wins_df, losses_df)

        return {
            "athlete_id":        f"atp_{player_id}",
            "name":              name,
            "sport":             "tennis",
            "era":               era,
            "grand_slams":       self._title_count(wins_df, "G"),
            "masters_titles":    self._title_count(wins_df, "M"),
            "career_wins":       career_wins,
            "career_matches":    career_matches,
            "weeks_at_no1":      self._weeks_at_rank(player_rankings, 1),
            "yearend_no1_count": self._yearend_rank_count(player_rankings, 1),
            "years_in_top5":     self._years_at_rank(player_rankings, 5),
            "finals_won":        self._finals_count(wins_df),
            "finals_played":     self._finals_count(wins_df) + self._finals_count(losses_df),
            "h2h_top10_wins":    self._h2h_wins(wins_df, rank_col="loser_rank", threshold=10),
            "h2h_top10_total":   (
                self._h2h_wins(wins_df, rank_col="loser_rank", threshold=10)
                + self._h2h_wins(losses_df, rank_col="winner_rank", threshold=10)
            ),
            "hard_win_pct":      self._surface_win_rate(wins_df, losses_df, "Hard"),
            "clay_win_pct":      self._surface_win_rate(wins_df, losses_df, "Clay"),
            "grass_win_pct":     self._surface_win_rate(wins_df, losses_df, "Grass"),
        }

    @staticmethod
    def _title_count(wins: pd.DataFrame, level: str) -> int:
        return int(
            ((wins["tourney_level"] == level) & (wins["round"] == "F")).sum()
        )

    @staticmethod
    def _finals_count(df: pd.DataFrame) -> int:
        return int((df["round"] == "F").sum())

    @staticmethod
    def _h2h_wins(wins: pd.DataFrame, rank_col: str, threshold: int) -> int:
        return int(
            (wins[rank_col].fillna(999) <= threshold).sum()
        )

    @staticmethod
    def _surface_win_rate(
        wins: pd.DataFrame,
        losses: pd.DataFrame,
        surface: str,
    ) -> float:
        w = (wins["surface"] == surface).sum()
        l = (losses["surface"] == surface).sum()
        total = w + l
        return round(w / total, 4) if total > 0 else 0.0

    @staticmethod
    def _weeks_at_rank(rankings: pd.DataFrame, rank: int) -> int:
        return int((rankings["rank"] == rank).sum())

    @staticmethod
    def _yearend_rank_count(rankings: pd.DataFrame, rank: int) -> int:
        if rankings.empty:
            return 0
        last_per_year = (
            rankings.sort_values("ranking_date")
            .groupby("year")
            .last()
            .reset_index()
        )
        return int((last_per_year["rank"] == rank).sum())

    @staticmethod
    def _years_at_rank(rankings: pd.DataFrame, rank_threshold: int) -> int:
        if rankings.empty:
            return 0
        return int(
            rankings[rankings["rank"] <= rank_threshold]["year"].nunique()
        )

    @staticmethod
    def _assign_era(wins: pd.DataFrame, losses: pd.DataFrame) -> str:
        all_years = pd.concat([wins["year"], losses["year"]])
        if all_years.empty:
            return "Big 3 Era"
        peak_year = int(all_years.value_counts().idxmax())
        if peak_year >= 2004:
            return "Big 3 Era"
        if peak_year >= 1990:
            return "Modern Era"
        return "Open Era"
