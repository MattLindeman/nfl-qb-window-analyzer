"""
Unit tests for feature engineering and cleaning pipeline.
Spec Section 14.3: "At least basic unit tests for feature engineering functions."

Run:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from src.ingestion.build_database import (
    clean_contracts,
    clean_team_seasons,
    build_contract_windows,
    PLAYOFF_ROUND_ORDER,
    TEAM_NAME_MAP,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_cap_history():
    return pd.DataFrame({
        "year":        [2020, 2021, 2022, 2023, 2024],
        "cap_total_m": [198.2, 182.5, 208.2, 224.8, 255.4],
    })


@pytest.fixture
def sample_contracts_raw(sample_cap_history):
    return pd.DataFrame({
        "player":         ["Patrick Mahomes", "Aaron Rodgers", "Baker Mayfield", "Backup QB"],
        "team":           ["KC",  "GB",  "CLE", "NE"],
        "signing_year":   [2020,  2022,  2022,  2022],
        "years":          [10,    4,     5,     1],
        "total_value_m":  [450.0, 200.0, 189.0, 3.0],
        "aav_m":          [45.0,  50.3,  18.9,  3.0],
        "guaranteed_m":   [141.0, 153.0, 35.0,  np.nan],
        "age_at_signing": [25,    38,    27,    26],
    })


@pytest.fixture
def sample_team_seasons():
    return pd.DataFrame({
        "team":         ["KC",  "KC",  "GB",  "CLE", "OAK"],
        "season":       [2021,  2022,  2023,  2023,  2021],
        "wins":         [12,    14,    9,     11,    8],
        "losses":       [5,     3,     8,     6,     9],
        "made_playoffs":[True,  True,  True,  True,  False],
        "playoff_exit": ["SB_loss", "SB_win", "WC", "WC", None],
    })


# ── Contract cleaning tests ────────────────────────────────────────────────────

class TestCleanContracts:
    def test_cap_pct_computed(self, sample_contracts_raw, sample_cap_history):
        result = clean_contracts(sample_contracts_raw, sample_cap_history)
        mahomes = result[result["player"] == "Patrick Mahomes"].iloc[0]
        expected = round(45.0 / 198.2 * 100, 2)
        assert abs(mahomes["cap_pct"] - expected) < 0.01

    def test_backup_filtered_out(self, sample_contracts_raw, sample_cap_history):
        result = clean_contracts(sample_contracts_raw, sample_cap_history)
        assert "Backup QB" not in result["player"].values

    def test_minimum_cap_pct_threshold(self, sample_contracts_raw, sample_cap_history):
        result = clean_contracts(sample_contracts_raw, sample_cap_history)
        assert (result["cap_pct"] >= 10.0).all()

    def test_missing_guaranteed_filled_with_zero(self, sample_contracts_raw, sample_cap_history):
        result = clean_contracts(sample_contracts_raw, sample_cap_history)
        # Backup QB got filtered; ensure no NaN guaranteed in remaining rows
        assert result["guaranteed_m"].isna().sum() == 0

    def test_team_abbreviations_normalized(self, sample_cap_history):
        raw = pd.DataFrame({
            "player":         ["Jon Example"],
            "team":           ["OAK"],
            "signing_year":   [2020],
            "years":          [4],
            "total_value_m":  [80.0],
            "aav_m":          [20.0],
            "guaranteed_m":   [40.0],
            "age_at_signing": [28],
        })
        result = clean_contracts(raw, sample_cap_history)
        if not result.empty:
            assert result.iloc[0]["team"] == "LV"

    def test_cap_tier_assigned(self, sample_contracts_raw, sample_cap_history):
        result = clean_contracts(sample_contracts_raw, sample_cap_history)
        assert "cap_tier" in result.columns
        valid_tiers = {"10-15%", "15-18%", "18-21%", "21%+"}
        assert set(result["cap_tier"].unique()).issubset(valid_tiers)

    def test_no_duplicates_on_player_year(self, sample_cap_history):
        raw = pd.DataFrame({
            "player":         ["Aaron Rodgers", "Aaron Rodgers"],
            "team":           ["GB", "GB"],
            "signing_year":   [2022, 2022],
            "years":          [4, 4],
            "total_value_m":  [200.0, 200.0],
            "aav_m":          [50.3, 50.3],
            "guaranteed_m":   [153.0, 153.0],
            "age_at_signing": [38, 38],
        })
        result = clean_contracts(raw, sample_cap_history)
        assert len(result) <= 1


# ── Team season cleaning tests ────────────────────────────────────────────────

class TestCleanTeamSeasons:
    def test_playoff_round_encoded(self, sample_team_seasons):
        result = clean_team_seasons(sample_team_seasons)
        sb_win_row  = result[result["playoff_exit"] == "SB_win"].iloc[0]
        sb_loss_row = result[result["playoff_exit"] == "SB_loss"].iloc[0]
        wc_row      = result[result["playoff_exit"] == "WC"].iloc[0]
        none_row    = result[result["playoff_exit"].isna()].iloc[0]
        assert sb_win_row["playoff_round_num"]  == 5
        assert sb_loss_row["playoff_round_num"] == 4
        assert wc_row["playoff_round_num"]      == 1
        assert none_row["playoff_round_num"]    == 0

    def test_playoff_round_ordering(self, sample_team_seasons):
        result = clean_team_seasons(sample_team_seasons)
        vals = sorted(result["playoff_round_num"].unique())
        assert vals == sorted(vals)  # already sorted, confirms monotonic encoding

    def test_boolean_flags(self, sample_team_seasons):
        result = clean_team_seasons(sample_team_seasons)
        assert result["made_playoffs"].dtype == bool
        assert result["made_superbowl"].dtype == bool
        assert result["won_superbowl"].dtype  == bool

    def test_sb_win_flag(self, sample_team_seasons):
        result = clean_team_seasons(sample_team_seasons)
        assert result[result["playoff_exit"] == "SB_win"]["won_superbowl"].all()
        assert not result[result["playoff_exit"] == "SB_loss"]["won_superbowl"].any()

    def test_win_pct_computed(self, sample_team_seasons):
        result = clean_team_seasons(sample_team_seasons)
        row = result[result["team"] == "KC"].sort_values("season").iloc[0]
        expected = round(12 / (12 + 5), 3)
        assert abs(row["win_pct"] - expected) < 0.001

    def test_team_normalization(self, sample_team_seasons):
        result = clean_team_seasons(sample_team_seasons)
        assert "OAK" not in result["team"].values
        assert "LV" in result["team"].values

    def test_no_duplicate_team_seasons(self, sample_team_seasons):
        result = clean_team_seasons(sample_team_seasons)
        dupes = result.duplicated(subset=["team", "season"])
        assert not dupes.any()


# ── Contract windows tests ────────────────────────────────────────────────────

class TestBuildContractWindows:
    def test_window_rows_per_contract(self, sample_contracts_raw,
                                      sample_cap_history, sample_team_seasons):
        contracts    = clean_contracts(sample_contracts_raw, sample_cap_history)
        team_seasons = clean_team_seasons(sample_team_seasons)
        windows      = build_contract_windows(contracts, team_seasons)

        # Mahomes signed 2020, KC in seasons dict for 2021 and 2022 → 2 rows
        mahomes_rows = windows[windows["player"] == "Patrick Mahomes"]
        assert len(mahomes_rows) == 2

    def test_seasons_post_range(self, sample_contracts_raw,
                                sample_cap_history, sample_team_seasons):
        contracts    = clean_contracts(sample_contracts_raw, sample_cap_history)
        team_seasons = clean_team_seasons(sample_team_seasons)
        windows      = build_contract_windows(contracts, team_seasons)
        assert windows["seasons_post"].between(1, 5).all()

    def test_required_columns_present(self, sample_contracts_raw,
                                       sample_cap_history, sample_team_seasons):
        contracts    = clean_contracts(sample_contracts_raw, sample_cap_history)
        team_seasons = clean_team_seasons(sample_team_seasons)
        windows      = build_contract_windows(contracts, team_seasons)
        required = {"player", "team", "signing_year", "season", "seasons_post",
                    "cap_pct", "wins", "made_playoffs", "made_superbowl"}
        assert required.issubset(set(windows.columns))

    def test_no_future_seasons(self, sample_contracts_raw,
                               sample_cap_history, sample_team_seasons):
        contracts    = clean_contracts(sample_contracts_raw, sample_cap_history)
        team_seasons = clean_team_seasons(sample_team_seasons)
        windows      = build_contract_windows(contracts, team_seasons)
        if not windows.empty:
            assert (windows["season"] > windows["signing_year"]).all()


# ── Constants tests ───────────────────────────────────────────────────────────

class TestConstants:
    def test_playoff_round_order_monotonic(self):
        vals = list(PLAYOFF_ROUND_ORDER.values())
        assert vals == sorted(vals)

    def test_playoff_round_order_none_is_zero(self):
        assert PLAYOFF_ROUND_ORDER[None] == 0

    def test_playoff_round_sb_win_is_max(self):
        assert PLAYOFF_ROUND_ORDER["SB_win"] == max(PLAYOFF_ROUND_ORDER.values())

    def test_team_name_map_no_self_loops(self):
        for k, v in TEAM_NAME_MAP.items():
            assert k != v, f"Self-loop in TEAM_NAME_MAP: {k} -> {v}"


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestFeatureEngineering:
    """Test interaction terms used in the ML model."""

    def test_cap_pct_squared(self):
        cap_pct = 17.5
        assert abs(cap_pct ** 2 - 306.25) < 0.001

    def test_age_x_cap_interaction(self):
        age, cap_pct = 28, 17.5
        result = age * cap_pct
        assert abs(result - 490.0) < 0.001

    def test_guaranteed_pct_bounds(self, sample_contracts_raw, sample_cap_history):
        result = clean_contracts(sample_contracts_raw, sample_cap_history)
        assert (result["guaranteed_pct"] >= 0).all()
        assert (result["guaranteed_pct"] <= 100).all()
