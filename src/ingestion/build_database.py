"""
Build the SQLite database from raw CSVs.
Implements the full cleaning pipeline from spec Sections 5 & 6.

Run:
    python src/ingestion/build_database.py

Outputs:
    data/processed/nfl_qb.db
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR   = Path(__file__).parent.parent.parent / "data" / "raw"
PROC_DIR  = Path(__file__).parent.parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH   = PROC_DIR / "nfl_qb.db"

# Ordered encoding for playoff round (spec Section 6.2)
PLAYOFF_ROUND_ORDER = {
    None: 0,
    "WC": 1,
    "DIV": 2,
    "CCG": 3,
    "SB_loss": 4,
    "SB_win": 5,
}

# Historical franchise relocations / abbreviation mapping (spec Section 6.2)
TEAM_NAME_MAP = {
    "OAK": "LV",   # Raiders moved 2020
    "SD":  "LAC",  # Chargers moved 2017
    "STL": "LAR",  # Rams moved 2016
    "WSH": "WAS",
    "MN":  "MIN",
}


# ── Section 6.1: Contract Cleaning ────────────────────────────────────────────

def clean_contracts(raw: pd.DataFrame, cap_history: pd.DataFrame) -> pd.DataFrame:
    """Clean QB contracts and compute cap-normalized metrics."""
    df = raw.copy()

    # Remove duplicates (same player + signing year)
    before = len(df)
    df = df.drop_duplicates(subset=["player", "signing_year"])
    logger.info(f"Removed {before - len(df)} duplicate contract rows")

    # Filter: only signings at or after 2000
    df = df[df["signing_year"] >= 2000].copy()

    # Normalize team abbreviations
    df["team"] = df["team"].replace(TEAM_NAME_MAP)

    # Merge cap totals
    df = df.merge(cap_history.rename(columns={"year": "signing_year"}),
                  on="signing_year", how="left")

    # Compute cap percentage
    df["cap_pct"] = (df["aav_m"] / df["cap_total_m"] * 100).round(2)

    # Filter per spec: cap_pct >= 10% (removes backups and bridge contracts)
    before = len(df)
    df = df[df["cap_pct"] >= 10.0].copy()
    logger.info(f"Filtered to cap_pct >= 10%: {before} -> {len(df)} contracts")

    # Fill missing guaranteed money with 0 (documented assumption per spec)
    df["guaranteed_m"] = df["guaranteed_m"].fillna(0.0)

    # Guaranteed as % of total
    df["guaranteed_pct"] = (df["guaranteed_m"] / df["total_value_m"] * 100).round(1)

    # Cap tier labels
    df["cap_tier"] = pd.cut(
        df["cap_pct"],
        bins=[10, 15, 18, 21, 100],
        labels=["10-15%", "15-18%", "18-21%", "21%+"],
        right=False
    ).astype(str)

    df = df.reset_index(drop=True)
    df.index.name = "contract_id"
    df = df.reset_index()

    logger.info(f"Clean contracts: {len(df)} rows")
    return df


# ── Section 6.2: Team Season Cleaning ─────────────────────────────────────────

def clean_team_seasons(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich team season data."""
    df = raw.copy()

    # Normalize team abbreviations
    df["team"] = df["team"].replace(TEAM_NAME_MAP)

    # Encode playoff exit as ordered int (spec Section 6.2)
    df["playoff_round_num"] = df["playoff_exit"].map(PLAYOFF_ROUND_ORDER).fillna(0).astype(int)

    # Boolean flags
    df["made_playoffs"]   = df["made_playoffs"].astype(bool)
    df["made_superbowl"]  = df["playoff_exit"].isin(["SB_win", "SB_loss"])
    df["won_superbowl"]   = df["playoff_exit"] == "SB_win"

    # Win percentage
    total_games = df["wins"] + df["losses"]
    df["win_pct"] = (df["wins"] / total_games).round(3)

    # Point differential — use wins as proxy (we don't have scored/allowed in dev data)
    # Real scraper provides this; flag for downstream handling
    df["point_diff"] = ((df["wins"] - df["losses"]) * 3.2).round(0).astype(int)

    df = df.drop_duplicates(subset=["team", "season"]).reset_index(drop=True)
    logger.info(f"Clean team seasons: {len(df)} rows")
    return df


# ── Section 6.3: Build contract_windows Table ─────────────────────────────────

def build_contract_windows(
    contracts: pd.DataFrame,
    team_seasons: pd.DataFrame
) -> pd.DataFrame:
    """
    For each QB contract, look up team performance for the 5 seasons
    following the signing year. One row per (contract, season_offset).
    This is the primary analytical table.
    """
    rows = []
    for _, row in contracts.iterrows():
        for offset in range(1, 6):  # seasons 1-5 post-signing
            season = row["signing_year"] + offset
            ts = team_seasons[
                (team_seasons["team"] == row["team"]) &
                (team_seasons["season"] == season)
            ]
            if ts.empty:
                continue
            ts_row = ts.iloc[0]
            rows.append({
                "contract_id":       row["contract_id"],
                "player":            row["player"],
                "team":              row["team"],
                "signing_year":      row["signing_year"],
                "season":            season,
                "seasons_post":      offset,
                "cap_pct":           row["cap_pct"],
                "cap_tier":          row["cap_tier"],
                "age_at_signing":    row["age_at_signing"],
                "years":             row["years"],
                "aav_m":             row["aav_m"],
                "guaranteed_pct":    row["guaranteed_pct"],
                # Team outcomes
                "wins":              ts_row["wins"],
                "win_pct":           ts_row["win_pct"],
                "made_playoffs":     ts_row["made_playoffs"],
                "made_superbowl":    ts_row["made_superbowl"],
                "won_superbowl":     ts_row["won_superbowl"],
                "playoff_round_num": ts_row["playoff_round_num"],
                "point_diff":        ts_row["point_diff"],
            })

    df = pd.DataFrame(rows)
    logger.info(f"Contract windows: {len(df)} rows "
                f"({df['contract_id'].nunique()} contracts)")
    return df


# ── Write to SQLite ────────────────────────────────────────────────────────────

def write_to_db(
    contracts:        pd.DataFrame,
    team_seasons:     pd.DataFrame,
    contract_windows: pd.DataFrame,
    cap_history:      pd.DataFrame,
) -> None:
    """Write all tables to SQLite with proper schema."""
    logger.info(f"Writing to {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    # Table: qb_contracts
    contracts.to_sql("qb_contracts", conn, if_exists="replace", index=False)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_qb_contracts_player
        ON qb_contracts (player, signing_year)
    """)

    # Table: team_seasons
    team_seasons.to_sql("team_seasons", conn, if_exists="replace", index=False)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_team_seasons
        ON team_seasons (team, season)
    """)

    # Table: contract_windows (primary analytical table)
    contract_windows.to_sql("contract_windows", conn, if_exists="replace", index=False)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_cw_contract
        ON contract_windows (contract_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_cw_player
        ON contract_windows (player)
    """)

    # Table: cap_history
    cap_history.to_sql("cap_history", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()
    logger.info("Database written successfully.")


def validate_db() -> None:
    """Run the key SQL queries from spec Section 5.4 to validate schema."""
    conn = sqlite3.connect(DB_PATH)

    # 5.4.1: Average wins by cap percentage tier
    q1 = """
    SELECT
        cap_tier,
        seasons_post,
        ROUND(AVG(wins), 2)          AS avg_wins,
        ROUND(AVG(win_pct) * 100, 1) AS avg_win_pct,
        ROUND(AVG(CAST(made_playoffs AS FLOAT)) * 100, 1) AS playoff_rate,
        COUNT(*)                      AS n
    FROM contract_windows
    GROUP BY cap_tier, seasons_post
    ORDER BY cap_tier, seasons_post
    """
    df1 = pd.read_sql(q1, conn)
    logger.info("\n=== Avg wins by cap tier (first 8 rows) ===\n" + df1.head(8).to_string())

    # 5.4.2: Championship window length by cap tier
    q2 = """
    SELECT
        cap_tier,
        ROUND(AVG(CAST(made_playoffs AS FLOAT)) * 100, 1) AS playoff_rate_5yr,
        ROUND(AVG(CAST(made_superbowl AS FLOAT)) * 100, 1) AS sb_rate_5yr,
        ROUND(AVG(wins), 2)                                 AS avg_wins_5yr,
        COUNT(DISTINCT contract_id)                         AS n_contracts
    FROM contract_windows
    GROUP BY cap_tier
    ORDER BY cap_tier
    """
    df2 = pd.read_sql(q2, conn)
    logger.info("\n=== Championship window by cap tier ===\n" + df2.to_string())

    conn.close()


def main():
    logger.info("=== Building NFL QB Database ===")

    # Load raw
    contracts_raw    = pd.read_csv(RAW_DIR / "qb_contracts_raw.csv")
    cap_history      = pd.read_csv(RAW_DIR / "cap_history_raw.csv")
    team_seasons_raw = pd.read_csv(RAW_DIR / "team_seasons_raw.csv")

    # Clean
    contracts    = clean_contracts(contracts_raw, cap_history)
    team_seasons = clean_team_seasons(team_seasons_raw)
    windows      = build_contract_windows(contracts, team_seasons)

    # Write
    write_to_db(contracts, team_seasons, windows, cap_history)

    # Validate
    validate_db()

    logger.info(f"\nDatabase ready at {DB_PATH}")


if __name__ == "__main__":
    main()
