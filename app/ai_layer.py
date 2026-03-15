"""
AI layer — spec Section 13.
Finds historical comparables and generates narrative analysis via Anthropic API.
"""

import sqlite3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv
from pathlib import Path as _Path
load_dotenv(_Path(__file__).parent.parent / ".env")

DB_PATH  = Path(__file__).parent.parent / "data" / "processed" / "nfl_qb.db"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"

client = Anthropic()


def find_comparables(
    cap_pct: float,
    age: int,
    years: int,
    guaranteed_pct: float,
    n: int = 3,
) -> list[dict]:
    """
    Find n most similar historical contracts by Euclidean distance
    across normalized features — spec Section 13.3.
    """
    conn = sqlite3.connect(DB_PATH)
    contracts = pd.read_sql("""
        SELECT player, team, signing_year, cap_pct, age_at_signing,
               years, guaranteed_pct, cap_tier
        FROM qb_contracts
    """, conn)

    windows = pd.read_sql("""
        SELECT contract_id, player, signing_year,
               AVG(wins) AS avg_wins_5yr,
               MAX(CAST(made_playoffs AS INT)) AS made_playoffs_5yr,
               MAX(CAST(made_superbowl AS INT)) AS made_sb_5yr
        FROM contract_windows
        GROUP BY contract_id, player, signing_year
    """, conn)
    conn.close()

    df = contracts.merge(windows, on=["player", "signing_year"], how="inner")

    # Normalize for distance calculation
    features = ["cap_pct", "age_at_signing", "years", "guaranteed_pct"]
    query    = np.array([cap_pct, age, years, guaranteed_pct])

    feature_std = df[features].std().replace(0, 1)
    scaled_df   = (df[features] - df[features].mean()) / feature_std
    scaled_q    = (query - df[features].mean().values) / feature_std.values

    distances = np.sqrt(((scaled_df.values - scaled_q) ** 2).sum(axis=1))
    df["distance"] = distances
    top = df.nsmallest(n, "distance")

    return top[["player", "team", "signing_year", "cap_pct", "age_at_signing",
                "years", "guaranteed_pct", "avg_wins_5yr", "made_playoffs_5yr",
                "made_sb_5yr", "cap_tier"]].to_dict(orient="records")


def generate_window_analysis(
    qb_name: str,
    cap_pct: float,
    age: int,
    years: int,
    guaranteed_pct: float,
    predicted_wins: float,
    playoff_prob: float,
    comparables: list[dict],
) -> str:
    """
    Call Anthropic API to generate a narrative championship window analysis.
    Spec Section 13.2.
    """
    comp_text = "\n".join([
        f"  - {c['player']} ({c['team']}, {c['signing_year']}): "
        f"{c['cap_pct']:.1f}% cap, age {c['age_at_signing']}, "
        f"avg {c['avg_wins_5yr']:.1f} wins/season, "
        f"{'reached SB' if c['made_sb_5yr'] else 'no SB'} in 5-year window"
        for c in comparables
    ])

    prompt = f"""You are an NFL analytics expert. A user has entered the following QB contract parameters:

QB Name: {qb_name}
Cap Percentage: {cap_pct:.1f}% of salary cap
Age at Signing: {age}
Contract Length: {years} years
Guaranteed Money: {guaranteed_pct:.0f}% of total value

Model Predictions:
- Predicted avg wins/season: {predicted_wins:.1f}
- Estimated playoff probability per season: {playoff_prob:.0f}%

Most Similar Historical Contracts:
{comp_text}

Write a 3-paragraph championship window analysis (150-200 words total). Be specific:
1. What the model predicts and what it means for this team's window
2. What the most relevant historical comparable teaches us
3. Key risk factors or upside scenarios

Be analytical, not a cheerleader. Acknowledge uncertainty. Reference specific comparable QBs by name."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def get_playoff_probability(
    rf_model,
    cap_pct: float,
    age: int,
    years: int,
    guaranteed_pct: float,
) -> float:
    """
    Estimate playoff probability from predicted wins using empirical calibration.
    (Logistic curve fitted from our data.)
    """
    features = np.array([[
        cap_pct, age, years, guaranteed_pct,
        cap_pct ** 2, age * cap_pct
    ]])
    predicted_wins = rf_model.predict(features)[0]

    # Empirical sigmoid: P(playoff) from wins, calibrated to NFL base rate ~37.5%
    log_odds = -4.0 + 0.45 * predicted_wins
    prob = 1 / (1 + np.exp(-log_odds))
    return round(prob * 100, 1)
