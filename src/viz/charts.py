"""
Reusable chart functions — spec Section 10.
All charts use consistent styling and annotation standard.
Used in notebooks/EDA and Streamlit app.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

DB_PATH   = Path(__file__).parent.parent.parent / "data" / "processed" / "nfl_qb.db"
PROC_DIR  = Path(__file__).parent.parent.parent / "data" / "processed"
CHART_DIR = PROC_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ── Style config (spec Section 10.1) ──────────────────────────────────────────
PALETTE = {
    "10-15%": "#4878CF",
    "15-18%": "#6ACC65",
    "18-21%": "#D65F5F",
    "21%+":   "#B47CC7",
}
TIER_ORDER = ["10-15%", "15-18%", "18-21%", "21%+"]
SOURCE_LINE = "Sources: OverTheCap, Pro Football Reference"


def _style_axes(ax, title: str, subtitle: str, xlabel: str, ylabel: str) -> None:
    """Apply annotation standard from spec Section 10.3."""
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.figure.text(0.01, -0.03, SOURCE_LINE, fontsize=7, color="gray",
                   transform=ax.transAxes)
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold",
                     loc="left", pad=10)


def load_windows() -> pd.DataFrame:
    """Load the contract_windows table from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM contract_windows", conn)
    conn.close()
    df["cap_tier"] = pd.Categorical(df["cap_tier"], categories=TIER_ORDER, ordered=True)
    return df


def load_contracts() -> pd.DataFrame:
    """Load the qb_contracts table from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM qb_contracts", conn)
    conn.close()
    return df


# ── Chart 1: Wins by cap tier (boxplot) ───────────────────────────────────────

def plot_wins_by_tier(df: pd.DataFrame, save: bool = False) -> plt.Figure:
    """
    Box plots of wins by cap tier, faceted by seasons_post.
    Spec: plot_wins_by_tier(df)
    """
    seasons = sorted(df["seasons_post"].unique())
    fig, axes = plt.subplots(1, len(seasons), figsize=(16, 5), sharey=True)
    fig.suptitle("Team Wins by QB Cap Tier — Each Post-Signing Season",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, -0.02, SOURCE_LINE, fontsize=7, color="gray")

    for ax, season in zip(axes, seasons):
        sub = df[df["seasons_post"] == season]
        order = [t for t in TIER_ORDER if t in sub["cap_tier"].values]
        colors = [PALETTE[t] for t in order]
        bp = ax.boxplot(
            [sub[sub["cap_tier"] == t]["wins"].values for t in order],
            labels=order,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f"Year +{season}", fontsize=10, fontweight="bold")
        ax.set_xlabel("QB Cap %", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        if ax == axes[0]:
            ax.set_ylabel("Regular Season Wins", fontsize=10)
        n_label = "  ".join([f"{t}: n={len(sub[sub['cap_tier']==t])}" for t in order])
        ax.text(0.01, 0.01, n_label, fontsize=6, transform=ax.transAxes, color="gray")

    plt.tight_layout()
    if save:
        fig.savefig(CHART_DIR / "wins_by_tier.png", dpi=150, bbox_inches="tight")
    return fig


# ── Chart 2: Playoff rate by tier (bar + error bars) ──────────────────────────

def plot_playoff_rate_by_tier(df: pd.DataFrame, save: bool = False) -> plt.Figure:
    """Bar chart of 5-year playoff rate by cap tier with error bars."""
    summary = (
        df.groupby("cap_tier")["made_playoffs"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci95"] = summary["se"] * 1.96

    fig, ax = plt.subplots(figsize=(8, 5))
    # Clip CI so lower bar never goes below 0
    yerr_low  = np.minimum(summary["mean"] * 100, summary["ci95"] * 100)
    yerr_high = summary["ci95"] * 100
    bars = ax.bar(
        summary["cap_tier"],
        summary["mean"] * 100,
        yerr=[yerr_low, yerr_high],
        capsize=5,
        color=[PALETTE[t] for t in summary["cap_tier"]],
        alpha=0.8,
        error_kw=dict(elinewidth=1.5, ecolor="#555555"),
    )
    for bar, row in zip(bars, summary.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{row.mean*100:.0f}%\n(n={row.count})",
                ha="center", fontsize=9, fontweight="bold")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, 85)
    _style_axes(ax,
                "Playoff Rate in 5 Seasons Following QB Signing",
                f"All qualifying QB contracts 2000-2024, n={len(df['contract_id'].unique())}",
                "QB Cap % Tier", "Playoff Appearance Rate (%)")
    plt.tight_layout()
    if save:
        fig.savefig(CHART_DIR / "playoff_rate_by_tier.png", dpi=150, bbox_inches="tight")
    return fig


# ── Chart 3: Window decay (playoff rate over time post-signing) ───────────────

def plot_window_decay(df: pd.DataFrame, save: bool = False) -> plt.Figure:
    """Line chart showing playoff rate decline over years post-signing by tier."""
    grouped = (
        df.groupby(["cap_tier", "seasons_post"])["made_playoffs"]
        .agg(["mean", "count"])
        .reset_index()
    )
    # Suppress data points backed by fewer than 3 contracts — not meaningful
    grouped = grouped[grouped["count"] >= 3]

    fig, ax = plt.subplots(figsize=(9, 5))
    for tier in TIER_ORDER:
        sub = grouped[grouped["cap_tier"] == tier]
        if sub.empty:
            continue
        ax.plot(sub["seasons_post"], sub["mean"] * 100,
                marker="o", linewidth=2.5, label=tier, color=PALETTE[tier])

    ax.axhline(37.5, color="gray", linestyle="--", linewidth=1,
               label="League avg playoff rate (37.5%)")
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f"Year +{i}" for i in range(1, 6)])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(title="QB Cap % Tier", fontsize=9)
    _style_axes(ax,
                "Championship Window Decay by QB Cap Tier",
                "Playoff appearance rate in each of 5 seasons following contract signing",
                "Season Post-Signing", "Playoff Rate (%)")
    plt.tight_layout()
    if save:
        fig.savefig(CHART_DIR / "window_decay.png", dpi=150, bbox_inches="tight")
    return fig


# ── Chart 4: Feature importance (horizontal bar) ─────────────────────────────

def plot_feature_importance(importances: dict, save: bool = False) -> plt.Figure:
    """
    Horizontal bar chart of feature importances.
    importances: dict of {feature_name: importance_value}
    """
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1]))
    fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_imp) * 0.4)))
    bars = ax.barh(list(sorted_imp.keys()), list(sorted_imp.values()),
                   color="#4878CF", alpha=0.8)
    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}", va="center", fontsize=8)
    _style_axes(ax,
                "Random Forest Feature Importances — Predicting Team Wins",
                "Higher = more predictive of post-signing team success",
                "Importance (Mean Decrease in Impurity)", "Feature")
    plt.tight_layout()
    if save:
        fig.savefig(CHART_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    return fig


# ── Chart 5: Comparable contracts scatter ────────────────────────────────────

def plot_comparable_contracts(
    qb_name: str,
    df: pd.DataFrame,
    comparables: list[dict],
    predicted_wins: float,
    save: bool = False,
) -> plt.Figure:
    """Scatter of all historical contracts; highlight comparables."""
    contracts = load_contracts()
    five_yr = (
        df.groupby(["player", "signing_year"])["wins"]
        .mean()
        .reset_index()
        .rename(columns={"wins": "avg_wins_5yr"})
    )
    plot_df = contracts.merge(five_yr, on=["player", "signing_year"], how="inner")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(plot_df["cap_pct"], plot_df["avg_wins_5yr"],
               alpha=0.35, color="gray", s=40, label="All contracts")

    comp_names = {c["player"] for c in comparables}
    comp_df = plot_df[plot_df["player"].isin(comp_names)]
    ax.scatter(comp_df["cap_pct"], comp_df["avg_wins_5yr"],
               color="#D65F5F", s=100, zorder=5, label="Historical comparables")
    for _, row in comp_df.iterrows():
        ax.annotate(f"{row['player']} '{str(row['signing_year'])[2:]}",
                    (row["cap_pct"], row["avg_wins_5yr"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=7)

    # Mark prediction
    ax.axhline(predicted_wins, color="#4878CF", linestyle="--", linewidth=1.5,
               label=f"Predicted: {predicted_wins:.1f} wins/season")

    _style_axes(ax,
                f"Comparable Contracts — {qb_name}",
                f"Avg wins in 5 seasons following contract signing",
                "QB Cap % of Salary Cap", "Avg Wins per Season (Years 1-5)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save:
        fig.savefig(CHART_DIR / f"comparable_{qb_name.replace(' ','_')}.png",
                    dpi=150, bbox_inches="tight")
    return fig


# ── Chart 6: Cap % distribution ───────────────────────────────────────────────

def plot_cap_distribution(contracts: pd.DataFrame, save: bool = False) -> plt.Figure:
    """Distribution of QB cap percentages over the dataset."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(contracts["cap_pct"], bins=20, color="#4878CF", alpha=0.75, edgecolor="white")
    for pct, label in [(15, "15%"), (18, "18%"), (21, "21%")]:
        ax.axvline(pct, color="gray", linestyle="--", linewidth=1)
        ax.text(pct + 0.2, ax.get_ylim()[1] * 0.9, label, fontsize=8, color="gray")
    _style_axes(ax,
                "Distribution of QB Cap Percentages at Signing (2000-2024)",
                f"n={len(contracts)} qualifying contracts (cap_pct ≥ 10%)",
                "Cap % of Total Salary Cap", "Number of Contracts")
    plt.tight_layout()
    if save:
        fig.savefig(CHART_DIR / "cap_distribution.png", dpi=150, bbox_inches="tight")
    return fig


# ── Chart 7: Cap % vs avg wins scatter ────────────────────────────────────────

def plot_cap_vs_wins(df: pd.DataFrame, save: bool = False) -> plt.Figure:
    """Scatter plot of cap_pct vs avg wins with regression line."""
    agg = (
        df.groupby(["contract_id", "cap_pct", "player", "signing_year"])["wins"]
        .mean()
        .reset_index()
        .rename(columns={"wins": "avg_wins"})
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(agg["cap_pct"], agg["avg_wins"], alpha=0.5, color="#4878CF", s=50)

    # Regression line
    m, b = np.polyfit(agg["cap_pct"], agg["avg_wins"], 1)
    xs = np.linspace(agg["cap_pct"].min(), agg["cap_pct"].max(), 100)
    ax.plot(xs, m * xs + b, color="#D65F5F", linewidth=2,
            label=f"OLS fit (slope={m:.2f} wins per 1%)")

    # Annotate key outliers
    for _, row in agg.iterrows():
        if row["avg_wins"] > 12.5 or row["avg_wins"] < 4.5:
            ax.annotate(f"{row['player']} '{str(int(row['signing_year']))[2:]}",
                        (row["cap_pct"], row["avg_wins"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=6.5)

    ax.legend(fontsize=9)
    _style_axes(ax,
                "QB Cap % vs. Average Wins in Post-Signing Window",
                f"Each dot = one QB contract; avg wins across 5 seasons post-signing",
                "QB Cap % of Salary Cap", "Avg Wins per Season (Years 1-5)")
    plt.tight_layout()
    if save:
        fig.savefig(CHART_DIR / "cap_vs_wins.png", dpi=150, bbox_inches="tight")
    return fig


# ── Chart 8: SB rate by tier ──────────────────────────────────────────────────

def plot_sb_rate_by_tier(df: pd.DataFrame, save: bool = False) -> plt.Figure:
    """Super Bowl appearance rate within 5 years by cap tier."""
    summary = (
        df.groupby("cap_tier")["made_superbowl"]
        .agg(["mean", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        summary["cap_tier"],
        summary["mean"] * 100,
        color=[PALETTE[t] for t in summary["cap_tier"]],
        alpha=0.8,
    )
    for bar, row in zip(bars, summary.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{row.mean*100:.0f}%",
                ha="center", fontsize=10, fontweight="bold")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, 35)
    _style_axes(ax,
                "Super Bowl Appearance Rate Within 5 Years of QB Contract",
                f"All qualifying contracts 2000-2024",
                "QB Cap % Tier", "Super Bowl Appearance Rate (%)")
    plt.tight_layout()
    if save:
        fig.savefig(CHART_DIR / "sb_rate_by_tier.png", dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    # Generate and save all charts
    print("Generating all charts...")
    df = load_windows()
    contracts = load_contracts()

    plot_wins_by_tier(df, save=True)
    print("  wins_by_tier.png")
    plot_playoff_rate_by_tier(df, save=True)
    print("  playoff_rate_by_tier.png")
    plot_window_decay(df, save=True)
    print("  window_decay.png")
    plot_cap_distribution(contracts, save=True)
    print("  cap_distribution.png")
    plot_cap_vs_wins(df, save=True)
    print("  cap_vs_wins.png")
    plot_sb_rate_by_tier(df, save=True)
    print("  sb_rate_by_tier.png")
    print(f"Charts saved to {CHART_DIR}")
