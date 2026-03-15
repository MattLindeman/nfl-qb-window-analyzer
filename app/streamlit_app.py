"""
NFL QB Championship Window Analyzer — Streamlit App
Spec Sections 12 & 13.

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import sqlite3
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.viz.charts import (
    load_windows, load_contracts, plot_wins_by_tier,
    plot_playoff_rate_by_tier, plot_window_decay,
    plot_cap_distribution, plot_cap_vs_wins, plot_sb_rate_by_tier,
    DB_PATH, PROC_DIR,
)

# ── App config (spec Section 12.5) ────────────────────────────────────────────
st.set_page_config(
    page_title="NFL QB Championship Window Analyzer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_PATH = PROC_DIR / "rf_model.joblib"
ML_RESULTS = PROC_DIR / "ml_results.json"
REG_RESULTS = PROC_DIR / "regression_results.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_db_tables():
    windows   = load_windows()
    contracts = load_contracts()
    return windows, contracts


@st.cache_data
def load_results():
    ml = json.loads(ML_RESULTS.read_text()) if ML_RESULTS.exists() else {}
    rg = json.loads(REG_RESULTS.read_text()) if REG_RESULTS.exists() else {}
    return ml, rg


def predict_wins(model, cap_pct, age, years, guaranteed_pct):
    feats = np.array([[
        cap_pct, age, years, guaranteed_pct,
        cap_pct ** 2, age * cap_pct
    ]])
    return float(model.predict(feats)[0])


def wins_to_playoff_prob(predicted_wins):
    log_odds = -4.0 + 0.45 * predicted_wins
    return round(100 / (1 + np.exp(-log_odds)), 1)


def find_comparables(cap_pct, age, years, guaranteed_pct, n=3):
    conn = sqlite3.connect(DB_PATH)
    contracts = pd.read_sql("SELECT * FROM qb_contracts", conn)
    windows   = pd.read_sql("""
        SELECT player, signing_year,
               AVG(wins) AS avg_wins_5yr,
               MAX(CAST(made_playoffs AS INT)) AS made_playoffs_5yr,
               MAX(CAST(made_superbowl AS INT)) AS made_sb_5yr
        FROM contract_windows GROUP BY player, signing_year
    """, conn)
    conn.close()

    df = contracts.merge(windows, on=["player", "signing_year"], how="inner")
    features = ["cap_pct", "age_at_signing", "years", "guaranteed_pct"]
    query = np.array([cap_pct, age, years, guaranteed_pct])

    std = df[features].std().replace(0, 1)
    mean = df[features].mean()
    scaled = (df[features] - mean) / std
    sq = (query - mean.values) / std.values
    df["distance"] = np.sqrt(((scaled.values - sq) ** 2).sum(axis=1))
    return df.nsmallest(n, "distance")


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #2c3e50;
    margin-bottom: 12px;
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #2c3e50; }
  .metric-label { font-size: 0.85rem; color: #7f8c8d; text-transform: uppercase; }
  .finding-box {
    background: #eaf4fb;
    border-radius: 8px;
    padding: 12px 16px;
    border-left: 4px solid #3498db;
    margin-bottom: 10px;
  }
  .ai-box {
    background: #fdfefe;
    border: 1px solid #bdc3c7;
    border-radius: 10px;
    padding: 20px;
    font-size: 0.95rem;
    line-height: 1.65;
  }
  .comp-card {
    background: #fff9f0;
    border-radius: 8px;
    padding: 12px;
    border-left: 3px solid #e67e22;
    margin-bottom: 8px;
  }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/a/a2/National_Football_League_logo.svg",
                 width=80)
st.sidebar.title("🏈 QB Window Analyzer")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Research Dashboard", "🔍 QB Contract Lookup", "🔮 Window Predictor"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Data: OverTheCap, Pro Football Reference\nEra: 2000–2024")

df, contracts = load_db_tables()
ml_results, reg_results = load_results()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Research Dashboard
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Research Dashboard":
    st.title("Does Paying a QB Max Money Close the Championship Window?")
    st.markdown(
        "An empirical analysis of every qualifying NFL QB contract from 2000–2024. "
        "We examine team performance in the 5 seasons following each signing."
    )

    # Key stat callouts (spec Section 12.2)
    col1, col2, col3, col4 = st.columns(4)

    tier_summary = (
        df.groupby("cap_tier")
        .agg(
            playoff_rate=("made_playoffs", "mean"),
            sb_rate=("made_superbowl", "mean"),
            avg_wins=("wins", "mean"),
            n_contracts=("contract_id", "nunique"),
        )
        .reset_index()
    )

    top_tier  = tier_summary[tier_summary["cap_tier"] == "21%+"].iloc[0]
    best_tier = tier_summary[tier_summary["cap_tier"] == "18-21%"].iloc[0]
    total_contracts = contracts["contract_id"].nunique()

    with col1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{top_tier['avg_wins']:.1f}</div>
          <div class="metric-label">Avg wins/season — 21%+ cap QB</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{top_tier['playoff_rate']*100:.0f}%</div>
          <div class="metric-label">Playoff rate — 21%+ cap QB (5yr)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{best_tier['playoff_rate']*100:.0f}%</div>
          <div class="metric-label">Playoff rate — 18–21% cap tier (5yr)</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{total_contracts}</div>
          <div class="metric-label">Qualifying QB contracts analyzed</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Charts
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Playoff Rate by Cap Tier")
        fig = plot_playoff_rate_by_tier(df)
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Championship Window Decay")
        fig = plot_window_decay(df)
        st.pyplot(fig)
        plt.close()

    st.subheader("Cap % vs. Average Wins (Each Contract)")
    fig = plot_cap_vs_wins(df)
    st.pyplot(fig)
    plt.close()

    # Model comparison table (spec Section 12.2)
    if ml_results.get("comparison_table"):
        st.markdown("---")
        st.subheader("📐 Model Comparison")
        comp_df = pd.DataFrame(ml_results["comparison_table"])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        if ml_results.get("feature_importances"):
            st.subheader("🌲 Random Forest Feature Importances")
            imp = ml_results["feature_importances"]
            imp_df = pd.DataFrame(
                sorted(imp.items(), key=lambda x: x[1], reverse=True),
                columns=["Feature", "Importance"]
            )
            st.bar_chart(imp_df.set_index("Feature"))

    # Key findings
    st.markdown("---")
    st.subheader("🔑 Key Findings")
    findings = [
        "The **18–21% cap tier** posts the highest playoff rate among all tiers — "
        "higher than even the top-paid QBs. Peak value may lie just below max-contract territory.",
        "The **age × cap interaction** is the single most important predictor of team success "
        "post-signing (RF importance: 0.51). Paying an older QB premium dollars carries "
        "disproportionate risk.",
        "The linear regression coefficient on cap_pct is **+0.19 wins per 1% increase**, "
        "but this effect borders on statistical significance — cap % alone explains very "
        "little of team variance.",
        "Championship windows **decay fastest** for the highest-paid QBs: playoff rate in "
        "Year 5 post-signing falls to near-league-average levels for the 21%+ tier.",
    ]
    for f in findings:
        st.markdown(f'<div class="finding-box">🏈 {f}</div>', unsafe_allow_html=True)

    if reg_results.get("interpretation"):
        st.markdown("---")
        st.subheader("📝 Statistical Interpretation")
        st.info(reg_results["interpretation"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: QB Contract Lookup
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 QB Contract Lookup":
    st.title("🔍 QB Contract Lookup")
    st.markdown("Search any qualifying QB signing from 2000–2024.")

    search = st.text_input("Search QB name", placeholder="e.g. Mahomes, Brady, Rodgers...")

    if search:
        mask = contracts["player"].str.contains(search, case=False, na=False)
        results = contracts[mask].sort_values("signing_year", ascending=False)
    else:
        results = contracts.sort_values("signing_year", ascending=False)

    if results.empty:
        st.warning(f"No contracts found for '{search}'")
    else:
        # Show contracts table
        display_cols = ["player", "team", "signing_year", "years", "aav_m",
                        "cap_pct", "cap_tier", "age_at_signing"]
        st.dataframe(
            results[display_cols].rename(columns={
                "player": "Player", "team": "Team",
                "signing_year": "Year", "years": "Length (yrs)",
                "aav_m": "AAV ($M)", "cap_pct": "Cap %",
                "cap_tier": "Tier", "age_at_signing": "Age"
            }),
            use_container_width=True, hide_index=True
        )

        # Detail view for a selected contract
        st.markdown("---")
        st.subheader("Contract Detail")
        player_options = results["player"].unique().tolist()
        selected_player = st.selectbox("Select player for detail view", player_options)
        year_options = results[results["player"] == selected_player]["signing_year"].tolist()
        selected_year = st.selectbox("Signing year", year_options)

        contract = contracts[
            (contracts["player"] == selected_player) &
            (contracts["signing_year"] == selected_year)
        ].iloc[0]

        window_data = df[
            (df["player"] == selected_player) &
            (df["signing_year"] == selected_year)
        ].sort_values("seasons_post")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cap %",       f"{contract['cap_pct']:.1f}%")
            st.metric("AAV",         f"${contract['aav_m']:.1f}M")
            st.metric("Guaranteed",  f"${contract['guaranteed_m']:.1f}M")
        with col2:
            st.metric("Age at Signing", contract["age_at_signing"])
            st.metric("Length",         f"{contract['years']} years")
            st.metric("Cap Tier",       contract["cap_tier"])
        with col3:
            if not window_data.empty:
                st.metric("Avg Wins (5yr)",    f"{window_data['wins'].mean():.1f}")
                st.metric("Playoff Rate",      f"{window_data['made_playoffs'].mean()*100:.0f}%")
                st.metric("SB Appearances",    int(window_data["made_superbowl"].sum()))

        if not window_data.empty:
            st.markdown("**Season-by-Season Performance Post-Signing:**")
            perf_df = window_data[["seasons_post", "season", "wins", "made_playoffs",
                                   "playoff_round_num"]].rename(columns={
                "seasons_post": "Yr Post",
                "season": "Season",
                "wins": "Wins",
                "made_playoffs": "Playoffs",
                "playoff_round_num": "Playoff Round",
            })
            st.dataframe(perf_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Window Predictor
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Window Predictor":
    st.title("🔮 Championship Window Predictor")
    st.markdown(
        "Enter a QB's contract parameters to get a model-driven projection "
        "and AI-generated analysis comparing to historical contracts."
    )

    col_input, col_output = st.columns([1, 1.4])

    with col_input:
        st.subheader("Contract Parameters")
        qb_name       = st.text_input("QB Name", value="Patrick Mahomes")
        cap_pct       = st.slider("Cap Percentage (%)", 10.0, 30.0, 17.5, 0.5)
        age           = st.slider("Age at Signing", 24, 40, 27)
        years         = st.slider("Contract Length (years)", 1, 10, 4)
        guaranteed_pct = st.slider("Guaranteed Money (% of total)", 20, 100, 65)

        analyze = st.button("🔍 Analyze Championship Window", type="primary",
                             use_container_width=True)

    with col_output:
        if analyze:
            model = load_model()
            predicted_wins = predict_wins(model, cap_pct, age, years, guaranteed_pct)
            playoff_prob   = wins_to_playoff_prob(predicted_wins)
            comparables    = find_comparables(cap_pct, age, years, guaranteed_pct)

            st.subheader(f"Projection: {qb_name}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Avg Wins", f"{predicted_wins:.1f}")
            m2.metric("Est. Playoff Prob.", f"{playoff_prob:.0f}%")

            cap_tier = "21%+" if cap_pct >= 21 else \
                       "18-21%" if cap_pct >= 18 else \
                       "15-18%" if cap_pct >= 15 else "10-15%"
            m3.metric("Cap Tier", cap_tier)

            st.markdown("---")
            st.subheader("📋 Historical Comparables")
            for _, row in comparables.iterrows():
                sb_str = "✅ Reached SB" if row["made_sb_5yr"] else "❌ No SB"
                po_str = "✅" if row["made_playoffs_5yr"] else "❌"
                st.markdown(f"""
                <div class="comp-card">
                  <b>{row['player']}</b> — {row['team']}, {int(row['signing_year'])}<br>
                  Cap: {row['cap_pct']:.1f}% | Age: {int(row['age_at_signing'])} |
                  {int(row['years'])}yr | {row['guaranteed_pct']:.0f}% guaranteed<br>
                  Avg wins: <b>{row['avg_wins_5yr']:.1f}/season</b> |
                  Playoffs: {po_str} | {sb_str}
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("🤖 AI Analysis")
            with st.spinner("Generating analysis..."):
                try:
                    from app.ai_layer import generate_window_analysis
                    analysis = generate_window_analysis(
                        qb_name=qb_name,
                        cap_pct=cap_pct,
                        age=age,
                        years=years,
                        guaranteed_pct=guaranteed_pct,
                        predicted_wins=predicted_wins,
                        playoff_prob=playoff_prob,
                        comparables=comparables.to_dict(orient="records"),
                    )
                    st.markdown(f'<div class="ai-box">{analysis}</div>',
                                unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"AI analysis unavailable: {e}. "
                               f"Set ANTHROPIC_API_KEY to enable.")
                    st.info(
                        f"**Model says:** {qb_name} at {cap_pct:.1f}% of the cap "
                        f"(age {age}, {years}yr deal) projects to **{predicted_wins:.1f} wins/season** "
                        f"with a **{playoff_prob:.0f}% playoff probability** per year. "
                        f"The most comparable historical contract is "
                        f"**{comparables.iloc[0]['player']} ({int(comparables.iloc[0]['signing_year'])})**, "
                        f"who averaged {comparables.iloc[0]['avg_wins_5yr']:.1f} wins/season "
                        f"over their window."
                    )
        else:
            st.info("👈 Enter parameters and click **Analyze** to run the projection.")
            st.markdown("""
            **How it works:**
            1. Parameters are passed to a Random Forest model trained on 57 historical QB contracts
            2. Historical comparables are found by Euclidean distance across normalized contract features
            3. Claude synthesizes the prediction and comparables into a narrative analysis
            """)
