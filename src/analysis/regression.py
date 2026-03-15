"""
Statistical modeling — spec Section 8.
Linear regression on avg wins; logistic regression on SB appearance.

Run:
    python src/analysis/regression.py

Outputs:
  data/processed/regression_results.json
"""

import sqlite3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

DB_PATH   = Path(__file__).parent.parent.parent / "data" / "processed" / "nfl_qb.db"
PROC_DIR  = Path(__file__).parent.parent.parent / "data" / "processed"
CHART_DIR = PROC_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
FEATURES = ["cap_pct", "age_at_signing", "years", "guaranteed_pct"]
FEATURES_EXTENDED = ["cap_pct", "age_at_signing", "years", "guaranteed_pct",
                     "cap_pct_sq", "age_x_cap"]
TARGET_WINS   = "avg_wins"
TARGET_SB     = "made_superbowl_5yr"


def load_modeling_data() -> pd.DataFrame:
    """
    Aggregate contract_windows to one row per contract.
    Target: avg wins over 5-year window.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT
            contract_id,
            player,
            signing_year,
            cap_pct,
            age_at_signing,
            years,
            guaranteed_pct,
            AVG(wins)                               AS avg_wins,
            AVG(win_pct)                            AS avg_win_pct,
            MAX(CAST(made_playoffs AS INT))         AS made_playoffs_5yr,
            MAX(CAST(made_superbowl AS INT))        AS made_superbowl_5yr,
            MAX(CAST(won_superbowl AS INT))         AS won_superbowl_5yr,
            MAX(playoff_round_num)                  AS best_playoff_round,
            COUNT(*)                                AS seasons_observed
        FROM contract_windows
        GROUP BY contract_id, player, signing_year, cap_pct,
                 age_at_signing, years, guaranteed_pct
        HAVING seasons_observed >= 2
    """, conn)
    conn.close()

    # Interaction terms (spec Section 8.3)
    df["cap_pct_sq"]   = df["cap_pct"] ** 2
    df["age_x_cap"]    = df["age_at_signing"] * df["cap_pct"]

    return df


def print_separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def run_linear_regression(df: pd.DataFrame) -> dict:
    """
    Full linear regression pipeline per spec Section 8.3.
    Returns dict of key results.
    """
    results = {}
    X = df[FEATURES].copy()
    y = df[TARGET_WINS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Baseline: predict mean
    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    results["baseline_rmse"] = round(baseline_rmse, 3)

    print_separator("Baseline Model")
    print(f"  Predict mean ({y_train.mean():.2f} wins): RMSE = {baseline_rmse:.3f}")

    # ── Simple regression: cap_pct only ──────────────────────────────────────
    print_separator("Simple Regression: cap_pct only")
    X_simple = X_train[["cap_pct"]]
    model_simple = sm.OLS(y_train, sm.add_constant(X_simple)).fit()
    print(model_simple.summary())

    coef = model_simple.params["cap_pct"]
    pval = model_simple.pvalues["cap_pct"]
    results["simple_coef_cap_pct"] = round(float(coef), 4)
    results["simple_pval_cap_pct"] = round(float(pval), 4)
    results["simple_r2"]           = round(float(model_simple.rsquared), 4)

    # ── Multiple regression ───────────────────────────────────────────────────
    print_separator("Multiple Regression: all features")
    X_multi = sm.add_constant(X_train[FEATURES])
    model_multi = sm.OLS(y_train, X_multi).fit()
    print(model_multi.summary())

    # VIF
    vif_data = pd.DataFrame()
    vif_data["feature"] = FEATURES
    vif_data["VIF"] = [
        variance_inflation_factor(X_train[FEATURES].values, i)
        for i in range(len(FEATURES))
    ]
    print("\nVIF Scores:")
    print(vif_data.to_string(index=False))
    results["vif"] = vif_data.set_index("feature")["VIF"].round(2).to_dict()

    test_pred_multi = model_multi.predict(sm.add_constant(X_test[FEATURES]))
    rmse_multi = np.sqrt(mean_squared_error(y_test, test_pred_multi))
    results["multi_rmse"]  = round(rmse_multi, 3)
    results["multi_r2"]    = round(float(model_multi.rsquared), 4)
    results["multi_adj_r2"] = round(float(model_multi.rsquared_adj), 4)

    # Coefficient table (spec Section 8.4)
    coef_table = pd.DataFrame({
        "feature":    model_multi.params.index,
        "coefficient": model_multi.params.values.round(4),
        "std_error":   model_multi.bse.values.round(4),
        "p_value":     model_multi.pvalues.values.round(4),
        "ci_lower":    model_multi.conf_int()[0].values.round(4),
        "ci_upper":    model_multi.conf_int()[1].values.round(4),
    })
    results["coef_table"] = coef_table.to_dict(orient="records")

    print("\nCoefficient Table:")
    print(coef_table.to_string(index=False))

    # ── Interaction model ─────────────────────────────────────────────────────
    print_separator("Interaction Model: cap_pct_sq + age_x_cap")
    X_int = sm.add_constant(df[FEATURES_EXTENDED])
    model_int = sm.OLS(y, X_int).fit()
    print(model_int.summary())
    results["interaction_r2"]     = round(float(model_int.rsquared), 4)
    results["interaction_adj_r2"] = round(float(model_int.rsquared_adj), 4)

    # ── Residual plots ────────────────────────────────────────────────────────
    fitted = model_multi.fittedvalues
    resid  = model_multi.resid

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residual vs fitted
    axes[0].scatter(fitted, resid, alpha=0.6, color="#4878CF", s=40)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs. Fitted", fontweight="bold")

    # QQ plot
    sm.qqplot(resid, line="s", ax=axes[1], alpha=0.6)
    axes[1].set_title("Q-Q Plot of Residuals", fontweight="bold")

    plt.tight_layout()
    fig.savefig(CHART_DIR / "regression_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()

    results["model_multi"] = model_multi  # return for Streamlit use
    return results


def run_logistic_regression(df: pd.DataFrame) -> dict:
    """
    Logistic regression: predict SB appearance within 5 years.
    Reports odds ratios — spec Section 8.5.
    """
    print_separator("Logistic Regression: Super Bowl Appearance (5yr)")

    X = df[FEATURES]
    y = df[TARGET_SB]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=500)
    model.fit(X_scaled, y)

    # Odds ratios from coefficients
    odds_ratios = np.exp(model.coef_[0])
    results = {
        "features":    FEATURES,
        "odds_ratios": {f: round(float(o), 3)
                        for f, o in zip(FEATURES, odds_ratios)},
        "roc_auc":     round(roc_auc_score(y, model.predict_proba(X_scaled)[:, 1]), 3),
    }

    print(f"\nAUC: {results['roc_auc']}")
    print("\nOdds Ratios:")
    for feat, or_ in results["odds_ratios"].items():
        direction = "↑" if or_ > 1 else "↓"
        print(f"  {feat:20s}: {or_:.3f} {direction}")

    return results


def interpret_results(linear_results: dict) -> str:
    """
    Generate the written interpretation required by spec Section 8.4.
    """
    cap_coef = linear_results.get("simple_coef_cap_pct", 0)
    cap_pval = linear_results.get("simple_pval_cap_pct", 1)
    sig      = "statistically significant at p < 0.05" if cap_pval < 0.05 else "not statistically significant at p < 0.05"

    return (
        f"A 1 percentage point increase in QB cap share is associated with "
        f"{cap_coef:+.2f} wins per season, holding all else constant, and this "
        f"effect is {sig} (p={cap_pval:.3f}). "
        f"The multiple regression model explains {linear_results.get('multi_adj_r2', 0)*100:.1f}% "
        f"of variance in post-signing team wins (adjusted R²)."
    )


def main():
    df = load_modeling_data()
    print(f"Modeling dataset: {len(df)} contracts")
    print(df[FEATURES + [TARGET_WINS, TARGET_SB]].describe().round(2))

    linear_results  = run_linear_regression(df)
    logistic_results = run_logistic_regression(df)

    interpretation = interpret_results(linear_results)
    print_separator("Written Interpretation (spec Section 8.4)")
    print(interpretation)

    # Save results (excluding non-serializable model object)
    save_results = {k: v for k, v in linear_results.items() if k != "model_multi"}
    save_results["logistic"] = logistic_results
    save_results["interpretation"] = interpretation

    out_path = PROC_DIR / "regression_results.json"
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
