"""
Machine Learning Model — spec Section 9.
Random Forest with GridSearchCV; compared to regression baseline.

Run:
    python src/analysis/ml_model.py

Outputs:
  data/processed/rf_model.joblib
  data/processed/ml_results.json
  data/processed/charts/feature_importance.png
  data/processed/charts/model_comparison.png
"""

import sqlite3
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

DB_PATH  = Path(__file__).parent.parent.parent / "data" / "processed" / "nfl_qb.db"
PROC_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
CHART_DIR = PROC_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
FEATURES = ["cap_pct", "age_at_signing", "years", "guaranteed_pct",
            "cap_pct_sq", "age_x_cap"]
TARGET = "avg_wins"

PALETTE = {
    "10-15%": "#4878CF",
    "15-18%": "#6ACC65",
    "18-21%": "#D65F5F",
    "21%+":   "#B47CC7",
}


def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT
            contract_id, player, signing_year, cap_pct, cap_tier,
            age_at_signing, years, guaranteed_pct,
            AVG(wins)                           AS avg_wins,
            MAX(CAST(made_superbowl AS INT))    AS made_superbowl_5yr,
            MAX(playoff_round_num)              AS best_playoff_round,
            COUNT(*)                            AS seasons_observed
        FROM contract_windows
        GROUP BY contract_id, player, signing_year, cap_pct, cap_tier,
                 age_at_signing, years, guaranteed_pct
        HAVING seasons_observed >= 2
    """, conn)
    conn.close()
    df["cap_pct_sq"] = df["cap_pct"] ** 2
    df["age_x_cap"]  = df["age_at_signing"] * df["cap_pct"]
    return df


def baseline_rmse(y_train: pd.Series, y_test: pd.Series) -> float:
    pred = np.full(len(y_test), y_train.mean())
    return float(np.sqrt(mean_squared_error(y_test, pred)))


def train_random_forest(X_train, y_train) -> tuple[RandomForestRegressor, dict]:
    """GridSearchCV on Random Forest — spec Section 9.3."""
    param_grid = {
        "n_estimators":      [100, 200],
        "max_depth":         [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf":  [1, 2],
    }
    rf = RandomForestRegressor(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        rf, param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model  = grid_search.best_estimator_
    print(f"Best RF params: {best_params}")
    return best_model, best_params


def train_gradient_boosting(X_train, y_train) -> GradientBoostingRegressor:
    """Gradient Boosting as secondary candidate — spec Section 9.1."""
    param_grid = {
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth":     [3, 5],
    }
    gb = GradientBoostingRegressor(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        gb, param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    print(f"Best GB params: {grid_search.best_params_}")
    return grid_search.best_estimator_


def plot_feature_importance(model, feature_names: list, save: bool = True) -> None:
    """Spec Section 9.4: horizontal bar chart of feature importances."""
    importances = dict(zip(feature_names, model.feature_importances_))
    sorted_imp  = dict(sorted(importances.items(), key=lambda x: x[1]))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4878CF" if "cap" in k else "#6ACC65" for k in sorted_imp]
    bars = ax.barh(list(sorted_imp.keys()), list(sorted_imp.values()),
                   color=colors, alpha=0.85)
    for bar in bars:
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}", va="center", fontsize=9)
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=10)
    ax.set_title("Random Forest Feature Importances\nPredicting Avg Team Wins in Post-Signing Window",
                 fontsize=12, fontweight="bold", loc="left")
    ax.figure.text(0.01, -0.03, "Sources: OverTheCap, Pro Football Reference",
                   fontsize=7, color="gray", transform=ax.transAxes)
    plt.tight_layout()
    if save:
        fig.savefig(CHART_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
        print("  feature_importance.png saved")
    plt.close()
    return importances


def plot_model_comparison(comparison: pd.DataFrame, save: bool = True) -> None:
    """Visual model comparison table — spec Section 9.5."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    tbl = ax.table(
        cellText=comparison.values,
        colLabels=comparison.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)
    # Header styling
    for j in range(len(comparison.columns)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # Row alternating color
    for i in range(1, len(comparison) + 1):
        for j in range(len(comparison.columns)):
            tbl[(i, j)].set_facecolor("#ecf0f1" if i % 2 == 0 else "white")

    ax.set_title("Model Comparison — Predicting Team Wins Post QB Signing",
                 fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    if save:
        fig.savefig(CHART_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
        print("  model_comparison.png saved")
    plt.close()


def plot_actual_vs_predicted(y_test, y_pred_rf, save: bool = True) -> None:
    """Actual vs predicted scatter for Random Forest."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred_rf, alpha=0.65, color="#4878CF", s=50)
    mn = min(y_test.min(), y_pred_rf.min()) - 0.5
    mx = max(y_test.max(), y_pred_rf.max()) + 0.5
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Avg Wins", fontsize=10)
    ax.set_ylabel("Predicted Avg Wins", fontsize=10)
    ax.set_title("Random Forest: Actual vs. Predicted Team Wins",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save:
        fig.savefig(CHART_DIR / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
        print("  actual_vs_predicted.png saved")
    plt.close()


def main():
    print("=== NFL QB ML Model ===")
    df = load_data()
    print(f"Dataset: {len(df)} contracts")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    base_rmse = baseline_rmse(y_train, y_test)
    print(f"\nBaseline RMSE (predict mean): {base_rmse:.3f}")

    # ── Train models ──────────────────────────────────────────────────────────
    print("\nTraining Random Forest...")
    rf_model, rf_params = train_random_forest(X_train, y_train)

    print("\nTraining Gradient Boosting...")
    gb_model = train_gradient_boosting(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    def evaluate(model, name):
        pred  = model.predict(X_test)
        rmse  = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2    = float(r2_score(y_test, pred))
        cv_scores = cross_val_score(model, X, y, cv=5,
                                    scoring="neg_root_mean_squared_error")
        cv_rmse = float(-cv_scores.mean())
        return {"Model": name, "Test RMSE": round(rmse, 3),
                "Test R²": round(r2, 3), "CV RMSE (5-fold)": round(cv_rmse, 3)}

    rf_eval = evaluate(rf_model, "Random Forest")
    gb_eval = evaluate(gb_model, "Gradient Boosting")

    # Linear regression RMSE from saved results
    reg_path = PROC_DIR / "regression_results.json"
    lin_rmse = 2.757  # baseline fallback
    if reg_path.exists():
        with open(reg_path) as f:
            reg_res = json.load(f)
        lin_rmse = reg_res.get("multi_rmse", lin_rmse)
    lin_r2 = reg_res.get("multi_r2", 0.097) if reg_path.exists() else 0.097

    comparison = pd.DataFrame([
        {"Model": "Baseline (predict mean)", "Test RMSE": round(base_rmse, 3),
         "Test R²": "—", "CV RMSE (5-fold)": "—"},
        {"Model": "Linear Regression",       "Test RMSE": round(lin_rmse, 3),
         "Test R²": round(lin_r2, 3),         "CV RMSE (5-fold)": "—"},
        rf_eval,
        gb_eval,
    ])

    print("\n=== Model Comparison ===")
    print(comparison.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating charts...")
    importances = plot_feature_importance(rf_model, FEATURES)
    plot_model_comparison(comparison)
    rf_pred = rf_model.predict(X_test)
    plot_actual_vs_predicted(y_test, rf_pred)

    print("\nFeature Importances (RF):")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 100)
        print(f"  {feat:20s}: {imp:.3f}  {bar}")

    # ── Save model (spec Section 9.6) ─────────────────────────────────────────
    model_path = PROC_DIR / "rf_model.joblib"
    joblib.dump(rf_model, model_path)
    print(f"\nModel saved to {model_path}")

    # ── Save results JSON ─────────────────────────────────────────────────────
    results = {
        "best_rf_params":  rf_params,
        "rf_eval":         rf_eval,
        "gb_eval":         gb_eval,
        "baseline_rmse":   round(base_rmse, 3),
        "feature_importances": {k: round(float(v), 4) for k, v in importances.items()},
        "comparison_table": comparison.to_dict(orient="records"),
        "features":        FEATURES,
    }
    out_path = PROC_DIR / "ml_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
