# 🏈 NFL QB Championship Window Analyzer

> *Does paying a quarterback max money close the championship window?*  
> A full data science portfolio project: scraping → SQL → statistics → ML → interactive app.

---

## Key Findings

- 📉 **The 21%+ cap tier underperforms the 18–21% tier** — teams paying QBs above 21% of the cap posted a 50% playoff rate vs. 65% for the tier just below, over a 5-year post-signing window
- 🧓 **Age × cap percentage is the strongest predictor** of post-signing team success (RF importance: 0.51) — paying an older QB max money is categorically riskier than paying a young one the same percentage
- 📅 **Windows decay fastest at the top** — the highest-paid QB tier declines to near league-average playoff probability by year 4–5 post-signing; the 18–21% tier holds up longer
- 🎯 **Random Forest reduced RMSE 19%** vs. the predict-mean baseline (2.76 → 2.23), capturing nonlinear interactions that plain linear regression misses

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![SQLite](https://img.shields.io/badge/SQLite-3-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.43-red)
![Anthropic](https://img.shields.io/badge/Anthropic-API-blueviolet)
![pytest](https://img.shields.io/badge/pytest-25%20passing-green)

---

## Installation & Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/nfl-qb-window-analyzer.git
cd nfl-qb-window-analyzer

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 5. Build the database (scrape data or use dev dataset)
python src/ingestion/generate_dev_data.py   # fast dev data
# -- OR --
python src/ingestion/scrape_overthecap.py   # real data (requires internet)
python src/ingestion/scrape_pfr.py

python src/ingestion/build_database.py

# 6. Run analysis
python src/analysis/regression.py
python src/analysis/ml_model.py

# 7. Launch the app
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
nfl_qb_project/
├── app/
│   ├── streamlit_app.py      # Three-page Streamlit application
│   └── ai_layer.py           # Anthropic API integration & comparable finder
├── src/
│   ├── ingestion/
│   │   ├── generate_dev_data.py    # Realistic synthetic dataset (dev)
│   │   ├── scrape_overthecap.py    # Real OTC scraper (requires internet)
│   │   ├── scrape_pfr.py           # Pro Football Reference scraper
│   │   └── build_database.py       # Full cleaning & SQLite pipeline
│   ├── analysis/
│   │   ├── regression.py           # OLS + logistic regression (statsmodels)
│   │   └── ml_model.py             # Random Forest + GridSearchCV
│   └── viz/
│       └── charts.py               # 8 reusable chart functions
├── tests/
│   └── test_pipeline.py            # 25 unit tests (pytest)
├── writeup/
│   └── championship_window.md      # 2,000-word publishable research piece
├── data/
│   ├── raw/                        # CSVs from scrapers (gitignored)
│   └── processed/                  # SQLite DB + model artifacts (gitignored)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Methodology

**Data**: 71 qualifying QB contracts (AAV ≥ 10% of cap), 2000–2024. All values normalized to cap percentage rather than raw dollars to account for 4× cap growth over the era. Team performance tracked for 5 seasons post-signing (263 contract-season observations).

**Statistical Modeling**: Baseline OLS regression establishes a +0.19 win/season coefficient per cap percentage point (p = 0.051). Logistic regression on Super Bowl appearance confirms higher cap % marginally improves odds (OR: 1.09), but age at signing reduces them (OR: 0.93). VIF analysis confirms multicollinearity between cap features warranting the interaction model.

**Machine Learning**: Random Forest with 5-fold GridSearchCV outperforms regression (Test RMSE: 2.23 vs. 2.54). Feature importance analysis reveals the age × cap interaction term (importance: 0.51) dominates raw cap percentage (0.12) — the key finding of the project.

**AI Layer**: Anthropic's Claude synthesizes model predictions and Euclidean-distance historical comparables into plain-language championship window narratives.

---

## Running Tests

```bash
python -m pytest tests/ -v
# 25 passed in 1.76s
```

---

## Data Sources

- **[OverTheCap](https://overthecap.com)** — QB contract history, AAV, guaranteed money, salary cap totals
- **[Pro Football Reference](https://pro-football-reference.com)** — Team win/loss records, playoff results, 2000–2024

---

## Published Writeup

→ [`writeup/championship_window.md`](writeup/championship_window.md) — ready for Substack, Medium, or LinkedIn

---

*Built as a data science portfolio project demonstrating the full stack: data acquisition, SQL modeling, statistical analysis, machine learning, and AI API integration.*
