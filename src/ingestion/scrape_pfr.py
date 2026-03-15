"""
Scraper for Pro Football Reference team season data (wins, playoff results).
Run locally (requires internet). Outputs: data/raw/team_seasons_raw.csv

Usage:
    python src/ingestion/scrape_pfr.py
"""

import pandas as pd
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2000
END_YEAR = 2024
DELAY_SECONDS = 3  # PFR is more rate-sensitive than OTC

PFR_STANDINGS_URL = "https://www.pro-football-reference.com/years/{year}/"


def scrape_season(year: int) -> pd.DataFrame:
    """Scrape team standings for a single NFL season from PFR."""
    url = PFR_STANDINGS_URL.format(year=year)
    logger.info(f"Fetching season {year}: {url}")

    # pandas.read_html handles PFR's HTML tables cleanly
    tables = pd.read_html(url, attrs={"id": "AFC"}) + pd.read_html(url, attrs={"id": "NFC"})

    frames = []
    for tbl in tables:
        tbl = tbl.copy()
        # Drop division header rows (they repeat the column names)
        tbl = tbl[tbl["Tm"] != "Tm"]
        tbl["season"] = year
        frames.append(tbl)

    df = pd.concat(frames, ignore_index=True)
    time.sleep(DELAY_SECONDS)
    return df


def scrape_all_seasons() -> pd.DataFrame:
    """Scrape all seasons from START_YEAR to END_YEAR."""
    frames = []
    for year in range(START_YEAR, END_YEAR + 1):
        try:
            df = scrape_season(year)
            frames.append(df)
            logger.info(f"  -> {len(df)} teams for {year}")
        except Exception as e:
            logger.warning(f"  -> Failed for {year}: {e}")

    return pd.concat(frames, ignore_index=True)


def main():
    logger.info("=== Scraping Pro Football Reference ===")
    df = scrape_all_seasons()

    out_path = RAW_DIR / "team_seasons_raw.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} rows to {out_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
