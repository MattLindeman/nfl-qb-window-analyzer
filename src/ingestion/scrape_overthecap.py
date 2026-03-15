"""
Scraper for OverTheCap QB contract data.
Run locally (requires internet). Outputs: data/raw/qb_contracts_raw.csv

Usage:
    python src/ingestion/scrape_overthecap.py
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

OTC_CONTRACT_URL = "https://overthecap.com/contract-history/quarterback"
OTC_CAP_URL = "https://overthecap.com/salary-cap-history"

DELAY_SECONDS = 2  # Respect robots.txt


def fetch_page(url: str) -> BeautifulSoup:
    """Fetch a page and return parsed BeautifulSoup. Raises on failure."""
    logger.info(f"Fetching {url}")
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    time.sleep(DELAY_SECONDS)
    return BeautifulSoup(resp.text, "html.parser")


def scrape_qb_contracts() -> pd.DataFrame:
    """Scrape QB contract history table from OverTheCap."""
    soup = fetch_page(OTC_CONTRACT_URL)

    # OTC renders contracts in a standard HTML table
    tables = pd.read_html(str(soup), flavor="bs4")
    if not tables:
        raise ValueError("No tables found on OTC contract page")

    df = tables[0]
    logger.info(f"Raw contract table shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def scrape_cap_history() -> pd.DataFrame:
    """Scrape annual salary cap totals from OverTheCap."""
    soup = fetch_page(OTC_CAP_URL)
    tables = pd.read_html(str(soup), flavor="bs4")
    if not tables:
        raise ValueError("No tables found on OTC cap history page")

    df = tables[0]
    logger.info(f"Cap history table shape: {df.shape}")
    return df


def main():
    logger.info("=== Scraping OverTheCap ===")

    contracts_df = scrape_qb_contracts()
    contracts_path = RAW_DIR / "qb_contracts_raw.csv"
    contracts_df.to_csv(contracts_path, index=False)
    logger.info(f"Saved contracts to {contracts_path}")

    cap_df = scrape_cap_history()
    cap_path = RAW_DIR / "cap_history_raw.csv"
    cap_df.to_csv(cap_path, index=False)
    logger.info(f"Saved cap history to {cap_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
