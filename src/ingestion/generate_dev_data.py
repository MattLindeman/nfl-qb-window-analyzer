"""
Generate realistic synthetic data for development.
Grounded in real QB contract signings and team outcomes (2000-2024).

Outputs:
  data/raw/qb_contracts_raw.csv
  data/raw/cap_history_raw.csv
  data/raw/team_seasons_raw.csv

Run:
    python src/ingestion/generate_dev_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. NFL salary cap by year (real values, millions) ─────────────────────────
CAP_BY_YEAR = {
    2000: 62.172, 2001: 67.405, 2002: 71.101, 2003: 75.007, 2004: 80.582,
    2005: 85.500, 2006: 102.000, 2007: 109.000, 2008: 116.000, 2009: 123.000,
    2010: 102.000, 2011: 120.375, 2012: 120.600, 2013: 123.000, 2014: 133.000,
    2015: 143.280, 2016: 155.270, 2017: 167.000, 2018: 177.200, 2019: 188.200,
    2020: 198.200, 2021: 182.500, 2022: 208.200, 2023: 224.800, 2024: 255.400,
}

# ── 2. QB contract signings (real signings, realistic values) ──────────────────
# Each tuple: (player, team, signing_year, years, total_value_M, aav_M, guaranteed_M, age_at_signing)
# Sources: real contracts from memory, rounded to nearest $0.5M
QB_CONTRACTS_RAW = [
    # Early 2000s
    ("Daunte Culpepper",   "MIN", 2002, 10, 102.0,  10.2,  18.0, 25),
    ("Michael Vick",       "ATL", 2004, 10, 130.0,  13.0,  37.0, 23),
    ("Donovan McNabb",     "PHI", 2002,  9,  99.5,  11.1,  19.0, 25),
    ("Drew Bledsoe",       "BUF", 2002,  7,  53.0,   7.6,  20.0, 30),
    ("Jeff Garcia",        "CLE", 2004,  4,  25.0,   6.3,   8.0, 34),
    ("Trent Green",        "KC",  2003,  5,  30.0,   6.0,  10.0, 32),
    ("Peyton Manning",     "IND", 2004,  7,  98.0,  14.0,  34.5, 28),
    ("Peyton Manning",     "IND", 2011,  5, 100.0,  20.0,  69.0, 35),
    ("Peyton Manning",     "DEN", 2012,  5,  96.0,  19.2,  58.0, 36),
    ("Ben Roethlisberger", "PIT", 2008,  8,  87.4,  10.9,  33.4, 26),
    ("Ben Roethlisberger", "PIT", 2015,  5,  87.4,  21.9,  30.8, 33),
    ("Eli Manning",        "NYG", 2008,  6,  97.5,  16.3,  35.0, 27),
    ("Eli Manning",        "NYG", 2015,  4,  84.0,  21.0,  34.6, 34),
    ("Matt Ryan",          "ATL", 2013,  5,  59.5,  11.9,  30.0, 28),
    ("Matt Ryan",          "ATL", 2018,  5, 150.0,  30.0,  100.0, 33),
    ("Philip Rivers",      "LAC", 2009,  6, 100.0,  16.7,  35.0, 27),
    ("Philip Rivers",      "LAC", 2015,  4,  83.3,  20.8,  37.5, 33),
    ("Jay Cutler",         "CHI", 2009,  7, 126.7,  18.1,  30.0, 26),
    ("Tony Romo",          "DAL", 2013,  6, 108.0,  18.0,  55.0, 33),
    ("Tony Romo",          "DAL", 2016,  4,  54.0,  19.6,  34.0, 36),
    ("Joe Flacco",         "BAL", 2013,  6, 120.6,  20.1,  52.0, 28),
    ("Andy Dalton",        "CIN", 2014,  6,  96.0,  16.0,  17.0, 27),
    ("Andrew Luck",        "IND", 2016,  6, 140.0,  23.3,  87.0, 26),
    ("Cam Newton",         "CAR", 2015,  5, 103.8,  20.8,  60.0, 26),
    ("Russell Wilson",     "SEA", 2015,  4, 140.0,  21.9,  31.0, 26),
    ("Russell Wilson",     "SEA", 2019,  4, 140.0,  35.0,  107.0, 30),
    ("Russell Wilson",     "DEN", 2022,  5, 245.0,  49.0,  165.0, 33),
    ("Aaron Rodgers",      "GB",  2013,  5, 110.0,  22.0,  54.5, 29),
    ("Aaron Rodgers",      "GB",  2018,  4, 134.0,  33.5,  103.0, 34),
    ("Aaron Rodgers",      "GB",  2022,  4, 200.0,  50.3,  153.0, 38),
    ("Aaron Rodgers",      "NYJ", 2023,  4, 112.5,  37.5,  75.0, 39),
    ("Drew Brees",         "NO",  2006,  6,  60.0,  10.0,  20.0, 27),
    ("Drew Brees",         "NO",  2012,  5, 100.0,  20.0,  60.0, 33),
    ("Drew Brees",         "NO",  2016,  2,  44.0,  22.0,  22.0, 37),
    ("Tom Brady",          "NE",  2005,  6,  60.0,  10.0,  22.0, 28),
    ("Tom Brady",          "NE",  2010,  4,  72.0,  18.0,  48.0, 33),
    ("Tom Brady",          "NE",  2013,  3,  57.0,  22.5,  30.0, 35),
    ("Tom Brady",          "NE",  2016,  2,  41.0,  20.5,  28.0, 39),
    ("Tom Brady",          "TB",  2020,  2,  50.0,  25.0,  25.0, 42),
    ("Matthew Stafford",   "DET", 2013,  5,  76.5,  17.7,  43.0, 25),
    ("Matthew Stafford",   "DET", 2017,  5, 135.0,  27.0,  92.0, 29),
    ("Matthew Stafford",   "LAR", 2022,  4, 160.0,  40.0,  135.0, 34),
    ("Derek Carr",         "LV",  2017,  5, 125.0,  25.0,  70.0, 26),
    ("Derek Carr",         "LV",  2022,  5, 121.5,  32.5,  100.0, 31),
    ("Kirk Cousins",       "WSH", 2016,  1,  19.95, 19.95,  19.95, 28),
    ("Kirk Cousins",       "MIN", 2018,  3,  84.0,  28.0,  84.0, 29),
    ("Kirk Cousins",       "MIN", 2020,  2,  66.0,  33.0,  61.0, 32),
    ("Kirk Cousins",       "MIN", 2022,  4, 100.0,  35.0,  90.0, 34),
    ("Kirk Cousins",       "ATL", 2024,  4, 180.0,  45.0, 100.0, 35),
    ("Dak Prescott",       "DAL", 2021,  4, 160.0,  40.0, 126.0, 27),
    ("Dak Prescott",       "DAL", 2024,  4, 240.0,  60.0, 231.0, 30),
    ("Carson Wentz",       "PHI", 2019,  4, 128.0,  32.0, 107.9, 26),
    ("Jared Goff",         "LAR", 2019,  4, 134.0,  33.5, 110.0, 24),
    ("Jared Goff",         "DET", 2023,  4, 212.0,  53.0, 170.0, 28),
    ("Patrick Mahomes",    "KC",  2020, 10, 450.0,  45.0, 141.0, 25),
    ("Patrick Mahomes",    "KC",  2024,  2,  89.0,  44.5,  82.5, 28),
    ("Josh Allen",         "BUF", 2021,  6, 258.0,  43.0, 150.0, 25),
    ("Josh Allen",         "BUF", 2024,  6, 330.0,  55.0, 250.0, 28),
    ("Lamar Jackson",      "BAL", 2023,  5, 260.0,  52.0, 185.0, 26),
    ("Joe Burrow",         "CIN", 2023,  5, 275.0,  55.0, 219.0, 26),
    ("Justin Herbert",     "LAC", 2023,  5, 262.5,  52.5, 218.5, 25),
    ("Tua Tagovailoa",     "MIA", 2023,  4, 212.4,  53.1, 166.9, 25),
    ("Trevor Lawrence",    "JAX", 2023,  5, 275.0,  55.0, 200.0, 23),
    ("Jordan Love",        "GB",  2024,  4, 220.0,  55.0, 160.0, 25),
    ("Kyler Murray",       "ARI", 2022,  5, 230.5,  46.1, 160.0, 24),
    ("Deshaun Watson",     "CLE", 2022,  5, 230.0,  46.0, 230.0, 26),
    ("Sam Darnold",        "CAR", 2021,  3,  21.6,   7.2,   7.5, 24),
    ("Geno Smith",         "SEA", 2023,  3, 105.0,  35.0,  40.0, 32),
    ("Daniel Jones",       "NYG", 2023,  4, 160.0,  40.0, 82.6, 26),
    ("Baker Mayfield",     "CLE", 2022,  5, 189.0,  18.9,  35.0, 27),
    ("Marcus Mariota",     "TEN", 2018,  1,  21.0,  20.9,  20.9, 25),
    ("Sam Bradford",       "PHI", 2015,  2,  35.0,  17.5,  22.0, 27),
    ("Alex Smith",         "KC",  2017,  4, 94.0,   23.5,  46.0, 33),
    ("Jimmy Garoppolo",    "SF",  2018,  5, 137.5,  27.5,  74.1, 26),
    ("Ryan Tannehill",     "TEN", 2020,  4, 118.0,  29.5,  62.0, 31),
    ("Matt Schaub",        "HOU", 2009,  6,  62.0,  10.3,  24.0, 28),
    ("Chad Henne",         "JAX", 2014,  3,  12.8,   4.3,   4.3, 29),
    ("Nick Foles",         "JAX", 2019,  4,  88.0,  22.0,  50.6, 30),
    ("Jameis Winston",     "NO",  2021,  1,   5.3,   5.3,   5.3, 27),
    ("Teddy Bridgewater",  "DEN", 2021,  3,  60.0,  20.0,  25.0, 28),
    ("Gardner Minshew",    "PHI", 2022,  2,  10.0,   5.0,   4.5, 24),
    ("Case Keenum",        "DEN", 2018,  2,  36.0,  18.0,  25.0, 30),
    ("Ryan Fitzpatrick",   "NYJ", 2016,  2,  12.0,   6.0,   6.0, 33),
    ("Tyrod Taylor",       "BUF", 2016,  4,  27.5,  10.0,  18.0, 27),
]

# ── 3. Team seasons (wins, playoff result) 2000-2024 ──────────────────────────
# Format: (team, year, wins, losses, made_playoffs, playoff_exit)
# playoff_exit: None, 'WC', 'DIV', 'CCG', 'SB_loss', 'SB_win'
TEAM_SEASONS_RAW = [
    # 2000
    ("BAL", 2000, 12, 4, True, "SB_win"),    ("NYG", 2000, 12, 4, True, "SB_loss"),
    ("MN",  2000, 11, 5, True, "CCG"),        ("TEN", 2000, 13, 3, True, "DIV"),
    ("OAK", 2000, 12, 4, True, "CCG"),        ("MIA", 2000, 11, 5, True, "WC"),
    ("PIT", 2000, 9,  7, True, "DIV"),        ("PHI", 2000, 11, 5, True, "WC"),
    ("NO",  2000, 10, 6, True, "WC"),         ("KC",  2000, 7,  9, False, None),

    # 2001
    ("NE",  2001, 11, 5, True, "SB_win"),    ("STL", 2001, 14, 2, True, "SB_loss"),
    ("PIT", 2001, 13, 3, True, "CCG"),        ("BAL", 2001, 10, 6, True, "WC"),
    ("PHI", 2001, 11, 5, True, "CCG"),        ("CHI", 2001, 13, 3, True, "DIV"),
    ("GB",  2001, 12, 4, True, "WC"),         ("SF",  2001, 12, 4, True, "WC"),

    # 2002
    ("TB",  2002, 12, 4, True, "SB_win"),    ("OAK", 2002, 11, 5, True, "SB_loss"),
    ("TEN", 2002, 11, 5, True, "WC"),         ("PIT", 2002, 10, 5, True, "WC"),
    ("PHI", 2002, 12, 4, True, "CCG"),        ("ATL", 2002, 9,  6, True, "WC"),
    ("GB",  2002, 12, 4, True, "WC"),         ("SF",  2002, 10, 6, True, "DIV"),

    # 2003
    ("NE",  2003, 14, 2, True, "SB_win"),    ("CAR", 2003, 11, 5, True, "SB_loss"),
    ("IND", 2003, 12, 4, True, "CCG"),        ("TEN", 2003, 12, 4, True, "DIV"),
    ("PHI", 2003, 12, 4, True, "CCG"),        ("GB",  2003, 10, 6, True, "WC"),
    ("STL", 2003, 12, 4, True, "WC"),         ("SEA", 2003, 10, 6, True, "WC"),

    # 2004
    ("NE",  2004, 14, 2, True, "SB_win"),    ("PHI", 2004, 13, 3, True, "SB_loss"),
    ("PIT", 2004, 15, 1, True, "CCG"),        ("IND", 2004, 12, 4, True, "DIV"),
    ("ATL", 2004, 11, 5, True, "CCG"),        ("GB",  2004, 10, 6, True, "WC"),
    ("MIN", 2004, 8,  8, True, "DIV"),        ("SEA", 2004, 9,  7, True, "WC"),

    # 2005
    ("PIT", 2005, 11, 5, True, "SB_win"),    ("SEA", 2005, 13, 3, True, "SB_loss"),
    ("IND", 2005, 14, 2, True, "DIV"),        ("DEN", 2005, 13, 3, True, "CCG"),
    ("CHI", 2005, 11, 5, True, "DIV"),        ("NYG", 2005, 11, 5, True, "WC"),
    ("CAR", 2005, 11, 5, True, "CCG"),        ("WAS", 2005, 10, 6, True, "WC"),

    # 2006
    ("IND", 2006, 12, 4, True, "SB_win"),    ("CHI", 2006, 13, 3, True, "SB_loss"),
    ("NE",  2006, 12, 4, True, "CCG"),        ("SD",  2006, 14, 2, True, "DIV"),
    ("NO",  2006, 10, 6, True, "CCG"),        ("PHI", 2006, 10, 6, True, "WC"),
    ("SEA", 2006, 9,  7, True, "WC"),         ("PHI", 2006, 10, 6, True, "WC"),

    # 2007
    ("NYG", 2007, 10, 6, True, "SB_win"),    ("NE",  2007, 16, 0, True, "SB_loss"),
    ("GB",  2007, 13, 3, True, "CCG"),        ("DAL", 2007, 13, 3, True, "DIV"),
    ("IND", 2007, 13, 3, True, "WC"),         ("SD",  2007, 11, 5, True, "DIV"),
    ("WAS", 2007, 9,  7, True, "WC"),         ("SEA", 2007, 10, 6, True, "DIV"),

    # 2008
    ("PIT", 2008, 12, 4, True, "SB_win"),    ("ARI", 2008, 9,  7, True, "SB_loss"),
    ("TEN", 2008, 13, 3, True, "DIV"),        ("BAL", 2008, 11, 5, True, "CCG"),
    ("PHI", 2008, 9,  6, True, "CCG"),        ("NYG", 2008, 12, 4, True, "DIV"),
    ("CAR", 2008, 12, 4, True, "DIV"),        ("ARI", 2008, 9,  7, True, "SB_loss"),

    # 2009
    ("NO",  2009, 13, 3, True, "SB_win"),    ("IND", 2009, 14, 2, True, "SB_loss"),
    ("SD",  2009, 13, 3, True, "DIV"),        ("NE",  2009, 10, 6, True, "WC"),
    ("MIN", 2009, 12, 4, True, "CCG"),        ("DAL", 2009, 11, 5, True, "DIV"),
    ("ARI", 2009, 10, 6, True, "WC"),         ("GB",  2009, 11, 5, True, "WC"),

    # 2010
    ("GB",  2010, 10, 6, True, "SB_win"),    ("PIT", 2010, 12, 4, True, "SB_loss"),
    ("NE",  2010, 14, 2, True, "CCG"),        ("BAL", 2010, 12, 4, True, "CCG"),
    ("ATL", 2010, 13, 3, True, "DIV"),        ("PHI", 2010, 10, 6, True, "WC"),
    ("CHI", 2010, 11, 5, True, "CCG"),        ("SEA", 2010, 7,  9, True, "DIV"),

    # 2011
    ("NYG", 2011, 9,  7, True, "SB_win"),    ("NE",  2011, 13, 3, True, "SB_loss"),
    ("GB",  2011, 15, 1, True, "DIV"),        ("SF",  2011, 13, 3, True, "CCG"),
    ("BAL", 2011, 12, 4, True, "CCG"),        ("HOU", 2011, 10, 6, True, "DIV"),
    ("NO",  2011, 13, 3, True, "DIV"),        ("ATL", 2011, 10, 6, True, "WC"),

    # 2012
    ("BAL", 2012, 10, 6, True, "SB_win"),    ("SF",  2012, 11, 4, True, "SB_loss"),
    ("NE",  2012, 12, 4, True, "CCG"),        ("DEN", 2012, 13, 3, True, "DIV"),
    ("ATL", 2012, 13, 3, True, "CCG"),        ("SEA", 2012, 11, 5, True, "WC"),
    ("HOU", 2012, 12, 4, True, "DIV"),        ("GB",  2012, 11, 5, True, "DIV"),

    # 2013
    ("SEA", 2013, 13, 3, True, "SB_win"),    ("DEN", 2013, 13, 3, True, "SB_loss"),
    ("NE",  2013, 12, 4, True, "CCG"),        ("IND", 2013, 11, 5, True, "DIV"),
    ("SF",  2013, 12, 4, True, "CCG"),        ("NO",  2013, 11, 5, True, "DIV"),
    ("CAR", 2013, 12, 4, True, "WC"),         ("PHI", 2013, 10, 6, True, "WC"),

    # 2014
    ("NE",  2014, 12, 4, True, "SB_win"),    ("SEA", 2014, 12, 4, True, "SB_loss"),
    ("DEN", 2014, 12, 4, True, "DIV"),        ("IND", 2014, 11, 5, True, "CCG"),
    ("GB",  2014, 12, 4, True, "CCG"),        ("DAL", 2014, 12, 4, True, "DIV"),
    ("BAL", 2014, 10, 6, True, "WC"),         ("ARI", 2014, 11, 5, True, "DIV"),

    # 2015
    ("DEN", 2015, 12, 4, True, "SB_win"),    ("CAR", 2015, 15, 1, True, "SB_loss"),
    ("NE",  2015, 12, 4, True, "CCG"),        ("KC",  2015, 11, 5, True, "DIV"),
    ("ARI", 2015, 13, 3, True, "CCG"),        ("MIN", 2015, 11, 5, True, "WC"),
    ("SEA", 2015, 10, 6, True, "DIV"),        ("GB",  2015, 10, 6, True, "WC"),

    # 2016
    ("NE",  2016, 14, 2, True, "SB_win"),    ("ATL", 2016, 11, 5, True, "SB_loss"),
    ("KC",  2016, 12, 4, True, "DIV"),        ("PIT", 2016, 11, 5, True, "CCG"),
    ("GB",  2016, 10, 6, True, "CCG"),        ("DAL", 2016, 13, 3, True, "DIV"),
    ("SEA", 2016, 10, 6, True, "WC"),         ("HOU", 2016, 9,  7, True, "DIV"),

    # 2017
    ("PHI", 2017, 13, 3, True, "SB_win"),    ("NE",  2017, 13, 3, True, "SB_loss"),
    ("PIT", 2017, 13, 3, True, "DIV"),        ("JAX", 2017, 10, 6, True, "CCG"),
    ("MIN", 2017, 13, 3, True, "CCG"),        ("NO",  2017, 11, 5, True, "DIV"),
    ("CAR", 2017, 11, 5, True, "WC"),         ("ATL", 2017, 10, 6, True, "WC"),

    # 2018
    ("NE",  2018, 11, 5, True, "SB_win"),    ("LAR", 2018, 13, 3, True, "SB_loss"),
    ("KC",  2018, 12, 4, True, "CCG"),        ("IND", 2018, 10, 6, True, "DIV"),
    ("NO",  2018, 13, 3, True, "CCG"),        ("PHI", 2018, 9,  7, True, "WC"),
    ("SEA", 2018, 10, 6, True, "WC"),         ("DAL", 2018, 10, 6, True, "DIV"),

    # 2019
    ("KC",  2019, 12, 4, True, "SB_win"),    ("SF",  2019, 13, 3, True, "SB_loss"),
    ("BAL", 2019, 14, 2, True, "DIV"),        ("HOU", 2019, 10, 6, True, "DIV"),
    ("GB",  2019, 13, 3, True, "CCG"),        ("SEA", 2019, 11, 5, True, "DIV"),
    ("NO",  2019, 13, 3, True, "WC"),         ("PHI", 2019, 9,  7, True, "WC"),

    # 2020 (17 game schedule, 14 playoff teams)
    ("TB",  2020, 11, 5, True, "SB_win"),    ("KC",  2020, 14, 2, True, "SB_loss"),
    ("BUF", 2020, 13, 3, True, "CCG"),        ("BAL", 2020, 11, 5, True, "DIV"),
    ("GB",  2020, 13, 3, True, "CCG"),        ("NO",  2020, 12, 4, True, "DIV"),
    ("SEA", 2020, 12, 4, True, "WC"),         ("LAR", 2020, 10, 6, True, "DIV"),
    ("CLE", 2020, 11, 5, True, "WC"),         ("IND", 2020, 11, 5, True, "WC"),
    ("WAS", 2020, 7,  9, True, "WC"),         ("CHI", 2020, 8,  8, True, "WC"),

    # 2021
    ("LAR", 2021, 12, 5, True, "SB_win"),    ("CIN", 2021, 10, 7, True, "SB_loss"),
    ("KC",  2021, 12, 5, True, "CCG"),        ("BUF", 2021, 11, 6, True, "DIV"),
    ("GB",  2021, 13, 4, True, "DIV"),        ("SF",  2021, 10, 7, True, "CCG"),
    ("TB",  2021, 13, 4, True, "DIV"),        ("DAL", 2021, 12, 5, True, "WC"),
    ("ARI", 2021, 11, 6, True, "WC"),         ("LAC", 2021, 9,  8, True, "WC"),

    # 2022
    ("KC",  2022, 14, 3, True, "SB_win"),    ("PHI", 2022, 14, 3, True, "SB_loss"),
    ("BUF", 2022, 13, 3, True, "DIV"),        ("CIN", 2022, 12, 4, True, "CCG"),
    ("SF",  2022, 13, 4, True, "CCG"),        ("DAL", 2022, 12, 5, True, "DIV"),
    ("MIN", 2022, 13, 4, True, "WC"),         ("NYG", 2022, 9,  7, True, "DIV"),

    # 2023
    ("KC",  2023, 11, 6, True, "SB_win"),    ("SF",  2023, 12, 5, True, "SB_loss"),
    ("BUF", 2023, 11, 6, True, "DIV"),        ("BAL", 2023, 13, 4, True, "CCG"),
    ("DET", 2023, 12, 5, True, "CCG"),        ("TB",  2023, 9,  8, True, "DIV"),
    ("DAL", 2023, 12, 5, True, "WC"),         ("PHI", 2023, 11, 6, True, "WC"),
    ("LAR", 2023, 10, 7, True, "WC"),         ("CLE", 2023, 11, 6, True, "WC"),

    # 2024 (partial list of playoff teams)
    ("PHI", 2024, 14, 3, True, "SB_win"),    ("KC",  2024, 15, 2, True, "SB_loss"),
    ("BUF", 2024, 13, 4, True, "CCG"),        ("BAL", 2024, 12, 5, True, "DIV"),
    ("DET", 2024, 15, 2, True, "CCG"),        ("MIN", 2024, 14, 3, True, "WC"),
    ("LAR", 2024, 10, 7, True, "DIV"),        ("HOU", 2024, 10, 7, True, "WC"),
]


def build_contracts_df() -> pd.DataFrame:
    cols = ["player", "team", "signing_year", "years", "total_value_m",
            "aav_m", "guaranteed_m", "age_at_signing"]
    df = pd.DataFrame(QB_CONTRACTS_RAW, columns=cols)
    return df


def build_cap_history_df() -> pd.DataFrame:
    rows = [{"year": yr, "cap_total_m": cap} for yr, cap in CAP_BY_YEAR.items()]
    return pd.DataFrame(rows)


def build_team_seasons_df() -> pd.DataFrame:
    cols = ["team", "season", "wins", "losses", "made_playoffs", "playoff_exit"]
    df = pd.DataFrame(TEAM_SEASONS_RAW, columns=cols)

    # Fill out non-playoff teams with realistic records
    # Build a complete grid: all 32 teams x all seasons
    all_teams = [
        "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN",
        "DET","GB","HOU","IND","JAX","KC","LAC","LAR","LV","MIA",
        "MIN","NE","NO","NYG","NYJ","OAK","PHI","PIT","SD","SEA",
        "SF","STL","TB","TEN","WAS","WSH"
    ]
    # Normalize old team names
    team_map = {"OAK": "LV", "SD": "LAC", "STL": "LAR", "WSH": "WAS", "MN": "MIN"}

    df["team"] = df["team"].replace(team_map)

    np.random.seed(42)
    seasons = list(range(2000, 2025))
    existing = set(zip(df["team"], df["season"]))

    rows = []
    for team in all_teams:
        team_norm = team_map.get(team, team)
        for season in seasons:
            if (team_norm, season) not in existing:
                # Generate a plausible non-playoff record
                wins = int(np.clip(np.random.normal(7, 2.5), 2, 10))
                losses = 16 - wins if season <= 2020 else 17 - wins
                rows.append({
                    "team": team_norm,
                    "season": season,
                    "wins": wins,
                    "losses": max(losses, 0),
                    "made_playoffs": False,
                    "playoff_exit": None
                })

    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df = df.drop_duplicates(subset=["team", "season"])
    df = df.sort_values(["team", "season"]).reset_index(drop=True)
    return df


def main():
    print("Generating development data...")

    contracts = build_contracts_df()
    contracts.to_csv(RAW_DIR / "qb_contracts_raw.csv", index=False)
    print(f"  qb_contracts_raw.csv    — {len(contracts)} QB signings")

    cap = build_cap_history_df()
    cap.to_csv(RAW_DIR / "cap_history_raw.csv", index=False)
    print(f"  cap_history_raw.csv     — {len(cap)} years")

    seasons = build_team_seasons_df()
    seasons.to_csv(RAW_DIR / "team_seasons_raw.csv", index=False)
    print(f"  team_seasons_raw.csv    — {len(seasons)} team-seasons")

    print("Done. Raw data written to data/raw/")


if __name__ == "__main__":
    main()
