#!/usr/bin/env python3
"""
EDGE Predict - Brain v1.0
Complete sports prediction engine. Primary targets: CBS Confidence Pool
and March Madness brackets. Secondary: general predictions and betting picks.

Files:
  brain.py   - all math, training, signals, predictions (this file)
  index.html - visual display, reads data.json
  data.json  - permanent storage (weights, predictions, record)
  train.yml  - GitHub Actions automation

Design rules applied everywhere:
  1. Validate inputs at the start of every function
  2. Check denominator is not zero before every division
  3. Every API call has try/except and a documented fallback
  4. Every probability is clamped to [0.05, 0.95]
  5. Walk-forward validation - no data leakage during training
  6. No hardcoded accuracy ceilings
  7. Functions are short and single-purpose
  8. Every list access checks length first
"""

# ============================================================
# SECTION 1: IMPORTS
# ============================================================
import os
import sys
import json
import math
import copy
import time
import logging
import argparse
import datetime
import traceback
from typing import Optional, Dict, List, Tuple, Any

import pickle
import base64
import io
import re

import requests
import numpy as np

# XGBoost / scikit-learn — graceful fallback if not installed
try:
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    log_placeholder = None  # logger not yet created; warn at runtime

# Module-level model store — populated by load_xgb_models() at startup
XGB_MODELS: Dict[str, Any] = {}
XGB_FEATURE_NAMES: List[str] = []

# --- New intelligence layer imports ---
try:
    from mind import train_mind, update_mind, refine_prediction as mind_refine
    MIND_AVAILABLE = True
except ImportError:
    MIND_AVAILABLE = False

try:
    from edge import update_edge
    EDGE_AVAILABLE = True
except ImportError:
    EDGE_AVAILABLE = False

try:
    from scout import train_scout, update_scout
    SCOUT_AVAILABLE = True
except ImportError:
    SCOUT_AVAILABLE = False

try:
    from cbs import run_cbs
    CBS_AVAILABLE = True
except ImportError:
    CBS_AVAILABLE = False

try:
    from march import run_march
    MARCH_AVAILABLE = True
except ImportError:
    MARCH_AVAILABLE = False

try:
    from transformer import train_transformer, predict_transformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

try:
    from graph import train_graph, predict_graph
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False

try:
    from stack import train_stack, stack_predict
    STACK_AVAILABLE = True
except ImportError:
    STACK_AVAILABLE = False

try:
    from explain import initialize_explain, explain_prediction
    EXPLAIN_AVAILABLE = True
except ImportError:
    EXPLAIN_AVAILABLE = False

try:
    from simulate import train_simulate, simulate_game  # noqa: F401
    # DISABLED: simulate.py requires data['nfl_drives'] and data['nba_possessions']
    # which no code currently populates. Training produces an empty model.
    # Re-enable once the nflverse play-by-play data pipeline is built.
    SIMULATE_AVAILABLE = False
except ImportError:
    SIMULATE_AVAILABLE = False

try:
    from audit import run_audit
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False

try:
    from baseline import run_baseline
    BASELINE_AVAILABLE = True
except ImportError:
    BASELINE_AVAILABLE = False

try:
    from tournament import run_tournament
    TOURNAMENT_AVAILABLE = True
except ImportError:
    TOURNAMENT_AVAILABLE = False

try:
    from tune import run_tune
    TUNE_AVAILABLE = True
except ImportError:
    TUNE_AVAILABLE = False

try:
    from portfolio import run_portfolio
    PORTFOLIO_AVAILABLE = True
except ImportError:
    PORTFOLIO_AVAILABLE = False

# ============================================================
# SECTION 2: LOGGING
# ============================================================
LOG_BUFFER: List[dict] = []


class BufferedHandler(logging.Handler):
    """Captures log records into LOG_BUFFER for saving to data.json."""
    def emit(self, record: logging.LogRecord) -> None:
        LOG_BUFFER.append({
            'time': datetime.datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': self.format(record)
        })


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout), BufferedHandler()]
)
log = logging.getLogger(__name__)

# ============================================================
# SECTION 3: CONSTANTS
# ALL configurable values live here only. Changing a value here
# changes it everywhere automatically.
# ============================================================

APP_VERSION        = '1.0'
DATA_FILE          = 'data.json'
TRAIN_START_YEAR   = 1999  # nflverse: 1999+; ESPN: ~2002+ (skips silently if no data)
CURRENT_YEAR       = datetime.datetime.utcnow().year
YESTERDAY          = (datetime.datetime.utcnow() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

# API keys - always from environment variables, never hardcoded
ODDS_API_KEY      = os.environ.get('ODDS_API_KEY', '')
OPENWEATHER_KEY   = os.environ.get('OPENWEATHER_KEY', '')
BALLDONTLIE_KEY   = os.environ.get('BALLDONTLIE_KEY', '')

# Training parameters
LEARNING_RATE          = 0.015
MAX_TRAIN_ITERATIONS   = 150
CONVERGENCE_DELTA      = 0.0003   # Stop when improvement < this per iteration
CONVERGENCE_PLATEAU    = 3        # Stop after this many consecutive non-improving iterations
MIN_PATTERN_GAMES      = 5        # Minimum games before pattern is trusted
MAX_PATTERNS_PER_TYPE  = 2000
MAX_TEAM_HISTORY       = 15       # Most recent games stored per team

# How many historical games to store per sport in data['all_games_by_sport'].
# These are used by transformer.py and graph.py for advanced model training.
# Priority sports (NFL graph, NCAAB transformer+graph) get full or near-full history.
# Fill-in sports (NBA, MLB, NHL, NCAAF) get enough for graph training only.
#
# NFL:    All ~6,750 games (1999-2024, 25 seasons * ~270 games). Graph only - no transformer.
# NCAABM: 25,000 most recent games (~4-5 seasons of all D1). Transformer + graph.
# NCAABW: 25,000 most recent games (~4-5 seasons of all D1). Transformer + graph.
# NBA:    2,500 most recent games (~2 seasons). Light transformer + graph.
# MLB:    2,500 most recent games (~1 season). Light transformer + graph.
# NHL:    2,500 most recent games (~2 seasons). Light transformer + graph.
# NCAAF:  2,500 most recent games (~3-4 seasons). Graph only - transformer inactive.
#
# Total estimated data.json addition: ~12.5 MB (well within GitHub's 100 MB limit).
GAME_STORAGE_LIMITS: Dict[str, int] = {
    'nfl':    7000,   # Generous ceiling — full 25-year history fits under this
    'ncaabm': 25000,
    'ncaabw': 25000,
    'nba':    2500,
    'mlb':    2500,
    'nhl':    2500,
    'ncaaf':  2500,
}
WEIGHT_MIN             = 0.05
WEIGHT_MAX             = 8.0
MONTE_CARLO_SIMS       = 500000   # Simulations for March Madness (overkill = rock-solid)
CBS_SIMS               = 50000    # Simulations for CBS week-survival optimization

# Home field advantage defaults (2024 research values)
HFA_DEFAULT: Dict[str, float] = {
    'nfl': 0.575, 'nba': 0.600, 'mlb': 0.540, 'nhl': 0.550,
    'ncaaf': 0.620, 'ncaabm': 0.640, 'ncaabw': 0.640
}

# Pythagorean exponents - research validated per sport
PYTHAG_EXP: Dict[str, float] = {
    'nfl': 2.37, 'nba': 13.91, 'mlb': 1.83, 'nhl': 2.15,
    'ncaaf': 2.37, 'ncaabm': 10.25, 'ncaabw': 10.25
}

# Margin normalization per sport - what counts as a "big" margin
MARGIN_NORM: Dict[str, int] = {
    'nfl': 14, 'nba': 10, 'mlb': 2, 'nhl': 1,
    'ncaaf': 14, 'ncaabm': 8, 'ncaabw': 8
}

# Close game threshold per sport - for turnover proxy calculation
CLOSE_GAME_THRESH: Dict[str, int] = {
    'nfl': 8, 'nba': 6, 'mlb': 2, 'nhl': 1,
    'ncaaf': 8, 'ncaabm': 6, 'ncaabw': 6
}

# Blowout threshold per sport - for bounce-back signal
BLOWOUT_THRESH: Dict[str, int] = {
    'nfl': 21, 'nba': 25, 'mlb': 5, 'nhl': 3,
    'ncaaf': 21, 'ncaabm': 20, 'ncaabw': 20
}

# Recent form window per sport
FORM_WINDOW: Dict[str, int] = {
    'nfl': 6, 'nba': 4, 'mlb': 7, 'nhl': 5,
    'ncaaf': 4, 'ncaabm': 5, 'ncaabw': 5
}

# Sports played outdoors where weather matters
OUTDOOR_SPORTS = frozenset({'nfl', 'mlb', 'ncaaf'})

# Venues with documented home field advantage
VENUE_HFA_OVERRIDE: Dict[str, float] = {
    'Arrowhead Stadium': 0.789,
    'Highmark Stadium': 0.789,
    "Levi's Stadium": 0.733,
    'Caesars Superdome': 0.689,
    'M&T Bank Stadium': 0.684,
    'U.S. Bank Stadium': 0.655,
    'Allegiant Stadium': 0.600,
    'SoFi Stadium': 0.560,
    'Gillette Stadium': 0.430,
    'Bank of America Stadium': 0.430
}

# Stadium coordinates for weather lookups (lat, lon).
# ESPN's venue address object does not include latitude/longitude — these are the
# authoritative coordinates used by fetch_espn_scoreboard() as fallback.
# Keys are lowercase ESPN fullName values. Any unmatched venue gracefully returns
# no weather data (same as the previous broken state).
VENUE_COORDS: Dict[str, Tuple[float, float]] = {
    # ── NFL ──────────────────────────────────────────────────────────────────
    'arrowhead stadium':                    (39.0489, -94.4839),   # Chiefs
    'highmark stadium':                     (42.7738, -78.7870),   # Bills
    "levi's stadium":                       (37.4032, -121.9698),  # 49ers
    'caesars superdome':                    (29.9511, -90.0812),   # Saints
    'm&t bank stadium':                     (39.2780, -76.6227),   # Ravens
    'u.s. bank stadium':                    (44.9736, -93.2575),   # Vikings
    'allegiant stadium':                    (36.0909, -115.1833),  # Raiders
    'sofi stadium':                         (33.9535, -118.3392),  # Rams / Chargers
    'gillette stadium':                     (42.0909, -71.2643),   # Patriots
    'bank of america stadium':              (35.2258, -80.8531),   # Panthers
    'at&t stadium':                         (32.7480, -97.0931),   # Cowboys
    'lincoln financial field':              (39.9007, -75.1675),   # Eagles
    'metlife stadium':                      (40.8135, -74.0745),   # Giants / Jets
    'soldier field':                        (41.8623, -87.6167),   # Bears
    'lambeau field':                        (44.5013, -88.0622),   # Packers
    'ford field':                           (42.3400, -83.0456),   # Lions
    'raymond james stadium':                (27.9759, -82.5033),   # Buccaneers
    'mercedes-benz stadium':                (33.7554, -84.4007),   # Falcons
    'empower field at mile high':           (39.7439, -105.0201),  # Broncos
    'empower field':                        (39.7439, -105.0201),  # Broncos alt name
    'lumen field':                          (47.5952, -122.3316),  # Seahawks
    'state farm stadium':                   (33.5276, -112.2626),  # Cardinals
    'lucas oil stadium':                    (39.7601, -86.1639),   # Colts
    'nrg stadium':                          (29.6847, -95.4107),   # Texans
    'acrisure stadium':                     (40.4468, -80.0158),   # Steelers
    'heinz field':                          (40.4468, -80.0158),   # Steelers legacy name
    'paycor stadium':                       (39.0955, -84.5160),   # Bengals
    'huntington bank field':                (41.5061, -81.6995),   # Browns
    'hard rock stadium':                    (25.9580, -80.2389),   # Dolphins
    'everbank stadium':                     (30.3240, -81.6373),   # Jaguars
    'tiaa bank field':                      (30.3240, -81.6373),   # Jaguars legacy name
    'nissan stadium':                       (36.1665, -86.7713),   # Titans
    'fedexfield':                           (38.9077, -76.8644),   # Commanders (FedExField)
    'northwest stadium':                    (38.9077, -76.8644),   # Commanders new name
    # ── MLB ──────────────────────────────────────────────────────────────────
    'yankee stadium':                       (40.8296, -73.9262),   # Yankees
    'fenway park':                          (42.3467, -71.0972),   # Red Sox
    'rogers centre':                        (43.6414, -79.3894),   # Blue Jays
    'oriole park at camden yards':          (39.2838, -76.6217),   # Orioles
    'camden yards':                         (39.2838, -76.6217),   # Orioles alt name
    'tropicana field':                      (27.7683, -82.6534),   # Rays
    'guaranteed rate field':                (41.8300, -87.6338),   # White Sox
    'progressive field':                    (41.4962, -81.6852),   # Guardians
    'comerica park':                        (42.3390, -83.0485),   # Tigers
    'kauffman stadium':                     (39.0514, -94.4803),   # Royals
    'target field':                         (44.9817, -93.2781),   # Twins
    'minute maid park':                     (29.7573, -95.3555),   # Astros
    'angel stadium':                        (33.8003, -117.8827),  # Angels
    't-mobile park':                        (47.5914, -122.3325),  # Mariners
    'globe life field':                     (32.7473, -97.0845),   # Rangers
    'citi field':                           (40.7571, -73.8458),   # Mets
    'citizens bank park':                   (39.9057, -75.1665),   # Phillies
    'nationals park':                       (38.8730, -77.0074),   # Nationals
    'loandepot park':                       (25.7781, -80.2197),   # Marlins
    'truist park':                          (33.8908, -84.4679),   # Braves
    'wrigley field':                        (41.9484, -87.6553),   # Cubs
    'busch stadium':                        (38.6226, -90.1928),   # Cardinals
    'american family field':                (43.0284, -87.9712),   # Brewers
    'great american ball park':             (39.0979, -84.5082),   # Reds
    'pnc park':                             (40.4469, -80.0057),   # Pirates
    'dodger stadium':                       (34.0739, -118.2400),  # Dodgers
    'oracle park':                          (37.7786, -122.3893),  # Giants
    'petco park':                           (32.7073, -117.1566),  # Padres
    'coors field':                          (39.7559, -104.9942),  # Rockies
    'chase field':                          (33.4453, -112.0667),  # Diamondbacks
    # ── NCAAF ────────────────────────────────────────────────────────────────
    'michigan stadium':                     (42.2659, -83.7487),   # Michigan
    'beaver stadium':                       (40.8121, -77.8564),   # Penn State
    'ohio stadium':                         (40.0017, -83.0197),   # Ohio State
    'kyle field':                           (30.6100, -96.3404),   # Texas A&M
    'neyland stadium':                      (35.9549, -83.9252),   # Tennessee
    'tiger stadium':                        (30.4121, -91.1837),   # LSU
    'bryant-denny stadium':                 (33.2084, -87.5503),   # Alabama
    'darrell k royal-texas memorial stadium': (30.2838, -97.7328), # Texas
    'darrell k royal–texas memorial stadium': (30.2838, -97.7328), # Texas alt dash
    'notre dame stadium':                   (41.6985, -86.2336),   # Notre Dame
    'memorial stadium':                     (40.8208, -96.7054),   # Nebraska
    'rose bowl stadium':                    (34.1614, -118.1676),  # UCLA / bowl
    'rose bowl':                            (34.1614, -118.1676),  # UCLA / bowl alt
    'jordan-hare stadium':                  (32.6024, -85.4917),   # Auburn
    'ben hill griffin stadium':             (29.6499, -82.3486),   # Florida
    'sanford stadium':                      (33.9498, -83.3733),   # Georgia
    'los angeles memorial coliseum':        (34.0141, -118.2879),  # USC
    'united airlines memorial coliseum':    (34.0141, -118.2879),  # USC alt name
    'autzen stadium':                       (44.0566, -123.0688),  # Oregon
    'doak campbell stadium':                (30.4388, -84.3070),   # Florida State
    'husky stadium':                        (47.6503, -122.3013),  # Washington
    'gaylord family oklahoma memorial stadium': (35.2058, -97.4456), # Oklahoma
    'camp randall stadium':                 (43.0699, -89.4117),   # Wisconsin
    'kroger field':                         (38.0228, -84.5028),   # Kentucky
    'vaught-hemingway stadium':             (34.3663, -89.5374),   # Ole Miss
    'davis wade stadium at scott field':    (33.4558, -88.7893),   # Mississippi State
    'folsom field':                         (40.0097, -105.2661),  # Colorado
    'carter-finley stadium':                (35.8033, -78.7222),   # NC State
    'kenan memorial stadium':               (35.9139, -79.0464),   # UNC
}

# Warm climate NFL teams that struggle in cold away venues
WARM_CLIMATE_NFL = frozenset({
    'Miami Dolphins', 'Tampa Bay Buccaneers', 'Los Angeles Rams',
    'Los Angeles Chargers', 'Las Vegas Raiders', 'Arizona Cardinals',
    'New Orleans Saints', 'Atlanta Falcons', 'Jacksonville Jaguars',
    'Carolina Panthers'
})

# High altitude venues - visiting teams perform measurably worse
HIGH_ALTITUDE_VENUES = frozenset({'Mile High Stadium', 'Empower Field at Mile High',
    'Empower Field', 'Ball Arena', 'Coors Field'})

# City coordinates for travel distance calculation (lat, lon)
CITY_COORDS: Dict[str, Tuple[float, float]] = {
    'los angeles': (34.0522, -118.2437), 'san francisco': (37.7749, -122.4194),
    'seattle': (47.6062, -122.3321), 'las vegas': (36.1699, -115.1398),
    'phoenix': (33.4484, -112.0740), 'denver': (39.7392, -104.9903),
    'dallas': (32.7767, -96.7970), 'houston': (29.7604, -95.3698),
    'chicago': (41.8781, -87.6298), 'minneapolis': (44.9778, -93.2650),
    'kansas city': (39.0997, -94.5786), 'new orleans': (29.9511, -90.0715),
    'new york': (40.7128, -74.0060), 'boston': (42.3601, -71.0589),
    'philadelphia': (39.9526, -75.1652), 'washington': (38.9072, -77.0369),
    'miami': (25.7617, -80.1918), 'atlanta': (33.7490, -84.3880),
    'charlotte': (35.2271, -80.8431), 'cleveland': (41.4993, -81.6944),
    'detroit': (42.3314, -83.0458), 'nashville': (36.1627, -86.7816),
    'indianapolis': (39.7684, -86.1581), 'cincinnati': (39.1031, -84.5120),
    'pittsburgh': (40.4406, -79.9959), 'baltimore': (39.2904, -76.6122),
    'buffalo': (42.8864, -78.8784), 'green bay': (44.5133, -88.0133),
    'tampa': (27.9506, -82.4572), 'jacksonville': (30.3322, -81.6557),
    'memphis': (35.1495, -90.0490), 'salt lake city': (40.7608, -111.8910),
    'portland': (45.5051, -122.6750), 'sacramento': (38.5816, -121.4944),
    'san antonio': (29.4241, -98.4936), 'oklahoma city': (35.4676, -97.5164),
    'toronto': (43.6532, -79.3832), 'milwaukee': (43.0389, -87.9065),
    'orlando': (28.5383, -81.3792), 'san diego': (32.7157, -117.1611),
    'minneapolis': (44.9778, -93.2650), 'raleigh': (35.7796, -78.6382),
    'columbus': (39.9612, -82.9988), 'winnipeg': (49.8951, -97.1384),
    'calgary': (51.0447, -114.0719), 'edmonton': (53.5461, -113.4937),
    'vancouver': (49.2827, -123.1207), 'ottawa': (45.4215, -75.6972),
    'montreal': (45.5017, -73.5673)
}

# City timezone offsets (UTC hours) - for body-clock travel penalty
CITY_TIMEZONES: Dict[str, int] = {
    'los angeles': -8, 'san francisco': -8, 'seattle': -8, 'las vegas': -8,
    'portland': -8, 'sacramento': -8, 'san diego': -8,
    'phoenix': -7, 'denver': -7, 'salt lake city': -7,
    'dallas': -6, 'houston': -6, 'chicago': -6, 'minneapolis': -6,
    'kansas city': -6, 'new orleans': -6, 'memphis': -6, 'milwaukee': -6,
    'san antonio': -6, 'oklahoma city': -6, 'nashville': -6, 'green bay': -6,
    'new york': -5, 'boston': -5, 'philadelphia': -5, 'washington': -5,
    'miami': -5, 'atlanta': -5, 'charlotte': -5, 'cleveland': -5,
    'detroit': -5, 'indianapolis': -5, 'cincinnati': -5, 'pittsburgh': -5,
    'baltimore': -5, 'buffalo': -5, 'tampa': -5, 'jacksonville': -5,
    'orlando': -5, 'raleigh': -5, 'columbus': -5,
    'toronto': -5, 'montreal': -5, 'ottawa': -5,
    'winnipeg': -6, 'calgary': -7, 'edmonton': -7, 'vancouver': -8
}

# Historical March Madness seed upset rates 1985-2024
# Key: (lower_seed_number, higher_seed_number), Value: probability lower seed wins
# Covers all possible first-round and later-round matchups
SEED_UPSET_RATES: Dict[Tuple[int, int], float] = {
    # First round (standard matchups)
    (1, 16): 0.993, (2, 15): 0.934, (3, 14): 0.847,
    (4, 13): 0.795, (5, 12): 0.647, (6, 11): 0.631,
    (7, 10): 0.602, (8, 9): 0.514,
    # Later-round matchups (computed from historical data)
    (1, 8): 0.840, (1, 9): 0.850, (2, 7): 0.730,
    (1, 5): 0.780, (1, 4): 0.740, (2, 3): 0.620,
    (1, 12): 0.930, (1, 13): 0.955, (2, 10): 0.870,
    (2, 11): 0.880, (3, 11): 0.760, (3, 10): 0.710,
    (4, 12): 0.710, (4, 5): 0.530, (3, 7): 0.660,
    (2, 6): 0.700, (1, 2): 0.680, (1, 3): 0.720,
    (1, 6): 0.790, (1, 7): 0.810, (2, 4): 0.650,
    (2, 5): 0.670, (3, 4): 0.560, (3, 5): 0.580,
    (3, 6): 0.610, (4, 6): 0.540, (4, 7): 0.560,
    (5, 6): 0.510, (5, 7): 0.510, (6, 7): 0.510,
}

# ESPN hidden API base
ESPN_BASE = 'https://site.api.espn.com/apis/site/v2/sports'

# ESPN sport paths
ESPN_PATHS: Dict[str, str] = {
    'nfl':    'football/nfl',
    'nba':    'basketball/nba',
    'mlb':    'baseball/mlb',
    'nhl':    'hockey/nhl',
    'ncaaf':  'football/college-football',
    'ncaabm': 'basketball/mens-college-basketball',
    'ncaabw': 'basketball/womens-college-basketball'
}

# TheSportsDB league IDs - historical fallback
TSDB_BASE = 'https://www.thesportsdb.com/api/v1/json/3'
TSDB_LEAGUE_IDS: Dict[str, str] = {
    'nfl': '4391', 'nba': '4387', 'mlb': '4424', 'nhl': '4380',
    'ncaaf': '4389', 'ncaabm': '4388', 'ncaabw': '4955'
}

# BallDontLie sport keys
BDN_BASE = 'https://api.balldontlie.io/v1'
BDN_SPORTS: Dict[str, str] = {
    'nba': 'nba', 'nfl': 'nfl', 'mlb': 'mlb', 'nhl': 'nhl'
}

# Odds API sport keys
ODDS_SPORT_KEYS: Dict[str, str] = {
    'nfl':    'americanfootball_nfl',
    'ncaaf':  'americanfootball_ncaaf',
    'nba':    'basketball_nba',
    'ncaabm': 'basketball_ncaab',
    'ncaabw': 'basketball_ncaawb',
    'mlb':    'baseball_mlb',
    'nhl':    'icehockey_nhl'
}

# All sports this program handles
ALL_SPORTS = list(ESPN_PATHS.keys())

# ESPN v2 Stats API — team-level stats (no key required)
ESPN_STATS_BASE = 'https://site.api.espn.com/apis/site/v2/sports'

# Model file name for persisting XGBoost models
XGB_MODEL_FILE = 'models.pkl'

# Minimum games in training buffer before triggering incremental XGBoost refit
XGB_BUFFER_MIN  = 50
# XGBoost hyperparameters
XGB_PARAMS = {
    'n_estimators':    500,
    'max_depth':       6,
    'learning_rate':   0.05,
    'subsample':       0.8,
    'colsample_bytree':0.8,
    'eval_metric':     'logloss',
    'random_state':    42,
    'use_label_encoder': False,
}

# Feature names — order must match build_feature_vector() exactly
XGB_FEATURE_NAMES_DEFAULT: List[str] = [
    'hfa', 'h2h_prob', 'venue_prob', 'dow_prob', 'form_diff',
    'pythag_diff', 'elo_diff', 'standings_diff', 'ff_diff',
    'efficiency_diff', 'sp_diff', 'turnover_diff', 'rest_adj',
    'bb_diff', 'travel_miles', 'tz_diff', 'altitude_flag', 'weather_drag',
    'injury_impact', 'ml_prob', 'spread_prob',
    'qb_rating_diff', 'net_rating_diff',
    'back_to_back_home', 'back_to_back_away',
    'efficiency_margin_diff', 'goalie_sv_pct_diff',
    'power_play_diff', 'park_factor', 'conference_strength_diff', 'sos_diff',
    'div_game',
]

# MLB stadium park factors (run-scoring multiplier vs league average 1.0)
MLB_PARK_FACTORS: Dict[str, float] = {
    'Coors Field': 1.20, 'Great American Ball Park': 1.10,
    'Fenway Park': 1.08, 'Wrigley Field': 1.07, 'Yankee Stadium': 1.04,
    'Globe Life Field': 1.03, 'American Family Field': 1.02,
    'Oracle Park': 0.93, 'Petco Park': 0.92, 'Tropicana Field': 0.94,
    'T-Mobile Park': 0.95, 'Dodger Stadium': 0.97, 'Chase Field': 0.96,
    'loanDepot park': 0.93, 'Kauffman Stadium': 0.96,
}

# NFL division membership — for division game signal
NFL_DIVISIONS: Dict[str, str] = {
    'New England Patriots': 'AFC East', 'Buffalo Bills': 'AFC East',
    'Miami Dolphins': 'AFC East', 'New York Jets': 'AFC East',
    'Baltimore Ravens': 'AFC North', 'Pittsburgh Steelers': 'AFC North',
    'Cleveland Browns': 'AFC North', 'Cincinnati Bengals': 'AFC North',
    'Tennessee Titans': 'AFC South', 'Houston Texans': 'AFC South',
    'Indianapolis Colts': 'AFC South', 'Jacksonville Jaguars': 'AFC South',
    'Kansas City Chiefs': 'AFC West', 'Las Vegas Raiders': 'AFC West',
    'Denver Broncos': 'AFC West', 'Los Angeles Chargers': 'AFC West',
    'Dallas Cowboys': 'NFC East', 'Philadelphia Eagles': 'NFC East',
    'New York Giants': 'NFC East', 'Washington Commanders': 'NFC East',
    'Chicago Bears': 'NFC North', 'Green Bay Packers': 'NFC North',
    'Minnesota Vikings': 'NFC North', 'Detroit Lions': 'NFC North',
    'Atlanta Falcons': 'NFC South', 'New Orleans Saints': 'NFC South',
    'Tampa Bay Buccaneers': 'NFC South', 'Carolina Panthers': 'NFC South',
    'Los Angeles Rams': 'NFC West', 'San Francisco 49ers': 'NFC West',
    'Seattle Seahawks': 'NFC West', 'Arizona Cardinals': 'NFC West',
}

# NCAAF major rivalry game pairs — (team_a, team_b)
NCAAF_RIVALRY_GAMES: List[Tuple[str, str]] = [
    ('Alabama Crimson Tide', 'Auburn Tigers'),
    ('Michigan Wolverines', 'Ohio State Buckeyes'),
    ('Oklahoma Sooners', 'Texas Longhorns'),
    ('Georgia Bulldogs', 'Florida Gators'),
    ('Notre Dame Fighting Irish', 'USC Trojans'),
    ('Army Black Knights', 'Navy Midshipmen'),
    ('Florida State Seminoles', 'Florida Gators'),
    ('Clemson Tigers', 'South Carolina Gamecocks'),
]

# Default signal weights - starting point before any training
DEFAULT_WEIGHTS: Dict[str, float] = {
    # Core signals (all sports)
    'h2h':              1.5,
    'hfa':              1.5,
    'venue':            1.2,
    'dow':              0.8,
    'form':             1.8,
    'pythag':           1.3,
    'rest':             1.0,
    'elo':              1.3,
    'sp':               1.4,
    'injury':           1.5,
    'weather':          1.0,
    'crowd':            1.0,
    'spread':           1.2,
    'standings':        1.1,
    'travel':           0.9,
    'timezone':         0.8,
    'efficiency':       1.4,
    'turnover':         1.2,
    'four_factors':     1.3,
    'altitude':         0.7,
    'bounce_back':      0.6,
    # NFL signals
    'qb_rating':        2.0,
    'red_zone':         1.3,
    'third_down':       1.2,
    'division_game':    0.9,
    'back_to_back':     1.1,
    # NBA signals
    'net_rating':       1.8,
    'pace':             0.9,
    'defensive_rating': 1.4,
    # MLB signals
    'park_factor':      0.8,
    # NHL signals
    'goalie_sv_pct':    1.9,
    'power_play':       1.3,
    'penalty_kill':     1.2,
    'corsi':            1.5,
    # NCAAF signals
    'conference_strength': 1.4,
    'rivalry_game':     0.8,
    # NCAAB signals
    'efficiency_margin':1.8,
    'sos':              1.2,
    'tempo':            0.9,
}

# ============================================================
# SECTION 4: DATA I/O
# All file system access is isolated here.
# ============================================================

def get_empty_data() -> dict:
    """
    Returns a complete empty data structure with every key that
    index.html and this file expect to find. When keys are added
    here, they appear on all installs after the next save/load cycle.
    """
    empty_rec = {'wins': 0, 'losses': 0}
    empty_w   = copy.deepcopy(DEFAULT_WEIGHTS)

    return {
        'version':            APP_VERSION,
        'last_trained':       None,
        'last_daily_update':  None,
        'training_status':    'untrained',
        'training_log':       [],
        'weights':            {s: copy.deepcopy(empty_w) for s in ALL_SPORTS},
        'calibration':        {s: {} for s in ALL_SPORTS},
        'closing_line_values':[],
        'patterns':           {'h2h': {}, 'venue': {}, 'dow': {}},
        'team_history':       {s: {} for s in ALL_SPORTS},
        'elo_ratings':        {s: {} for s in ALL_SPORTS},
        'pythagorean':        {s: {} for s in ALL_SPORTS},
        'team_stats':         {s: {} for s in ALL_SPORTS},
        'standings':          {s: {} for s in ALL_SPORTS},
        'predictions':        {s: [] for s in ALL_SPORTS},
        'cbs_picks': {
            'generated_at': None, 'week_number': None,
            'num_games': 0, 'max_possible_points': 0,
            'expected_points': 0, 'assignment': []
        },
        'march_mens': {
            'generated_at': None, 'simulations_run': 0,
            'recommended_bracket': [], 'upset_alerts': []
        },
        'march_womens': {
            'generated_at': None, 'simulations_run': 0,
            'recommended_bracket': [], 'upset_alerts': []
        },
        'betting_picks': {
            'generated_at': None,
            'safe_bet': None, 'value_bet': None,
            'longshot': None, 'parlay': None
        },
        'running_record': {s: copy.deepcopy(empty_rec) for s in ALL_SPORTS},
        'training_record': {
            'date': None, 'duration_seconds': None,
            'data_collected': {s: 0 for s in ALL_SPORTS},
            'accuracy': {s: 0.0 for s in ALL_SPORTS}
        },
        'espn_stats':    {s: {} for s in ALL_SPORTS},
        'barttorvik':    {},
        'sos':           {s: {} for s in ALL_SPORTS},
        'training_buffer': {s: [] for s in ALL_SPORTS},
        'xgb_trained':   False,
        'all_games_by_sport': {s: [] for s in ALL_SPORTS},
        # --- Intelligence layer keys (managed by mind.py, edge.py, scout.py) ---
        'mind': {
            'version': '1.0', 'last_trained': None, 'lstm_trained': False,
            'calibration_bins': {}, 'calibration_context': {},
            'lstm_normalization': {},
            'momentum_weights': {
                'nfl': 0.06, 'nba': 0.09, 'mlb': 0.05,
                'nhl': 0.07, 'ncaaf': 0.05, 'ncaabm': 0.08, 'ncaabw': 0.08
            },
            'performance': {}, 'error_log': [],
        },
        'edge': {
            'version': '1.0', 'last_updated': None,
            'line_history': {}, 'sharp_signals': {}, 'sentiment': {},
            'reporter_accounts': {
                'nfl':  ['adamschefter', 'rapsheet', 'TomPelissero'],
                'nba':  ['wojespn', 'ShamsCharania'],
                'mlb':  ['JeffPassan', 'Feinsand'],
                'nhl':  ['PierreVLeBrun', 'TSNBobMcKenzie'],
            },
            'public_teams': {
                'nfl':    ['dallas cowboys', 'new england patriots', 'green bay packers', 'kansas city chiefs'],
                'nba':    ['los angeles lakers', 'golden state warriors', 'boston celtics'],
                'mlb':    ['new york yankees', 'los angeles dodgers', 'chicago cubs'],
                'nhl':    ['toronto maple leafs', 'montreal canadiens'],
                'ncaabm': ['duke blue devils', 'kentucky wildcats', 'kansas jayhawks'],
                'ncaabw': [], 'ncaaf': [],
            },
            'rate_limit_state': {}, 'cache': {}, 'error_log': [],
        },
        'scout': {
            'version': '1.0', 'last_updated': None,
            'players': {},
            'officials': {s: {} for s in ALL_SPORTS},
            'team_style': {}, 'lineup_adjustments': {},
            'official_assignments': {}, 'team_stats_cache': {},
            'cache': {}, 'error_log': [],
        },
    }


def deep_merge(base: dict, override: dict) -> dict:
    """
    Merges override into base recursively. Keys in base that are
    missing from override are preserved. This ensures new keys
    added to get_empty_data() always exist after loading old data.
    """
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override if override is not None else base

    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_data() -> dict:
    """
    Loads data.json. Returns fresh empty structure on any failure.
    Merges saved data onto empty structure so all keys always exist.
    """
    empty = get_empty_data()

    if not os.path.exists(DATA_FILE):
        log.info('No data.json found - starting fresh')
        return empty

    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            raw = f.read()

        if not raw or not raw.strip():
            log.warning('data.json is empty - starting fresh')
            return empty

        saved = json.loads(raw)

        if not isinstance(saved, dict):
            log.warning('data.json has invalid structure - starting fresh')
            return empty

        merged = deep_merge(empty, saved)
        log.info(f'Data loaded (version: {merged.get("version", "unknown")})')
        return merged

    except json.JSONDecodeError as exc:
        log.error(f'data.json corrupted: {exc} - starting fresh')
        return empty
    except OSError as exc:
        log.error(f'Cannot read data.json: {exc} - starting fresh')
        return empty


def save_data(data: dict) -> bool:
    """
    Saves data dict to data.json using a temp file + atomic rename.
    Verifies the written file is valid JSON before replacing the real file.
    Returns True on success, False on any failure.
    """
    if not isinstance(data, dict):
        log.error('save_data: non-dict passed - refusing')
        return False

    data['last_saved'] = datetime.datetime.utcnow().isoformat()
    data['training_log'] = LOG_BUFFER[-500:]

    # Strip heavy training/model keys before writing to disk.
    # These are only needed in memory during a workflow run — not by the page.
    HEAVY_KEYS = {'training_buffer', 'all_games_by_sport', 'transformer',
                  'mind', 'graph', 'stack', 'explain', 'simulate_meta'}
    save_dict = {k: v for k, v in data.items() if k not in HEAVY_KEYS}

    temp_path = DATA_FILE + '.tmp'

    try:
        json_str = json.dumps(save_dict, ensure_ascii=False, indent=2, default=str)


        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(json_str)

        # Verify written file is valid before replacing real file
        with open(temp_path, 'r', encoding='utf-8') as f:
            json.loads(f.read())

        os.replace(temp_path, DATA_FILE)
        kb = os.path.getsize(DATA_FILE) / 1024
        log.info(f'Data saved ({kb:.1f} KB)')
        return True

    except Exception as exc:
        log.error(f'Failed to save data: {exc}')
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return False

# ============================================================
# SECTION 5: HTTP UTILITIES
# All network requests go through fetch_json or fetch_text.
# Nothing outside this section calls requests directly.
# ============================================================

BACKOFF_DELAYS = [2, 5, 15, 30]


def fetch_json(
    url: str,
    headers: Optional[Dict] = None,
    params: Optional[Dict] = None,
    max_retries: int = 4,
    timeout: int = 30
) -> Optional[dict]:
    """
    GETs a URL and returns parsed JSON or None.
    Never raises - all errors logged and None returned.
    - 429: wait 60s and retry
    - 5xx: retry with backoff
    - 4xx: do not retry
    - Timeout/connection: retry with backoff
    """
    if not url:
        log.warning('fetch_json called with empty URL')
        return None

    for attempt in range(max_retries):
        try:
            resp = requests.get(
                url,
                headers=headers or {},
                params=params or {},
                timeout=timeout
            )

            if resp.status_code == 200:
                try:
                    return resp.json()
                except json.JSONDecodeError:
                    log.warning(f'Invalid JSON from {url[:80]}')
                    return None

            if resp.status_code == 429:
                log.warning('Rate limited - waiting 60s')
                time.sleep(60)
                continue

            if resp.status_code >= 500 and attempt < max_retries - 1:
                delay = BACKOFF_DELAYS[min(attempt, len(BACKOFF_DELAYS) - 1)]
                log.warning(f'HTTP {resp.status_code} from {url[:60]} - retry in {delay}s')
                time.sleep(delay)
                continue

            log.warning(f'HTTP {resp.status_code} from {url[:60]}')
            return None

        except requests.Timeout:
            if attempt < max_retries - 1:
                time.sleep(BACKOFF_DELAYS[min(attempt, len(BACKOFF_DELAYS) - 1)])
                continue
            log.error(f'Timeout after {max_retries} attempts: {url[:60]}')
            return None

        except requests.ConnectionError as exc:
            if attempt < max_retries - 1:
                time.sleep(BACKOFF_DELAYS[min(attempt, len(BACKOFF_DELAYS) - 1)])
                continue
            log.error(f'Connection error: {exc}')
            return None

        except Exception as exc:
            log.error(f'Unexpected fetch error: {exc}')
            return None

    return None


def fetch_text(
    url: str,
    headers: Optional[Dict] = None,
    max_retries: int = 3,
    timeout: int = 60
) -> Optional[str]:
    """
    GETs a URL and returns raw text or None.
    Used for CSV files.
    """
    if not url:
        return None

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers or {}, timeout=timeout)

            if resp.status_code == 200:
                return resp.text

            if resp.status_code >= 500 and attempt < max_retries - 1:
                time.sleep(BACKOFF_DELAYS[min(attempt, len(BACKOFF_DELAYS) - 1)])
                continue

            log.warning(f'fetch_text HTTP {resp.status_code}: {url[:60]}')
            return None

        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(BACKOFF_DELAYS[min(attempt, len(BACKOFF_DELAYS) - 1)])
                continue
            log.error(f'fetch_text failed: {exc}')
            return None

    return None

# ============================================================
# SECTION 6: SAFE TYPE CONVERSIONS
# Prevents TypeErrors throughout the code.
# ============================================================

def safe_int(value: Any) -> Optional[int]:
    """Converts value to int. Returns None if impossible."""
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError, AttributeError):
        return None


def safe_float(value: Any) -> Optional[float]:
    """Converts to float. Returns None if impossible or NaN/Inf."""
    if value is None:
        return None
    try:
        result = float(str(value).strip())
        if math.isnan(result) or math.isinf(result):
            return None
        return result
    except (ValueError, TypeError, AttributeError):
        return None


def safe_date(value: Any) -> Optional[str]:
    """
    Attempts to parse a date string and return YYYY-MM-DD format.
    Returns None if parsing fails.
    """
    if not value:
        return None
    s = str(value).strip()
    # Already YYYY-MM-DD
    if len(s) >= 10 and s[4] == '-' and s[7] == '-':
        try:
            datetime.datetime.strptime(s[:10], '%Y-%m-%d')
            return s[:10]
        except ValueError:
            return None
    # Try ISO format with T separator
    try:
        return datetime.datetime.fromisoformat(s[:19]).strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return None


def clamp_prob(p: float) -> float:
    """Clamps probability to [0.05, 0.95] - never absolute 0 or 1."""
    if not isinstance(p, (int, float)) or math.isnan(p) or math.isinf(p):
        return 0.5
    return max(0.05, min(0.95, p))


def clamp_weight(w: float) -> float:
    """Clamps weight to [WEIGHT_MIN, WEIGHT_MAX]."""
    if not isinstance(w, (int, float)) or math.isnan(w) or math.isinf(w):
        return 1.0
    return max(WEIGHT_MIN, min(WEIGHT_MAX, w))

# ============================================================
# SECTION 7: ESPN DATA FETCHERS
# ESPN has a stable undocumented API that requires no key.
# ============================================================

def fetch_espn_scoreboard(sport: str) -> List[dict]:
    """
    Fetches upcoming/current games from ESPN scoreboard.
    Returns list of normalized game dicts.
    """
    path = ESPN_PATHS.get(sport)
    if not path:
        log.warning(f'fetch_espn_scoreboard: unknown sport {sport}')
        return []

    data = fetch_json(f'{ESPN_BASE}/{path}/scoreboard')
    if not data:
        return []

    games = []

    for event in data.get('events', []):
        if not isinstance(event, dict):
            continue

        comps = event.get('competitions', [])
        if not isinstance(comps, list) or not comps:
            continue

        comp = comps[0]
        if not isinstance(comp, dict):
            continue

        competitors = comp.get('competitors', [])
        if not isinstance(competitors, list) or len(competitors) < 2:
            continue

        home = next((c for c in competitors
                     if isinstance(c, dict) and c.get('homeAway') == 'home'), None)
        away = next((c for c in competitors
                     if isinstance(c, dict) and c.get('homeAway') == 'away'), None)

        if not home or not away:
            continue

        home_team = home.get('team') or {}
        away_team = away.get('team') or {}

        if not isinstance(home_team, dict):
            home_team = {}
        if not isinstance(away_team, dict):
            away_team = {}

        home_name = home_team.get('displayName', '').strip()
        away_name = away_team.get('displayName', '').strip()

        if not home_name or not away_name:
            continue

        venue_obj  = comp.get('venue') or {}
        venue_name = venue_obj.get('fullName', '') if isinstance(venue_obj, dict) else ''
        venue_addr = venue_obj.get('address') or {} if isinstance(venue_obj, dict) else {}

        if not isinstance(venue_addr, dict):
            venue_addr = {}

        # ESPN's address object contains city/state but not lat/lon.
        # Try ESPN first; fall back to VENUE_COORDS static lookup.
        _venue_lat = safe_float(venue_addr.get('latitude'))
        _venue_lon = safe_float(venue_addr.get('longitude'))
        if (_venue_lat is None or _venue_lon is None) and venue_name:
            _coords = VENUE_COORDS.get(venue_name.lower())
            if _coords:
                _venue_lat, _venue_lon = _coords

        home_rank = home.get('curatedRank') or {}
        away_rank = away.get('curatedRank') or {}

        games.append({
            'id':        str(event.get('id', '')),
            'home':      home_name,
            'away':      away_name,
            'home_id':   str(home_team.get('id', '')),
            'away_id':   str(away_team.get('id', '')),
            'date':      safe_date(event.get('date', '')) or '',
            'game_time': event.get('date', ''),  # Full ISO timestamp for primetime detection
            'venue':     venue_name,
            'venue_lat': _venue_lat,
            'venue_lon': _venue_lon,
            'home_seed': safe_int(home_rank.get('current')) or 0,
            'away_seed': safe_int(away_rank.get('current')) or 0,
            'status':    (event.get('status') or {}).get('type', {}).get('name', '')
        })

    log.info(f'ESPN scoreboard {sport}: {len(games)} games')
    return games


def fetch_espn_past_scores(sport: str, date_str: str) -> List[dict]:
    """
    Fetches completed game results for a specific date.
    date_str: YYYYMMDD format.
    Returns list of {home, away, home_score, away_score, home_won}.

    Uses sport-specific params (groups=50/49 for college basketball) to
    ensure ALL games are returned, not just featured matchups.
    """
    path = ESPN_PATHS.get(sport)
    if not path:
        return []

    if not date_str or len(date_str) != 8 or not date_str.isdigit():
        log.warning(f'fetch_espn_past_scores: invalid date_str "{date_str}"')
        return []

    # Use same sport-specific params as training to get complete results
    params: Dict[str, Any] = {'dates': date_str, 'limit': 500}
    sport_params = ESPN_SPORT_PARAMS.get(sport, {})
    for k, v in sport_params.items():
        if k != 'limit':
            params[k] = v

    data = fetch_json(f'{ESPN_BASE}/{path}/scoreboard', params=params)
    if not data:
        return []

    results = []

    for event in data.get('events', []):
        if not isinstance(event, dict):
            continue

        status = event.get('status') or {}
        if not isinstance(status, dict):
            continue

        completed = (status.get('type') or {}).get('completed', False)
        if not completed:
            continue

        comps = event.get('competitions', [])
        if not isinstance(comps, list) or not comps:
            continue

        comp = comps[0]
        if not isinstance(comp, dict):
            continue

        competitors = comp.get('competitors', [])
        if not isinstance(competitors, list) or len(competitors) < 2:
            continue

        home = next((c for c in competitors
                     if isinstance(c, dict) and c.get('homeAway') == 'home'), None)
        away = next((c for c in competitors
                     if isinstance(c, dict) and c.get('homeAway') == 'away'), None)

        if not home or not away:
            continue

        home_team = home.get('team') or {}
        away_team = away.get('team') or {}

        home_name  = home_team.get('displayName', '').strip() if isinstance(home_team, dict) else ''
        away_name  = away_team.get('displayName', '').strip() if isinstance(away_team, dict) else ''
        home_score = safe_int(home.get('score'))
        away_score = safe_int(away.get('score'))

        if not home_name or not away_name:
            continue
        if home_score is None or away_score is None:
            continue

        results.append({
            'home':       home_name,
            'away':       away_name,
            'home_score': home_score,
            'away_score': away_score,
            'home_won':   home_score > away_score
        })

    return results


def _names_match(a: str, b: str) -> bool:
    """
    Fuzzy team name match for auto-verification.
    Handles ESPN returning 'Kansas City' vs 'Kansas City Chiefs',
    or 'UNC' vs 'North Carolina Tar Heels', etc.
    Exact match first, then checks if one is a substring of the other.
    """
    if not a or not b:
        return False
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return True
    # One is a leading substring of the other (e.g. 'kansas city' in 'kansas city chiefs')
    if a in b or b in a:
        return True
    # Last word match (city name often sufficient: 'chiefs' in 'kansas city chiefs')
    a_last = a.split()[-1] if a.split() else ''
    b_last = b.split()[-1] if b.split() else ''
    if a_last and b_last and a_last == b_last and len(a_last) > 3:
        return True
    return False


def fetch_espn_standings(sport: str) -> Dict[str, dict]:
    """
    Fetches standings from ESPN including home/away split records.
    Returns dict of teamName -> standings dict.
    """
    path = ESPN_PATHS.get(sport)
    if not path:
        return {}

    data = fetch_json(f'{ESPN_BASE}/{path}/standings')
    if not data:
        return {}

    standings = {}

    # ESPN standings structure varies - try multiple known paths
    entries = (
        (data.get('standings') or {}).get('entries', [])
        or ((data.get('children') or [{}])[0].get('standings') or {}).get('entries', [])
        or []
    )

    if not isinstance(entries, list):
        return {}

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        team_obj  = entry.get('team') or {}
        team_name = team_obj.get('displayName', '').strip() if isinstance(team_obj, dict) else ''

        if not team_name:
            continue

        stat_map: Dict[str, Any] = {}
        for stat in (entry.get('stats') or []):
            if not isinstance(stat, dict):
                continue
            name  = stat.get('name', '')
            value = stat.get('value')
            if name and value is not None:
                stat_map[name] = value

        hw = safe_int(stat_map.get('homeWins', 0)) or 0
        hl = safe_int(stat_map.get('homeLosses', 0)) or 0
        aw = safe_int(stat_map.get('awayWins', 0)) or 0
        al = safe_int(stat_map.get('awayLosses', 0)) or 0

        home_total = hw + hl
        away_total = aw + al

        home_pct = (hw / home_total) if home_total > 0 else None
        away_pct = (aw / away_total) if away_total > 0 else None

        streak_raw = safe_int(stat_map.get('streak', 0)) or 0

        standings[team_name] = {
            'home_wins':    hw,
            'home_losses':  hl,
            'home_win_pct': home_pct,
            'away_wins':    aw,
            'away_losses':  al,
            'away_win_pct': away_pct,
            'overall_pct':  safe_float(stat_map.get('winPercent') or stat_map.get('winningPercentage')),
            'streak':       streak_raw
        }

    log.info(f'ESPN standings {sport}: {len(standings)} teams')
    return standings


def fetch_espn_injuries(team_id: str, sport: str) -> List[dict]:
    """
    Fetches injury report for a specific team.
    Returns list of {player, position, status, impact_estimate}.
    """
    if not team_id or not sport:
        return []

    path = ESPN_PATHS.get(sport)
    if not path:
        return []

    data = fetch_json(f'{ESPN_BASE}/{path}/teams/{team_id}/injuries', max_retries=2)
    if not data:
        return []

    # Impact estimates by position - how much a starter out at this position hurts
    POSITION_IMPACT: Dict[str, float] = {
        'QB': 0.16, 'RB': 0.08, 'WR': 0.06, 'TE': 0.05,
        'K': 0.02, 'OL': 0.04, 'DL': 0.05, 'LB': 0.05,
        'CB': 0.05, 'S': 0.04,
        'PG': 0.13, 'SG': 0.10, 'SF': 0.10, 'PF': 0.09, 'C': 0.11,
        'SP': 0.15, 'RP': 0.07,
        '1B': 0.06, '2B': 0.06, '3B': 0.06, 'SS': 0.07, 'OF': 0.05,
        'G': 0.20, 'D': 0.10, 'LW': 0.08, 'RW': 0.08
    }

    injuries = []

    for inj in (data.get('injuries') or []):
        if not isinstance(inj, dict):
            continue

        athlete = inj.get('athlete') or {}
        if not isinstance(athlete, dict):
            continue

        pos_obj = athlete.get('position') or {}
        position = pos_obj.get('abbreviation', '') if isinstance(pos_obj, dict) else ''
        status = inj.get('status', 'Questionable')

        injuries.append({
            'player':  athlete.get('displayName', 'Unknown'),
            'position': position,
            'status':   status,
            'impact':   POSITION_IMPACT.get(position, 0.04)
        })

    return injuries

# ============================================================
# SECTION 8: HISTORICAL GAME DATA — MULTI-SOURCE
#
# Sources ranked by depth and reliability:
#
#  NFL   : nflverse GitHub CSV (1999–present, single download, complete)
#  NCAAF : College Football Data API (2000–present, free key via CFBD_KEY)
#  ALL   : ESPN date-range scoreboard (2002–present, no key, reliable)
#  ALL   : TheSportsDB (fallback only — free tier is severely limited)
#
# fetch_season() is the single entry point called by training.
# It automatically selects the best available source per sport.
# ============================================================

# Season date windows — (MM-DD start, MM-DD end)
# For cross-year sports, start is year Y and end is year Y+1.
SEASON_WINDOWS: Dict[str, Tuple[str, str]] = {
    'nfl':    ('09-01', '02-15'),
    'nba':    ('10-01', '06-30'),
    'mlb':    ('03-20', '10-31'),
    'nhl':    ('10-01', '06-30'),
    'ncaaf':  ('08-25', '01-20'),
    'ncaabm': ('11-01', '04-15'),
    'ncaabw': ('11-01', '04-15'),
}
CROSS_YEAR_SPORTS = frozenset({'nfl', 'nba', 'nhl', 'ncaaf', 'ncaabm', 'ncaabw'})



def fetch_season(sport: str, year: int) -> List[dict]:
    """
    Main entry point for historical data.
    Tries the best source for each sport, falls back down the chain.
    Always returns a list of completed game dicts with home/away/scores.
    """
    if not sport or year < 1999 or year > CURRENT_YEAR:
        return []

    # NFL: nflverse is definitive — 1999-present, complete, single CSV download
    if sport == 'nfl':
        games = _fetch_nflverse(year)
        if games:
            return games
        log.warning(f'nflverse failed for NFL {year} — trying ESPN')

    # All sports: ESPN date-range queries (2002–present, no key)
    games = _fetch_espn_season(sport, year)
    if games:
        return games

    # Last resort: TheSportsDB (free tier, very limited)
    log.warning(f'ESPN returned 0 for {sport} {year} — trying TSDB fallback')
    return _fetch_tsdb(sport, year)


def _season_date_range(sport: str, year: int) -> Tuple['datetime.date', 'datetime.date']:
    """Returns (start, end) dates for a sport/season year."""
    start_str, end_str = SEASON_WINDOWS.get(sport, ('01-01', '12-31'))
    start = datetime.date(year, int(start_str[:2]), int(start_str[3:]))
    end_year = year + 1 if sport in CROSS_YEAR_SPORTS else year
    end = datetime.date(end_year, int(end_str[:2]), int(end_str[3:]))
    today_minus_1 = datetime.date.today() - datetime.timedelta(days=1)
    return start, min(end, today_minus_1)


# ── SOURCE 1: nflverse ───────────────────────────────────────────────────────
# Complete NFL game data 1999-present. Single CSV, no key, no pagination.
# URL: raw GitHub CSV updated throughout every season.

NFLVERSE_CSV_URL = (
    'https://raw.githubusercontent.com/nflverse/nflverse-data/master/data/games.csv'
)
# Backup URL in case primary moves
NFLVERSE_CSV_URL_BACKUP = (
    'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
)

# Cache parsed nflverse data in memory so we don't re-download mid-training
_NFLVERSE_CACHE: Optional[List[dict]] = None


def _fetch_nflverse(year: int) -> List[dict]:
    """
    Downloads the nflverse games.csv and returns games for a given season year.
    Caches the full CSV in memory so subsequent years don't re-download.
    """
    global _NFLVERSE_CACHE

    if _NFLVERSE_CACHE is None:
        log.info('Downloading nflverse games.csv (covers 1999–present)...')
        text = fetch_text(NFLVERSE_CSV_URL, timeout=120)
        if not text:
            log.warning('nflverse primary URL failed - trying backup URL')
            text = fetch_text(NFLVERSE_CSV_URL_BACKUP, timeout=120)
        if not text:
            log.warning('nflverse CSV download failed on both URLs')
            _NFLVERSE_CACHE = []
            return []
        _NFLVERSE_CACHE = _parse_nflverse_csv(text)
        log.info(f'nflverse loaded: {len(_NFLVERSE_CACHE)} total NFL games')

    games = [g for g in _NFLVERSE_CACHE if isinstance(g, dict) and g.get('season_year') == year]
    log.info(f'nflverse NFL {year}: {len(games)} games')
    return games


def _parse_nflverse_csv(text: str) -> List[dict]:
    """
    Parses nflverse games.csv into normalized game dicts.
    Relevant columns: season, game_type, gameday, home_team, away_team,
                      home_score, away_score, location, stadium
    Only includes regular season and playoff games with final scores.
    """
    if not text:
        return []

    lines = text.strip().split('\n')
    if len(lines) < 2:
        return []

    header = [h.strip().strip('"') for h in lines[0].split(',')]

    def col(row: list, name: str) -> str:
        try:
            idx = header.index(name)
            return row[idx].strip().strip('"') if idx < len(row) else ''
        except (ValueError, IndexError):
            return ''

    # NFL team abbreviation to full name
    NFL_ABV: Dict[str, str] = {
        'ARI':'Arizona Cardinals','ATL':'Atlanta Falcons','BAL':'Baltimore Ravens',
        'BUF':'Buffalo Bills','CAR':'Carolina Panthers','CHI':'Chicago Bears',
        'CIN':'Cincinnati Bengals','CLE':'Cleveland Browns','DAL':'Dallas Cowboys',
        'DEN':'Denver Broncos','DET':'Detroit Lions','GB':'Green Bay Packers',
        'HOU':'Houston Texans','IND':'Indianapolis Colts','JAX':'Jacksonville Jaguars',
        'KC':'Kansas City Chiefs','LAC':'Los Angeles Chargers','LA':'Los Angeles Rams',
        'LAR':'Los Angeles Rams','LV':'Las Vegas Raiders','MIA':'Miami Dolphins',
        'MIN':'Minnesota Vikings','NE':'New England Patriots','NO':'New Orleans Saints',
        'NYG':'New York Giants','NYJ':'New York Jets','OAK':'Las Vegas Raiders',
        'PHI':'Philadelphia Eagles','PIT':'Pittsburgh Steelers','SD':'Los Angeles Chargers',
        'SEA':'Seattle Seahawks','SF':'San Francisco 49ers','STL':'Los Angeles Rams',
        'TB':'Tampa Bay Buccaneers','TEN':'Tennessee Titans',
        'WAS':'Washington Commanders','WSH':'Washington Commanders'
    }

    games = []

    for line in lines[1:]:
        if not line.strip():
            continue

        # Handle quoted fields with commas inside them
        try:
            import csv as _csv
            row = next(_csv.reader([line]))
        except Exception:
            row = line.split(',')

        game_type = col(row, 'game_type').upper()
        if game_type not in ('REG', 'POST', 'WC', 'DIV', 'CON', 'SB'):
            continue

        season = safe_int(col(row, 'season'))
        if season is None:
            continue

        home_abv   = col(row, 'home_team')
        away_abv   = col(row, 'away_team')
        home_score = safe_int(col(row, 'home_score'))
        away_score = safe_int(col(row, 'away_score'))
        gameday    = safe_date(col(row, 'gameday'))
        stadium    = col(row, 'stadium')
        location   = col(row, 'location')

        if not home_abv or not away_abv:
            continue
        if home_score is None or away_score is None:
            continue
        if not gameday:
            continue

        home_name = NFL_ABV.get(home_abv, home_abv)
        away_name = NFL_ABV.get(away_abv, away_abv)

        # nflverse marks neutral site games with location == 'Neutral'
        # Treat neutral site as home=listed home (e.g. Super Bowl host city)
        games.append({
            'id':          f'nflverse_{season}_{home_abv}_{away_abv}',
            'home':        home_name,
            'away':        away_name,
            'date':        gameday,
            'venue':       stadium,
            'home_score':  home_score,
            'away_score':  away_score,
            'home_won':    home_score > away_score,
            'margin':      home_score - away_score,
            'season_year': season,
            'is_neutral':  location == 'Neutral'
        })

    return games




# ── SOURCE 3: ESPN date-range scoreboard ─────────────────────────────────────
# Works for all 7 sports. Reliable back to 2002–2005.
# Iterates through the season in 7-day chunks.

# ESPN requires sport-specific parameters to get complete data.
# College basketball: 'groups' param selects all D1 games.
#   groups=50 → all men's D1,  groups=49 → all women's D1
# Without 'groups', ESPN returns only a small subset (top-25 teams).
# College basketball also has many simultaneous games (100+/day),
# so we use 1-day chunks with limit=500 to avoid missing games.
ESPN_SPORT_PARAMS: Dict[str, dict] = {
    'ncaabm': {'groups': '50', 'limit': 500},
    'ncaabw': {'groups': '4955', 'limit': 500},
}
ESPN_CHUNK_DAYS: Dict[str, int] = {
    'ncaabm': 1,   # 100+ games/day — must go day-by-day
    'ncaabw': 1,   # same
    'mlb':    3,   # ~45 games per 3 days, safe with limit=500
}


def _fetch_espn_season(sport: str, year: int) -> List[dict]:
    """
    Fetches a full season from ESPN by querying date chunks.
    Uses sport-specific parameters to ensure complete coverage.

    College basketball uses day-by-day queries with groups param —
    without these ESPN returns only a fraction of D1 games.
    Other sports use 7-day chunks which is fast and complete.
    """
    path = ESPN_PATHS.get(sport)
    if not path:
        return []

    start_date, end_date = _season_date_range(sport, year)
    if start_date > end_date:
        return []

    # Sport-specific chunk size and extra params
    chunk_days  = ESPN_CHUNK_DAYS.get(sport, 7)
    extra_params = ESPN_SPORT_PARAMS.get(sport, {})
    base_limit  = extra_params.get('limit', 500)

    games: List[dict] = []
    seen_ids: set     = set()
    api_calls         = 0
    current           = start_date

    while current <= end_date:
        chunk_end  = min(current + datetime.timedelta(days=chunk_days - 1), end_date)
        date_param = current.strftime('%Y%m%d') + '-' + chunk_end.strftime('%Y%m%d')

        params = {'dates': date_param, 'limit': base_limit}
        params.update({k: v for k, v in extra_params.items() if k != 'limit'})

        data = fetch_json(
            f'{ESPN_BASE}/{path}/scoreboard',
            params=params,
            max_retries=3,
            timeout=25
        )
        api_calls += 1

        if data:
            for g in _parse_espn_response(data):
                key = g.get('id') or f"{g['home']}|{g['away']}|{g['date']}"
                if key not in seen_ids:
                    seen_ids.add(key)
                    games.append(g)

        current += datetime.timedelta(days=chunk_days)

        # Polite rate limiting — college bball has many more API calls so space them out
        if sport in ('ncaabm', 'ncaabw'):
            time.sleep(0.4 if api_calls % 10 != 0 else 2.0)
        else:
            time.sleep(0.3 if api_calls % 5 != 0 else 1.0)

    log.info(f'ESPN {sport} {year}: {len(games)} completed games ({api_calls} API calls)')
    return games


def _parse_espn_response(data: dict) -> List[dict]:
    """
    Parses an ESPN scoreboard API response.
    Returns only completed games with valid home/away scores.
    """
    if not isinstance(data, dict):
        return []

    results = []

    for event in data.get('events', []):
        if not isinstance(event, dict):
            continue

        status = (event.get('status') or {})
        if not isinstance(status, dict):
            continue
        if not (status.get('type') or {}).get('completed', False):
            continue

        comps = event.get('competitions', [])
        if not isinstance(comps, list) or not comps:
            continue

        comp = comps[0]
        if not isinstance(comp, dict):
            continue

        competitors = comp.get('competitors', [])
        if not isinstance(competitors, list) or len(competitors) < 2:
            continue

        home = next((c for c in competitors if isinstance(c, dict) and c.get('homeAway') == 'home'), None)
        away = next((c for c in competitors if isinstance(c, dict) and c.get('homeAway') == 'away'), None)

        if not home or not away:
            continue

        home_team = home.get('team') or {}
        away_team = away.get('team') or {}
        if not isinstance(home_team, dict): home_team = {}
        if not isinstance(away_team, dict): away_team = {}

        home_name  = home_team.get('displayName', '').strip()
        away_name  = away_team.get('displayName', '').strip()
        home_score = safe_int(home.get('score'))
        away_score = safe_int(away.get('score'))

        if not home_name or not away_name or home_score is None or away_score is None:
            continue

        venue_obj  = comp.get('venue') or {}
        venue_name = venue_obj.get('fullName', '') if isinstance(venue_obj, dict) else ''

        results.append({
            'id':         str(event.get('id', '')),
            'home':       home_name,
            'away':       away_name,
            'date':       safe_date(event.get('date', '')) or '',
            'venue':      venue_name,
            'home_score': home_score,
            'away_score': away_score,
            'home_won':   home_score > away_score,
            'margin':     home_score - away_score
        })

    return results


# ── SOURCE 4: TheSportsDB fallback ───────────────────────────────────────────
# Last resort only. Free tier is capped at ~100 records per request.

def _fetch_tsdb(sport: str, year: int) -> List[dict]:
    """Last-resort fallback to TheSportsDB. Very limited on free tier."""
    league_id = TSDB_LEAGUE_IDS.get(sport)
    if not league_id:
        return []

    data = fetch_json(f'{TSDB_BASE}/eventsseason.php?id={league_id}&s={year}')
    if not data:
        return []

    events = data.get('events') or []
    if not isinstance(events, list):
        return []

    games = []
    for event in events:
        if not isinstance(event, dict):
            continue
        home  = str(event.get('strHomeTeam', '') or '').strip()
        away  = str(event.get('strAwayTeam', '') or '').strip()
        date  = safe_date(event.get('dateEvent', ''))
        hs    = safe_int(event.get('intHomeScore'))
        as_   = safe_int(event.get('intAwayScore'))
        venue = str(event.get('strVenue', '') or '').strip()
        if not home or not away or hs is None or as_ is None or not date:
            continue
        if year == CURRENT_YEAR and date > YESTERDAY:
            continue
        games.append({'id': str(event.get('idEvent', '')), 'home': home, 'away': away,
                      'date': date, 'venue': venue, 'home_score': hs, 'away_score': as_,
                      'home_won': hs > as_, 'margin': hs - as_})

    log.info(f'TSDB fallback {sport} {year}: {len(games)} games')
    return games


# Backward-compat aliases used throughout the rest of the file
def fetch_espn_season(sport: str, year: int) -> List[dict]:
    return fetch_season(sport, year)

def fetch_tsdb_season(sport: str, year: int) -> List[dict]:
    return fetch_season(sport, year)

# ============================================================
# SECTION 9: BALLDONTLIE DATA FETCHER
# Box score stats for NBA, NFL, MLB, NHL.
# Requires free API key - provides turnovers, efficiency, etc.
# ============================================================

def fetch_bdn_team_stats(sport: str, season: int) -> Dict[str, dict]:
    """
    Fetches team season averages from BallDontLie for a given sport and season.
    Returns dict of teamName -> stats dict.
    Provides real box score data not available from TSDB.
    """
    if not BALLDONTLIE_KEY:
        return {}

    bdn_sport = BDN_SPORTS.get(sport)
    if not bdn_sport:
        return {}

    headers = {'Authorization': BALLDONTLIE_KEY}

    data = fetch_json(
        f'{BDN_BASE}/{bdn_sport}/season_averages',
        headers=headers,
        params={'season': season},
        max_retries=3
    )

    if not data:
        return {}

    stats = {}
    raw_list = data.get('data') or []

    if not isinstance(raw_list, list):
        return {}

    for entry in raw_list:
        if not isinstance(entry, dict):
            continue

        team_obj  = entry.get('team') or {}
        team_name = team_obj.get('full_name', '').strip() if isinstance(team_obj, dict) else ''

        if not team_name:
            continue

        stats[team_name] = {
            'pts':       safe_float(entry.get('pts')),
            'reb':       safe_float(entry.get('reb')),
            'ast':       safe_float(entry.get('ast')),
            'stl':       safe_float(entry.get('stl')),
            'blk':       safe_float(entry.get('blk')),
            'turnover':  safe_float(entry.get('turnover')),
            'fg_pct':    safe_float(entry.get('fg_pct')),
            'fg3_pct':   safe_float(entry.get('fg3_pct')),
            'ft_pct':    safe_float(entry.get('ft_pct')),
            'oreb':      safe_float(entry.get('oreb')),
            'dreb':      safe_float(entry.get('dreb'))
        }

    log.info(f'BallDontLie {sport} {season}: {len(stats)} teams')
    return stats


def fetch_bdn_games(sport: str, start_date: str, end_date: str) -> List[dict]:
    """
    Fetches games with box score data from BallDontLie for a date range.
    start_date, end_date: YYYY-MM-DD format.
    Returns list of game dicts with scores and stats.
    """
    if not BALLDONTLIE_KEY:
        return []

    bdn_sport = BDN_SPORTS.get(sport)
    if not bdn_sport:
        return []

    headers = {'Authorization': BALLDONTLIE_KEY}
    games   = []
    cursor  = None

    # BallDontLie uses cursor-based pagination
    while True:
        params: Dict[str, Any] = {
            'start_date': start_date,
            'end_date':   end_date,
            'per_page':   100
        }
        if cursor:
            params['cursor'] = cursor

        data = fetch_json(
            f'{BDN_BASE}/{bdn_sport}/games',
            headers=headers,
            params=params,
            max_retries=3
        )

        if not data:
            break

        page_games = data.get('data') or []
        if not isinstance(page_games, list):
            break

        for game in page_games:
            if not isinstance(game, dict):
                continue

            home_team = game.get('home_team') or {}
            away_team = game.get('visitor_team') or {}

            if not isinstance(home_team, dict) or not isinstance(away_team, dict):
                continue

            home_name  = home_team.get('full_name', '').strip()
            away_name  = away_team.get('full_name', '').strip()
            home_score = safe_int(game.get('home_team_score'))
            away_score = safe_int(game.get('visitor_team_score'))
            date_str   = safe_date(game.get('date', ''))

            if not home_name or not away_name:
                continue
            if home_score is None or away_score is None:
                continue
            if not date_str:
                continue

            games.append({
                'id':         str(game.get('id', '')),
                'home':       home_name,
                'away':       away_name,
                'date':       date_str,
                'home_score': home_score,
                'away_score': away_score,
                'home_won':   home_score > away_score,
                'margin':     home_score - away_score,
                'venue':      ''
            })

        # Check for next page
        meta = data.get('meta') or {}
        if not isinstance(meta, dict):
            break

        next_cursor = meta.get('next_cursor')
        if not next_cursor:
            break

        cursor = next_cursor

    return games

# ============================================================
# SECTION 9B: ESPN STATS API + BARTTORVIK FETCHERS
# ESPN v2 stats API provides QB rating, red zone%, goalie SV%, etc.
# Barttorvik provides NCAAB adjusted efficiency margin, tempo, SOS.
# Both are free, no API key required.
# ============================================================

def fetch_espn_stats(sport: str) -> Dict[str, dict]:
    """
    Fetches team-level stats from ESPN's v2 stats API.
    Returns dict of team_name -> stats dict.
    Covers: NFL (QB rating, red zone, 3rd down), NBA (net rating, pace),
            NHL (goalie SV%, PP%, PK%), MLB (ERA, WHIP per team).
    """
    path = ESPN_PATHS.get(sport)
    if not path:
        return {}

    url  = f'{ESPN_STATS_BASE}/{path}/statistics'
    data = fetch_json(url, params={'limit': 100}, max_retries=2)
    if not data:
        return {}

    stats: Dict[str, dict] = {}

    # ESPN v2 stats can come back in different shapes — handle both
    results = data.get('results') or data
    if isinstance(results, dict):
        # Try team-level stats array
        teams_list = results.get('teams') or results.get('athletes') or []
    elif isinstance(results, list):
        teams_list = results
    else:
        teams_list = []

    for entry in teams_list:
        if not isinstance(entry, dict):
            continue

        team_obj = entry.get('team') or entry
        if not isinstance(team_obj, dict):
            continue

        name = team_obj.get('displayName') or team_obj.get('name', '')
        name = name.strip()
        if not name:
            continue

        stat_map: Dict[str, Any] = {}

        # Flatten stats from various ESPN response structures
        for stat_block in (entry.get('stats') or entry.get('statistics') or []):
            if not isinstance(stat_block, dict):
                continue
            n = stat_block.get('name', '')
            v = stat_block.get('value')
            if n and v is not None:
                stat_map[n] = v

        # Also check nested 'categories'
        for cat in (entry.get('categories') or []):
            if not isinstance(cat, dict):
                continue
            for stat in (cat.get('stats') or cat.get('statistics') or []):
                if not isinstance(stat, dict):
                    continue
                n = stat.get('name', '')
                v = stat.get('value')
                if n and v is not None:
                    stat_map[n] = v

        if stat_map:
            stats[name] = stat_map

    log.info(f'ESPN stats {sport}: {len(stats)} teams')
    return stats


def _get_espn_stat(espn_stats: dict, sport: str, team: str, stat_name: str) -> Optional[float]:
    """
    Safely retrieves a single stat value from espn_stats dict.
    Returns None if not available.
    """
    if not isinstance(espn_stats, dict):
        return None
    sport_stats = espn_stats.get(sport) or {}
    team_stats  = sport_stats.get(team) or {}
    return safe_float(team_stats.get(stat_name))


def fetch_barttorvik(year: Optional[int] = None) -> Dict[str, dict]:
    """
    Fetches NCAAB team efficiency data from Barttorvik (free, no key).
    Returns dict of team_name -> {adj_em, tempo, adj_oe, adj_de, sos}.
    adj_em = adjusted efficiency margin (KenPom equivalent).
    """
    yr  = year or CURRENT_YEAR
    url = f'https://barttorvik.com/trank.php?year={yr}&json=1'
    data = fetch_json(url, max_retries=2, timeout=20)

    result: Dict[str, dict] = {}

    # Barttorvik returns a list of team arrays
    if not isinstance(data, list):
        log.warning(f'Barttorvik {yr}: unexpected response format')
        return result

    for row in data:
        if not isinstance(row, list) or len(row) < 10:
            continue
        try:
            # Barttorvik columns (approximate — verified against live API):
            # [0]=team, [1]=conf, [2]=record, [3]=adj_oe, [4]=adj_de,
            # [5]=adj_em, [6]=sos, [7]=opp_adj_em, [8]=ncsos, [9]=tempo
            name    = str(row[0]).strip()
            adj_oe  = safe_float(row[3])
            adj_de  = safe_float(row[4])
            adj_em  = safe_float(row[5])
            sos     = safe_float(row[6])
            tempo   = safe_float(row[9]) if len(row) > 9 else None
            if name:
                result[name] = {
                    'adj_em':  adj_em,
                    'adj_oe':  adj_oe,
                    'adj_de':  adj_de,
                    'sos':     sos,
                    'tempo':   tempo,
                }
        except (IndexError, TypeError):
            continue

    log.info(f'Barttorvik {yr}: {len(result)} teams')
    return result


def compute_sos(sport: str, team_history: dict) -> Dict[str, float]:
    """
    Computes strength of schedule for each team from historical results.
    SOS = opponent win% weighted by opponent's opponent win%.
    Returns dict of team_name -> sos_score (higher = harder schedule).
    """
    sport_hist = team_history.get(sport) or {}
    if not isinstance(sport_hist, dict):
        return {}

    # First pass: compute win% for every team
    win_pct: Dict[str, float] = {}
    for team, games in sport_hist.items():
        if not isinstance(games, list):
            continue
        wins   = sum(1 for g in games if isinstance(g, dict) and g.get('home_won') and g.get('is_home'))
        wins  += sum(1 for g in games if isinstance(g, dict) and not g.get('home_won') and not g.get('is_home'))
        total  = len(games)
        if total > 0:
            win_pct[team] = wins / total

    # Second pass: for each team, SOS = average opponent win%
    sos: Dict[str, float] = {}
    for team, games in sport_hist.items():
        if not isinstance(games, list):
            continue
        opp_pcts = []
        for g in games:
            if not isinstance(g, dict):
                continue
            opp = g.get('opponent', '')
            if opp and opp in win_pct:
                opp_pcts.append(win_pct[opp])
        if opp_pcts:
            sos[team] = sum(opp_pcts) / len(opp_pcts)

    return sos


def is_back_to_back(team: str, sport: str, game_date: str, team_history: dict) -> bool:
    """
    Returns True if the team played a game yesterday (back-to-back situation).
    Relevant primarily for NBA and NHL.
    """
    if not game_date or len(game_date) < 10:
        return False
    try:
        gd  = datetime.datetime.strptime(game_date[:10], '%Y-%m-%d').date()
        yd  = gd - datetime.timedelta(days=1)
        yd_str = yd.strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return False

    sport_hist = team_history.get(sport) or {}
    games = sport_hist.get(team) or []
    return any(
        isinstance(g, dict) and g.get('date', '')[:10] == yd_str
        for g in games
    )


def is_division_game(home: str, away: str) -> bool:
    """Returns True if both NFL teams are in the same division."""
    hd = NFL_DIVISIONS.get(home)
    ad = NFL_DIVISIONS.get(away)
    return bool(hd and ad and hd == ad)


def is_rivalry_game(home: str, away: str) -> bool:
    """Returns True if this is a known NCAAF rivalry matchup."""
    for a, b in NCAAF_RIVALRY_GAMES:
        if (home == a and away == b) or (home == b and away == a):
            return True
    return False


def get_park_factor(venue: str) -> float:
    """Returns MLB park factor for a given venue (1.0 = neutral)."""
    if not venue:
        return 1.0
    # Exact match first, then partial
    pf = MLB_PARK_FACTORS.get(venue)
    if pf is not None:
        return pf
    for park, factor in MLB_PARK_FACTORS.items():
        if park.lower() in venue.lower() or venue.lower() in park.lower():
            return factor
    return 1.0


def is_primetime_game(game_time: str, game_date: str) -> bool:
    """
    Returns True if this is an NFL primetime game (MNF, TNF, SNF).
    Primetime = kickoff at or after 7 PM ET on weekdays or Sunday.
    """
    if not game_time:
        return False
    try:
        # ESPN ISO timestamp is UTC; primetime starts at ~23:00 UTC = 7 PM ET
        dt = datetime.datetime.strptime(game_time[:19], '%Y-%m-%dT%H:%M:%S')
        return dt.hour >= 23 or dt.hour <= 2  # 11 PM UTC = 7 PM ET
    except (ValueError, TypeError):
        return False

# ============================================================
# SECTION 10: INTERNAL ELO RATINGS
# Builds Elo ratings from historical game data collected during training.
# FiveThirtyEight shut down in 2023 — their data is frozen. This replaces
# the stale 538 CSV fetch with an always-current internal calculation.
# Covers ALL 7 sports (538 only had NFL and NBA).
# ============================================================

# K-factors per sport — how aggressively to update ratings after each game
ELO_K_FACTORS: Dict[str, float] = {
    'nfl':    20.0,
    'nba':    15.0,
    'mlb':    10.0,
    'nhl':    12.0,
    'ncaaf':  24.0,
    'ncaabm': 24.0,
    'ncaabw': 24.0,
}

ELO_DEFAULT_RATING = 1500.0  # Starting Elo for any team not yet seen


def build_elo_from_games(games: List[dict], sport: str,
                          existing_elo: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Builds/updates Elo ratings from a list of games in chronological order.
    Each game updates the Elo ratings of both teams using the standard formula.

    existing_elo: pass in ratings from prior years to continue building on them.
    Returns updated dict of team_name -> elo_rating.
    """
    if not games:
        return existing_elo.copy() if existing_elo else {}

    elo: Dict[str, float] = existing_elo.copy() if isinstance(existing_elo, dict) else {}
    k = ELO_K_FACTORS.get(sport, 15.0)

    # Sort by date to ensure chronological processing
    sorted_games = sorted(games, key=lambda g: g.get('date', ''))

    for game in sorted_games:
        home = game.get('home', '')
        away = game.get('away', '')
        home_won = game.get('home_won')

        if not home or not away or home_won is None:
            continue

        home_elo = elo.get(home, ELO_DEFAULT_RATING)
        away_elo = elo.get(away, ELO_DEFAULT_RATING)

        # Expected scores using standard Elo formula
        expected_home = 1.0 / (1.0 + math.pow(10.0, (away_elo - home_elo) / 400.0))
        expected_away = 1.0 - expected_home

        actual_home = 1.0 if home_won else 0.0
        actual_away = 1.0 - actual_home

        elo[home] = home_elo + k * (actual_home - expected_home)
        elo[away] = away_elo + k * (actual_away - expected_away)

    return elo

# ============================================================
# SECTION 10B: XGBOOST MODEL MANAGEMENT
# Trains one XGBClassifier per sport on the full historical feature matrix.
# Falls back to weighted-average predict_game if XGBoost unavailable.
# ============================================================

def load_xgb_models() -> None:
    """
    Loads persisted XGBoost models from models.pkl into the global XGB_MODELS dict.
    Called once at startup from main().
    """
    global XGB_MODELS, XGB_FEATURE_NAMES
    if not XGBOOST_AVAILABLE:
        return

    if not os.path.exists(XGB_MODEL_FILE):
        return

    try:
        with open(XGB_MODEL_FILE, 'rb') as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict):
            XGB_MODELS      = bundle.get('models', {})
            XGB_FEATURE_NAMES = bundle.get('feature_names', XGB_FEATURE_NAMES_DEFAULT)
            log.info(f'XGBoost models loaded: {list(XGB_MODELS.keys())}')
    except Exception as exc:
        log.warning(f'Could not load XGBoost models: {exc}')


def save_xgb_models(models: dict, feature_names: List[str]) -> None:
    """Saves XGBoost models and feature names to models.pkl."""
    try:
        bundle = {'models': models, 'feature_names': feature_names}
        with open(XGB_MODEL_FILE, 'wb') as f:
            pickle.dump(bundle, f, protocol=4)
        log.info(f'XGBoost models saved: {list(models.keys())}')
    except Exception as exc:
        log.error(f'Failed to save XGBoost models: {exc}')


def build_feature_vector(
    home: str, away: str, game_date: str, venue: str, sport: str,
    patterns: dict, team_history: dict, elo_ratings: dict,
    pythagorean: dict, standings: dict, team_stats: dict,
    espn_stats: dict, barttorvik: dict, sos: dict,
    injuries_home: Optional[List[dict]] = None,
    injuries_away: Optional[List[dict]] = None,
    weather: Optional[dict] = None,
    odds_parsed: Optional[dict] = None,
    game_time: str = '',
    elo_snapshot: Optional[dict] = None,
    pythag_snapshot: Optional[dict] = None,
) -> np.ndarray:
    """
    Converts a game into a fixed-length feature vector for XGBoost.
    Uses np.nan for signals not applicable to a sport — XGBoost handles
    missing values natively.
    Returns 1D numpy array matching XGB_FEATURE_NAMES_DEFAULT order.
    """
    nan = np.nan

    # ---------- hfa ----------
    hfa = HFA_DEFAULT.get(sport, 0.575)
    if venue and venue in VENUE_HFA_OVERRIDE:
        hfa = VENUE_HFA_OVERRIDE[venue]

    # ---------- h2h_prob ----------
    h2h_prob = nan
    h2h_key  = '|||'.join(sorted([home, away]))
    h2h_entry = (patterns.get('h2h') or {}).get(h2h_key)
    if isinstance(h2h_entry, list) and len(h2h_entry) >= 2:
        total = h2h_entry[0] + h2h_entry[1]
        if total >= MIN_PATTERN_GAMES:
            home_is_first = sorted([home, away])[0] == home
            hw = h2h_entry[0] if home_is_first else h2h_entry[1]
            h2h_prob = hw / total

    # ---------- venue_prob ----------
    venue_prob = nan
    if venue and len(venue) > 2:
        ve = (patterns.get('venue') or {}).get(f'{home}|{venue[:30]}')
        if isinstance(ve, list) and len(ve) >= 2 and ve[0] + ve[1] >= MIN_PATTERN_GAMES:
            venue_prob = ve[0] / (ve[0] + ve[1])

    # ---------- dow_prob ----------
    dow_prob = nan
    if game_date and len(game_date) >= 10:
        try:
            dow = datetime.datetime.strptime(game_date[:10], '%Y-%m-%d').weekday()
            de  = (patterns.get('dow') or {}).get(f'{home}|{dow}')
            if isinstance(de, list) and len(de) >= 2 and de[0] + de[1] >= 4:
                dow_prob = de[0] / (de[0] + de[1])
        except (ValueError, TypeError):
            pass

    # ---------- form_diff ----------
    form_diff = nan
    hf = get_recent_form(home, sport, game_date, team_history)
    af = get_recent_form(away, sport, game_date, team_history)
    if hf is not None and af is not None:
        form_diff = hf['score'] - af['score']

    # ---------- pythag_diff ----------
    pythag_diff = nan
    # Use per-game snapshot during training to prevent leakage; fall back to current values
    _pythag = pythag_snapshot if pythag_snapshot is not None else pythagorean
    sp = (_pythag.get(sport) or {})
    hp = safe_float(sp.get(home))
    ap = safe_float(sp.get(away))
    if hp is not None and ap is not None:
        pythag_diff = hp - ap

    # ---------- elo_diff ----------
    elo_diff = nan
    # Use per-game snapshot during training to prevent leakage; fall back to current values
    _elo = elo_snapshot if elo_snapshot is not None else elo_ratings
    se = (_elo.get(sport) or {})
    he = safe_float(se.get(home))
    ae = safe_float(se.get(away))
    if he is not None and ae is not None:
        elo_diff = he - ae

    # ---------- standings_diff ----------
    standings_diff = nan
    hs = (standings.get(home) or {}) if isinstance(standings, dict) else {}
    as_ = (standings.get(away) or {}) if isinstance(standings, dict) else {}
    hop = safe_float(hs.get('away_win_pct'))
    awp = safe_float(as_.get('away_win_pct'))
    if hop is not None and awp is not None:
        standings_diff = hop - awp

    # ---------- ff_diff (basketball only) ----------
    ff_diff = nan
    if sport in ('nba', 'ncaabm', 'ncaabw') and isinstance(team_stats, dict):
        ss = team_stats.get(sport) or {}
        hff = four_factors_score(ss.get(home))
        aff = four_factors_score(ss.get(away))
        if hff is not None and aff is not None:
            ff_diff = hff - aff

    # ---------- efficiency_diff ----------
    efficiency_diff = nan
    if isinstance(team_stats, dict):
        ss  = team_stats.get(sport) or {}
        hst = ss.get(home) if isinstance(ss, dict) else None
        ast = ss.get(away) if isinstance(ss, dict) else None
        if isinstance(hst, dict) and isinstance(ast, dict):
            hp2 = safe_float(hst.get('pts'))
            ap2 = safe_float(ast.get('pts'))
            if hp2 is not None and ap2 is not None:
                efficiency_diff = (hp2 - ap2) / 20.0

    # ---------- sp_diff (NBA/NHL only) ----------
    sp_diff = nan
    if sport in ('nba', 'nhl') and isinstance(team_stats, dict):
        ss  = team_stats.get(sport) or {}
        hst = ss.get(home) if isinstance(ss, dict) else None
        ast = ss.get(away) if isinstance(ss, dict) else None
        if isinstance(hst, dict) and isinstance(ast, dict):
            hp3 = safe_float(hst.get('pts'))
            ap3 = safe_float(ast.get('pts'))
            if hp3 is not None and ap3 is not None:
                pace_range = 15.0 if sport == 'nba' else 1.0
                sp_diff    = (hp3 - ap3) / pace_range

    # ---------- turnover_diff ----------
    turnover_diff = nan
    if sport in ('nfl', 'nba', 'ncaabm', 'ncaabw') and isinstance(team_stats, dict):
        ss  = team_stats.get(sport) or {}
        hst = ss.get(home) if isinstance(ss, dict) else None
        ast = ss.get(away) if isinstance(ss, dict) else None
        if isinstance(hst, dict) and isinstance(ast, dict):
            ht = safe_float(hst.get('turnover'))
            at = safe_float(ast.get('turnover'))
            if ht is not None and at is not None:
                turnover_diff = at - ht  # positive = home turns it over less

    # ---------- rest_adj ----------
    rh  = get_rest_days(home, sport, game_date, team_history)
    ra  = get_rest_days(away, sport, game_date, team_history)
    rest_adj = rest_signal(rh, ra, sport)

    # ---------- bb_diff ----------
    bb_diff = bounce_back_signal(home, sport, game_date, team_history) - \
              bounce_back_signal(away, sport, game_date, team_history)

    # ---------- travel_miles ----------
    travel_miles = calculate_travel_miles(away, venue) if venue else 0.0

    # ---------- tz_diff ----------
    tz_diff = calculate_timezone_diff(away, venue) if venue else 0.0

    # ---------- altitude_flag ----------
    altitude_flag = 1.0 if venue and any(v.lower() in venue.lower() for v in HIGH_ALTITUDE_VENUES) else 0.0

    # ---------- weather_drag ----------
    weather_drag = 0.0
    if sport in OUTDOOR_SPORTS and isinstance(weather, dict):
        temp   = safe_float(weather.get('temp')) or 65.0
        wind   = safe_float(weather.get('wind')) or 5.0
        precip = safe_float(weather.get('precip')) or 0.0
        snow   = safe_float(weather.get('snow')) or 0.0
        if temp < 32: weather_drag += 0.03
        if temp < 20: weather_drag += 0.03
        if wind > 15: weather_drag += 0.03
        if wind > 25: weather_drag += 0.05
        if precip > 50: weather_drag += 0.02
        if snow > 0: weather_drag += 0.04

    # ---------- injury_impact ----------
    injury_impact = 0.0
    for inj_list, sign in [(injuries_home, -1.0), (injuries_away, 1.0)]:
        if inj_list and isinstance(inj_list, list):
            impact = sum(
                i.get('impact', 0.0)
                for i in inj_list
                if isinstance(i, dict) and i.get('status') == 'Out'
            )
            injury_impact += sign * impact

    # ---------- ml_prob / spread_prob ----------
    ml_prob    = nan
    spread_prob = nan
    if isinstance(odds_parsed, dict):
        ml_parsed = safe_float(odds_parsed.get('ml_prob'))
        sp_parsed = safe_float(odds_parsed.get('avg_spread'))
        if ml_parsed is not None:
            ml_prob     = ml_parsed
        if sp_parsed is not None:
            spread_prob = spread_to_prob(sp_parsed, sport)

    # ---------- qb_rating_diff (NFL only) ----------
    qb_rating_diff = nan
    if sport == 'nfl':
        hqb = _get_espn_stat(espn_stats, sport, home, 'passerRating') or \
              _get_espn_stat(espn_stats, sport, home, 'QBRating')
        aqb = _get_espn_stat(espn_stats, sport, away, 'passerRating') or \
              _get_espn_stat(espn_stats, sport, away, 'QBRating')
        if hqb is not None and aqb is not None:
            qb_rating_diff = (hqb - aqb) / 50.0  # Normalize: ~50 passer rating range

    # ---------- net_rating_diff (NBA only) ----------
    net_rating_diff = nan
    if sport == 'nba':
        hn = _get_espn_stat(espn_stats, sport, home, 'netRating') or \
             _get_espn_stat(espn_stats, sport, home, 'netPoints')
        an = _get_espn_stat(espn_stats, sport, away, 'netRating') or \
             _get_espn_stat(espn_stats, sport, away, 'netPoints')
        if hn is not None and an is not None:
            net_rating_diff = (hn - an) / 10.0  # Normalize: ~10 point range

    # ---------- back_to_back ----------
    btb_home = 1.0 if sport in ('nba', 'nhl') and is_back_to_back(home, sport, game_date, team_history) else 0.0
    btb_away = 1.0 if sport in ('nba', 'nhl') and is_back_to_back(away, sport, game_date, team_history) else 0.0

    # ---------- efficiency_margin_diff (NCAAB only) ----------
    em_diff = nan
    if sport in ('ncaabm', 'ncaabw') and isinstance(barttorvik, dict):
        hem = safe_float((barttorvik.get(home) or {}).get('adj_em'))
        aem = safe_float((barttorvik.get(away) or {}).get('adj_em'))
        if hem is not None and aem is not None:
            em_diff = (hem - aem) / 20.0  # Normalize: ~20 pt adj EM range

    # ---------- goalie_sv_pct_diff (NHL only) ----------
    goalie_diff = nan
    if sport == 'nhl':
        hsv = _get_espn_stat(espn_stats, sport, home, 'savePct') or \
              _get_espn_stat(espn_stats, sport, home, 'savePercentage')
        asv = _get_espn_stat(espn_stats, sport, away, 'savePct') or \
              _get_espn_stat(espn_stats, sport, away, 'savePercentage')
        if hsv is not None and asv is not None:
            goalie_diff = (hsv - asv) / 0.05  # Normalize: ~.05 SV% range

    # ---------- power_play_diff (NHL only) ----------
    pp_diff = nan
    if sport == 'nhl':
        hpp = _get_espn_stat(espn_stats, sport, home, 'powerPlayPct') or \
              _get_espn_stat(espn_stats, sport, home, 'powerPlayPercentage')
        app = _get_espn_stat(espn_stats, sport, away, 'powerPlayPct') or \
              _get_espn_stat(espn_stats, sport, away, 'powerPlayPercentage')
        if hpp is not None and app is not None:
            pp_diff = (hpp - app) / 10.0  # Normalize: ~10% PP range

    # ---------- park_factor (MLB only) ----------
    park = nan
    if sport == 'mlb':
        park = get_park_factor(venue)

    # ---------- conference_strength_diff (NCAAF only) ----------
    conf_diff = nan
    # Conference strength is computed from historical cross-conference results
    # stored in sos dict — used as proxy here

    # ---------- sos_diff ----------
    sos_diff = nan
    if isinstance(sos, dict):
        sport_sos = sos.get(sport) or {}
        hs_val = safe_float(sport_sos.get(home))
        as_val = safe_float(sport_sos.get(away))
        if hs_val is not None and as_val is not None:
            sos_diff = hs_val - as_val

    # ---------- div_game ----------
    # 1.0 if both teams are in the same NFL division, 0.0 otherwise.
    # Division games have different dynamics: road team performs better than typical HFA suggests.
    div_game = 0.0
    if sport == 'nfl':
        hd = NFL_DIVISIONS.get(home)
        ad = NFL_DIVISIONS.get(away)
        if hd and ad and hd == ad:
            div_game = 1.0

    vec = np.array([
        hfa, h2h_prob, venue_prob, dow_prob, form_diff,
        pythag_diff, elo_diff, standings_diff, ff_diff,
        efficiency_diff, sp_diff, turnover_diff, rest_adj,
        bb_diff, travel_miles, tz_diff, altitude_flag, weather_drag,
        injury_impact, ml_prob, spread_prob,
        qb_rating_diff, net_rating_diff,
        btb_home, btb_away,
        em_diff, goalie_diff,
        pp_diff, park, conf_diff, sos_diff,
        div_game,
    ], dtype=np.float32)

    return vec


def train_xgboost_models(all_games_by_sport: Dict[str, List[dict]], data: dict) -> Dict[str, Any]:
    """
    Trains one XGBClassifier per sport on the full historical feature matrix.
    Returns dict of sport -> fitted model.

    all_games_by_sport: {sport: [game_dict, ...]} — each game must have
        home_won field and all signal-computable fields.
    data: full data dict (patterns, team_history, elo, etc.)
    """
    if not XGBOOST_AVAILABLE:
        log.warning('XGBoost not available — skipping model training')
        return {}

    trained: Dict[str, Any] = {}

    for sport, games in all_games_by_sport.items():
        if not games:
            continue

        log.info(f'Training XGBoost for {sport.upper()} ({len(games)} games)...')

        X_rows = []
        y_rows = []

        for game in games:
            if not isinstance(game, dict):
                continue
            home_won = game.get('home_won')
            if home_won is None:
                continue

            fv = build_feature_vector(
                home=game.get('home', ''),
                away=game.get('away', ''),
                game_date=game.get('date', ''),
                venue=game.get('venue', ''),
                sport=sport,
                patterns=data.get('patterns', {}),
                team_history=data.get('team_history', {}),
                elo_ratings=data.get('elo_ratings', {}),
                pythagorean=data.get('pythagorean', {}),
                standings=data.get('standings', {}).get(sport, {}),
                team_stats=data.get('team_stats', {}),
                espn_stats=data.get('espn_stats', {}),
                barttorvik=data.get('barttorvik', {}),
                sos=data.get('sos', {}),
                game_time=game.get('game_time', ''),
                elo_snapshot=game.get('elo_snapshot'),
                pythag_snapshot=game.get('pythag_snapshot'),
            )
            X_rows.append(fv)
            y_rows.append(1 if home_won else 0)

        if len(X_rows) < 100:
            log.warning(f'XGBoost {sport}: too few samples ({len(X_rows)}) — skipping')
            continue

        X = np.array(X_rows)
        y = np.array(y_rows)

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X, y, verbose=False)

        # Evaluate on training data (rough check)
        preds    = model.predict(X)
        acc      = np.mean(preds == y)
        log.info(f'XGBoost {sport}: {len(X_rows)} games, training accuracy {acc:.1%}')

        trained[sport] = model

    return trained


def retrain_xgboost_incremental(data: dict) -> dict:
    """
    Incremental retraining is DISABLED.

    The correct growth path for the XGBoost models is --mode train, which trains
    chronologically across all seasons with no hindsight leakage. The incremental
    approach replaced the full model trained on 6,991+ games with a model trained on
    only the ~50 buffer games, actively degrading accuracy over the season.

    Resolved games are still buffered in data['training_buffer'] and are included
    in the next full --mode train run, so the model grows correctly at season end.
    """
    return data

# SECTION 11: WEATHER FETCHERS
# Open Meteo first (free, no key). OpenWeather as fallback.
# ============================================================

def fetch_open_meteo(lat: float, lon: float, game_date: str) -> Optional[dict]:
    """
    Fetches weather forecast from Open Meteo (no key required).
    Returns normalized weather dict or None on any failure.
    """
    if lat is None or lon is None:
        return None
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None
    if math.isnan(lat) or math.isnan(lon):
        return None

    url = 'https://api.open-meteo.com/v1/forecast'
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m,windspeed_10m,precipitation_probability,snowfall',
        'temperature_unit': 'fahrenheit',
        'windspeed_unit': 'mph',
        'forecast_days': 7
    }

    data = fetch_json(url, params=params, max_retries=2, timeout=20)
    if not data:
        return None

    hourly = data.get('hourly') or {}
    if not isinstance(hourly, dict):
        return None

    times = hourly.get('time') or []
    if not isinstance(times, list) or not times:
        return None

    # Find the hourly slot closest to the game time
    try:
        target_ts = datetime.datetime.fromisoformat(game_date[:10]).timestamp()
    except (ValueError, TypeError):
        return None

    best_idx  = 0
    best_diff = float('inf')

    for i, t in enumerate(times):
        try:
            ts   = datetime.datetime.fromisoformat(str(t)).timestamp()
            diff = abs(ts - target_ts)
            if diff < best_diff:
                best_diff = diff
                best_idx  = i
        except (ValueError, TypeError):
            continue

    temps  = hourly.get('temperature_2m') or []
    winds  = hourly.get('windspeed_10m') or []
    precips = hourly.get('precipitation_probability') or []
    snows  = hourly.get('snowfall') or []

    return {
        'temp':   safe_float(temps[best_idx]) if best_idx < len(temps) else 65.0,
        'wind':   safe_float(winds[best_idx]) if best_idx < len(winds) else 5.0,
        'precip': safe_float(precips[best_idx]) if best_idx < len(precips) else 0.0,
        'snow':   safe_float(snows[best_idx]) if best_idx < len(snows) else 0.0,
        'source': 'OpenMeteo'
    }


def fetch_openweather(venue: str, game_date: str) -> Optional[dict]:
    """
    Fetches weather from OpenWeather API (key required).
    Used as fallback when Open Meteo fails or no coordinates available.
    """
    if not OPENWEATHER_KEY or not venue:
        return None

    # Strip venue type words to get city name
    city = venue
    for word in ['stadium', 'arena', 'field', 'park', 'center', 'centre',
                 'dome', 'coliseum', 'complex', 'garden']:
        city = city.lower().replace(word, '')

    # Take first part before comma
    city = city.split(',')[0].strip()
    city = ' '.join(city.split())

    if not city or len(city) < 3:
        return None

    data = fetch_json(
        'https://api.openweathermap.org/data/2.5/forecast',
        params={'q': city, 'appid': OPENWEATHER_KEY, 'units': 'imperial'},
        max_retries=2,
        timeout=15
    )

    if not data:
        return None

    forecast_list = data.get('list') or []
    if not isinstance(forecast_list, list) or not forecast_list:
        return None

    try:
        target_ts = datetime.datetime.fromisoformat(game_date[:10]).timestamp()
    except (ValueError, TypeError):
        return None

    best_item = None
    best_diff = float('inf')

    for item in forecast_list:
        if not isinstance(item, dict):
            continue
        dt = safe_int(item.get('dt'))
        if dt is None:
            continue
        diff = abs(dt - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_item = item

    if not best_item:
        return None

    main = best_item.get('main') or {}
    wind = best_item.get('wind') or {}
    snow = best_item.get('snow') or {}

    return {
        'temp':   safe_float(main.get('temp') if isinstance(main, dict) else None) or 65.0,
        'wind':   safe_float(wind.get('speed') if isinstance(wind, dict) else None) or 5.0,
        'precip': (safe_float(best_item.get('pop')) or 0.0) * 100,
        'snow':   safe_float(snow.get('3h') if isinstance(snow, dict) else None) or 0.0,
        'source': 'OpenWeather'
    }


def get_weather(venue_lat: Optional[float], venue_lon: Optional[float],
                venue: str, game_date: str) -> Optional[dict]:
    """
    Main weather entry point. Tries Open Meteo first, then OpenWeather.
    Returns weather dict or None if both fail.
    """
    weather = fetch_open_meteo(venue_lat, venue_lon, game_date)
    if weather:
        return weather
    return fetch_openweather(venue, game_date)

# ============================================================
# SECTION 12: ODDS API FETCHER
# Betting lines provide market consensus signal.
# ============================================================

def fetch_betting_odds(sport: str) -> Optional[List[dict]]:
    """
    Fetches moneyline and spread odds from The Odds API.
    Returns raw list of games with bookmaker data, or None.
    """
    if not ODDS_API_KEY:
        return None

    sport_key = ODDS_SPORT_KEYS.get(sport)
    if not sport_key:
        return None

    data = fetch_json(
        f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds',
        params={
            'apiKey':  ODDS_API_KEY,
            'regions': 'us,uk,eu,au',
            'markets': 'h2h,spreads'
        },
        max_retries=2,
        timeout=15
    )

    if not isinstance(data, list):
        return None

    return data


def parse_odds_for_game(
    odds_data: List[dict],
    home: str,
    away: str
) -> Optional[dict]:
    """
    Finds a game in odds data and extracts average market probabilities.
    Removes bookmaker vig before averaging to get true market consensus.
    Returns dict with ml_prob and avg_spread, or None if game not found.
    """
    if not odds_data or not isinstance(odds_data, list):
        return None

    if not home or not away:
        return None

    # Find matching game - handle possible name mismatches
    game = next(
        (g for g in odds_data
         if isinstance(g, dict)
         and ((g.get('home_team') == home and g.get('away_team') == away)
              or (g.get('home_team') == away and g.get('away_team') == home))),
        None
    )

    if not game:
        return None

    is_reversed = game.get('home_team') == away

    ml_sum   = 0.0
    ml_count = 0
    spread_sum   = 0.0
    spread_count = 0

    for book in (game.get('bookmakers') or []):
        if not isinstance(book, dict):
            continue

        for market in (book.get('markets') or []):
            if not isinstance(market, dict):
                continue

            outcomes = market.get('outcomes') or []
            if not isinstance(outcomes, list):
                continue

            if market.get('key') == 'h2h':
                home_out = next((o for o in outcomes
                                 if isinstance(o, dict) and o.get('name') == (away if is_reversed else home)), None)
                away_out = next((o for o in outcomes
                                 if isinstance(o, dict) and o.get('name') == (home if is_reversed else away)), None)

                if home_out and away_out:
                    home_raw = _american_to_raw_prob(safe_float(home_out.get('price')))
                    away_raw = _american_to_raw_prob(safe_float(away_out.get('price')))

                    if home_raw is not None and away_raw is not None:
                        # Remove vig by dividing by total implied probability
                        total = home_raw + away_raw
                        if total > 0:
                            ml_sum   += home_raw / total
                            ml_count += 1

            elif market.get('key') == 'spreads':
                home_out = next((o for o in outcomes
                                 if isinstance(o, dict)
                                 and o.get('name') == (away if is_reversed else home)), None)
                if home_out:
                    point = safe_float(home_out.get('point'))
                    if point is not None:
                        spread_sum   += point
                        spread_count += 1

    ml_prob    = (ml_sum / ml_count) if ml_count > 0 else None
    avg_spread = (spread_sum / spread_count) if spread_count > 0 else None

    return {'ml_prob': ml_prob, 'avg_spread': avg_spread}


def _american_to_raw_prob(american: Optional[float]) -> Optional[float]:
    """
    Converts American odds to raw (vig-included) implied probability.
    +150 means bet 100 to win 150.
    -150 means bet 150 to win 100.
    """
    if american is None:
        return None
    if american > 0:
        return 100.0 / (american + 100.0)
    if american < 0:
        return (-american) / (-american + 100.0)
    return None


def spread_to_prob(spread: float, sport: str) -> float:
    """
    Converts a point spread to win probability using sport-specific standard deviations.
    spread is from the home team's perspective: negative = home favored.
    Uses normal distribution approximation.
    """
    # Standard deviations of scoring margin per sport
    STD_DEVS: Dict[str, float] = {
        'nfl': 13.45, 'nba': 11.0, 'mlb': 1.5, 'nhl': 1.0,
        'ncaaf': 16.0, 'ncaabm': 9.0, 'ncaabw': 9.0
    }

    sd = STD_DEVS.get(sport, 10.0)
    if sd <= 0:
        return 0.5

    # z-score: negative spread (home favored) -> home more likely to win
    z = -spread / sd

    # Approximation of the normal CDF: 0.5 * (1 + erf(z / sqrt(2)))
    # Using math.erf which is accurate to machine precision
    prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    return clamp_prob(prob)

# ============================================================
# SECTION 13: MATH UTILITIES
# All statistical calculations used in prediction.
# ============================================================

def pythagorean_ratio(pts_for: float, pts_against: float, sport: str) -> float:
    """
    Calculates Pythagorean expectation ratio for a team.
    Formula: PF^exp / (PF^exp + PA^exp)
    Better predictor of future wins than actual win percentage.
    Returns 0.5 if inputs are invalid.
    """
    exp = PYTHAG_EXP.get(sport, 2.37)

    # Guard: both must be positive numbers
    pf = max(1.0, float(pts_for)) if isinstance(pts_for, (int, float)) and not math.isnan(pts_for) else 1.0
    pa = max(1.0, float(pts_against)) if isinstance(pts_against, (int, float)) and not math.isnan(pts_against) else 1.0

    try:
        pf_exp = math.pow(pf, exp)
        pa_exp = math.pow(pa, exp)
        total  = pf_exp + pa_exp

        if total <= 0:
            return 0.5

        return pf_exp / total
    except (ValueError, OverflowError):
        return 0.5


def pythagorean_prob(home_ratio: Optional[float], away_ratio: Optional[float]) -> Optional[float]:
    """
    Converts two Pythagorean ratios to a win probability.
    Returns None if either ratio is missing.
    """
    if home_ratio is None or away_ratio is None:
        return None

    diff = home_ratio - away_ratio

    # Scale difference to probability: 0.8 factor from calibration research
    prob = 0.5 + diff * 0.8

    return clamp_prob(prob)


def elo_prob(home_elo: float, away_elo: float) -> float:
    """
    Standard Elo win probability formula.
    Returns probability that home team wins.
    """
    if not isinstance(home_elo, (int, float)) or not isinstance(away_elo, (int, float)):
        return 0.5

    return clamp_prob(1.0 / (1.0 + math.pow(10.0, (away_elo - home_elo) / 400.0)))


def wilson_lower_bound(wins: float, total: float) -> float:
    """
    Wilson score lower bound. Shrinks confidence when sample size is small.
    Prevents small samples from appearing more reliable than they are.
    Returns adjusted win probability (conservative estimate).
    """
    if total <= 0:
        return 0.5

    z = 1.96  # 95% confidence interval
    p = wins / total
    n = total

    denominator = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denominator
    margin = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denominator

    return max(0.01, min(0.99, center - margin))


def normalize_margin(margin: float, sport: str) -> float:
    """
    Normalizes a game margin to [0, 1] range using sport-specific scale.
    Used so form scores are comparable across sports.
    """
    norm = MARGIN_NORM.get(sport, 10)
    if norm <= 0:
        return 0.5

    # Shift margin into positive range then divide by max expected
    return max(0.0, min(1.0, (margin + norm) / (norm * 2.0)))


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates great-circle distance in miles between two coordinates.
    Used for travel distance signal.
    All inputs must be valid floats.
    """
    for val in [lat1, lon1, lat2, lon2]:
        if not isinstance(val, (int, float)) or math.isnan(val):
            return 0.0

    R = 3959.0  # Earth radius in miles

    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat   = math.radians(lat2 - lat1)
    dlon   = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2.0) ** 2
         + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2.0) ** 2)

    # Clamp to [0, 1] to prevent domain errors in asin from floating point imprecision
    a = max(0.0, min(1.0, a))

    c = 2.0 * math.asin(math.sqrt(a))

    return R * c


def get_city_from_team_name(team_name: str) -> Optional[str]:
    """
    Extracts city name from team name for coordinate lookup.
    Returns lowercase city string or None if not identifiable.
    """
    if not team_name:
        return None

    # Remove common words that are not cities
    clean = team_name.lower()
    for word in ['fc', 'sc', 'united', 'city', 'the']:
        clean = clean.replace(f' {word}', '').replace(f'{word} ', '')

    # Try to match against known cities
    for city in CITY_COORDS:
        if city in clean:
            return city

    # Split and try first word (most teams: "City Nickname")
    parts = team_name.strip().split()
    if len(parts) >= 2:
        first = parts[0].lower()
        if first in CITY_COORDS:
            return first

        # Handle two-word cities: "Los Angeles", "New York", etc.
        two_word = f'{parts[0]} {parts[1]}'.lower()
        if two_word in CITY_COORDS:
            return two_word

    return None


def calculate_travel_miles(away_team: str, venue_city: str) -> float:
    """
    Calculates travel distance for the away team to reach the venue.
    Returns miles traveled, or 0 if cities cannot be determined.
    """
    away_city = get_city_from_team_name(away_team)
    venue_city_lower = venue_city.lower() if venue_city else ''

    # Find venue city in coords
    home_city = None
    for city in CITY_COORDS:
        if city in venue_city_lower:
            home_city = city
            break

    if not away_city or not home_city:
        return 0.0

    away_coords = CITY_COORDS.get(away_city)
    home_coords = CITY_COORDS.get(home_city)

    if not away_coords or not home_coords:
        return 0.0

    return haversine_miles(away_coords[0], away_coords[1],
                           home_coords[0], home_coords[1])


def calculate_timezone_diff(away_team: str, venue_city: str) -> int:
    """
    Calculates number of timezone hours the away team crosses traveling to the venue.
    Positive means away team travels east (body clock disadvantage for east coast games).
    Negative means away team travels west.
    Returns 0 if cities cannot be determined.
    """
    away_city = get_city_from_team_name(away_team)
    venue_city_lower = venue_city.lower() if venue_city else ''

    home_city = None
    for city in CITY_TIMEZONES:
        if city in venue_city_lower:
            home_city = city
            break

    if not away_city or not home_city:
        return 0

    away_tz = CITY_TIMEZONES.get(away_city)
    home_tz = CITY_TIMEZONES.get(home_city)

    if away_tz is None or home_tz is None:
        return 0

    # Positive = away team traveled east (body clock ahead = disadvantage)
    # LA away (-8), NY home (-5): -5 - (-8) = +3 -> home boosted
    # NY away (-5), LA home (-8): -8 - (-5) = -3 -> minimal effect (west travel easier)
    return home_tz - away_tz

# ============================================================
# SECTION 14: TEAM HISTORY MANAGEMENT
# Stores recent game results per team for form and rest calculation.
# ============================================================

def update_team_history(games: List[dict], sport: str, team_history: dict) -> None:
    """
    Updates team_history with results from a list of games.
    Games must have: home, away, date, home_score, away_score, home_won.
    Updates in place - team_history[sport][team_name] = list of recent games.
    """
    if not isinstance(games, list) or not games:
        return

    if sport not in team_history:
        team_history[sport] = {}

    # Sort games chronologically before adding
    sorted_games = sorted(
        [g for g in games if isinstance(g, dict) and g.get('home') and g.get('away')],
        key=lambda g: g.get('date', '')
    )

    for game in sorted_games:
        home = game.get('home', '')
        away = game.get('away', '')
        date = game.get('date', '')

        if not home or not away or not date:
            continue

        home_score = safe_int(game.get('home_score'))
        away_score = safe_int(game.get('away_score'))

        if home_score is None or away_score is None:
            continue

        home_won = bool(game.get('home_won', home_score > away_score))
        margin   = home_score - away_score

        home_entry = {
            'date':    date,
            'win':     home_won,
            'margin':  margin,
            'is_home': True,
            'score_for':     home_score,
            'score_against': away_score
        }

        away_entry = {
            'date':    date,
            'win':     not home_won,
            'margin':  -margin,
            'is_home': False,
            'score_for':     away_score,
            'score_against': home_score
        }

        # Home team
        if home not in team_history[sport]:
            team_history[sport][home] = []
        team_history[sport][home].append(home_entry)
        if len(team_history[sport][home]) > MAX_TEAM_HISTORY:
            team_history[sport][home].pop(0)  # Remove oldest

        # Away team
        if away not in team_history[sport]:
            team_history[sport][away] = []
        team_history[sport][away].append(away_entry)
        if len(team_history[sport][away]) > MAX_TEAM_HISTORY:
            team_history[sport][away].pop(0)


def get_recent_form(
    team: str,
    sport: str,
    before_date: str,
    team_history: dict
) -> Optional[dict]:
    """
    Calculates a team's recent form score before a given date.
    Uses last N games (sport-specific window) weighted by margin of victory.
    Returns form dict or None if insufficient data.
    """
    if not team or not sport or not before_date:
        return None

    sport_history = team_history.get(sport) if isinstance(team_history, dict) else None
    if not sport_history:
        return None

    team_games = sport_history.get(team) if isinstance(sport_history, dict) else None
    if not team_games or not isinstance(team_games, list):
        return None

    # Filter to games before the target date
    before_games = [
        g for g in team_games
        if isinstance(g, dict) and g.get('date', '') < before_date
    ]

    window = FORM_WINDOW.get(sport, 5)

    # Take the most recent N games
    recent = before_games[-window:] if len(before_games) > window else before_games

    if len(recent) < 3:
        return None

    wins       = sum(1 for g in recent if g.get('win', False))
    win_rate   = wins / len(recent)
    avg_margin = sum(g.get('margin', 0) for g in recent) / len(recent)

    norm_margin = normalize_margin(avg_margin, sport)

    # Form score: 70% win rate component, 30% margin component
    form_score = win_rate * 0.7 + norm_margin * 0.3

    return {
        'wins':       wins,
        'losses':     len(recent) - wins,
        'win_rate':   win_rate,
        'avg_margin': avg_margin,
        'score':      form_score,
        'games':      len(recent)
    }


def get_rest_days(team: str, sport: str, game_date: str, team_history: dict) -> int:
    """
    Calculates days since a team's last game before game_date.
    Returns 7 as a neutral default if history is unavailable.
    """
    if not team or not sport or not game_date:
        return 7

    sport_history = team_history.get(sport) if isinstance(team_history, dict) else None
    if not sport_history:
        return 7

    team_games = sport_history.get(team) if isinstance(sport_history, dict) else None
    if not team_games or not isinstance(team_games, list):
        return 7

    past_games = [
        g for g in team_games
        if isinstance(g, dict) and g.get('date', '') < game_date
    ]

    if not past_games:
        return 7

    last_game = max(past_games, key=lambda g: g.get('date', ''))
    last_date = last_game.get('date', '')

    if not last_date:
        return 7

    try:
        last_dt = datetime.datetime.strptime(last_date, '%Y-%m-%d')
        game_dt = datetime.datetime.strptime(game_date[:10], '%Y-%m-%d')
        days    = (game_dt - last_dt).days
        return max(0, days)
    except (ValueError, TypeError):
        return 7


def rest_signal(rest_home: int, rest_away: int, sport: str) -> float:
    """
    Calculates rest advantage signal.
    Returns positive value when home team is better rested,
    negative when away team is better rested.
    """
    if not isinstance(rest_home, int) or not isinstance(rest_away, int):
        return 0.0

    # NBA and NHL: back-to-backs matter most - documented in multiple studies
    if sport in ('nba', 'nhl'):
        if rest_home == 0 and rest_away > 0:
            return -0.07  # Home on back-to-back - significant disadvantage
        if rest_away == 0 and rest_home > 0:
            return +0.07  # Away on back-to-back - significant disadvantage
        if rest_home == 0 and rest_away == 0:
            return 0.0    # Both tired - neutral
        return max(-0.03, min(0.03, (rest_home - rest_away) * 0.01))

    # NFL: short week and bye week effects
    if sport == 'nfl':
        if rest_home < 6 and rest_away >= 7:
            return -0.04
        if rest_away < 6 and rest_home >= 7:
            return +0.04
        if rest_home >= 12 and rest_away < 10:
            return +0.03  # Home had bye
        if rest_away >= 12 and rest_home < 10:
            return -0.03  # Away had bye
        return 0.0

    # General effect for other sports
    return max(-0.02, min(0.02, (rest_home - rest_away) * 0.008))


def bounce_back_signal(team: str, sport: str, game_date: str, team_history: dict) -> float:
    """
    Detects bounce-back/letdown situation based on last game margin.
    Teams that were blown out tend to respond with better effort next game.
    Teams that won big tend to have slight letdown risk.
    Returns signed float adjustment.
    """
    if not team or not sport or not game_date:
        return 0.0

    sport_history = team_history.get(sport) if isinstance(team_history, dict) else None
    if not sport_history:
        return 0.0

    team_games = (sport_history.get(team) or []) if isinstance(sport_history, dict) else []
    if not team_games:
        return 0.0

    past_games = [
        g for g in team_games
        if isinstance(g, dict) and g.get('date', '') < game_date
    ]

    if not past_games:
        return 0.0

    last = max(past_games, key=lambda g: g.get('date', ''))
    margin = last.get('margin', 0)

    blowout = BLOWOUT_THRESH.get(sport, 20)

    if margin < -blowout:
        return +0.025   # Blowout loss -> bounce-back
    if margin > blowout:
        return -0.015   # Blowout win -> potential letdown
    return 0.0


def four_factors_score(team_stats: Optional[dict]) -> Optional[float]:
    """
    Calculates Dean Oliver Four Factors composite score for basketball.
    Weights: eFG% 45%, Turnover% 25%, OReb% 15%, FT Rate 15%.
    Higher is better. Returns None if stats are unavailable.
    """
    if not team_stats or not isinstance(team_stats, dict):
        return None

    fg_pct = safe_float(team_stats.get('fg_pct'))
    fg3    = safe_float(team_stats.get('fg3_pct'))
    to     = safe_float(team_stats.get('turnover'))
    oreb   = safe_float(team_stats.get('oreb'))
    ft     = safe_float(team_stats.get('ft_pct'))
    pts    = safe_float(team_stats.get('pts'))

    if fg_pct is None:
        return None

    # Effective FG% (3-pointers worth 1.5x per field goal attempt)
    efg = fg_pct
    if fg3 is not None:
        efg = fg_pct + 0.5 * fg3  # Approximation without FGA breakdown
        efg = min(1.0, efg)

    # Turnover rate proxy: turnovers per points (lower is better for this team)
    to_rate = 0.15  # league average default
    if to is not None and pts is not None and pts > 0:
        to_rate = to / pts
        to_rate = max(0.05, min(0.30, to_rate))

    # Offensive rebound rate proxy
    oreb_rate = 0.25  # league average default
    if oreb is not None:
        oreb_rate = min(0.5, oreb / 10.0)  # Normalized

    # FT rate
    ft_rate = ft if ft is not None else 0.75

    # Four factors composite (higher = better offense)
    score = (efg * 0.45) - (to_rate * 0.25) + (oreb_rate * 0.15) + (ft_rate * 0.15)

    return score

# ============================================================
# SECTION 15: PATTERN MANAGEMENT
# H2H, venue, and day-of-week patterns stored as [wins, losses].
# ============================================================

def add_to_pattern(patterns: dict, pattern_type: str, key: str,
                   home_won: bool, weight: float = 1.0) -> None:
    """
    Adds a game result to a pattern bucket.
    patterns[pattern_type][key] = [home_wins, home_losses]
    weight allows blowout games to count more than close games.
    """
    if pattern_type not in patterns:
        patterns[pattern_type] = {}

    if key not in patterns[pattern_type]:
        patterns[pattern_type][key] = [0.0, 0.0]

    entry = patterns[pattern_type][key]
    if not isinstance(entry, list) or len(entry) < 2:
        patterns[pattern_type][key] = [0.0, 0.0]
        entry = patterns[pattern_type][key]

    if home_won:
        entry[0] += weight
    else:
        entry[1] += weight


def build_patterns_from_games(games: List[dict], patterns: dict) -> None:
    """
    Builds H2H, venue, and day-of-week patterns from a list of games.
    Adds to existing patterns (does not rebuild from scratch).
    Uses MOV weighting: blowouts count more than close games.
    """
    if not isinstance(games, list) or not games:
        return

    for game in games:
        if not isinstance(game, dict):
            continue

        home = game.get('home', '')
        away = game.get('away', '')
        date = game.get('date', '')

        if not home or not away:
            continue

        home_won = game.get('home_won')
        if home_won is None:
            continue

        home_won = bool(home_won)

        # MOV weight: blowouts are more signal, very close games are more noise
        margin = abs(safe_int(game.get('margin')) or 0)
        if margin >= 20:
            mov_weight = 2.0
        elif margin >= 10:
            mov_weight = 1.3
        elif margin <= 3:
            mov_weight = 0.85
        else:
            mov_weight = 1.0

        # H2H: key is sorted so direction doesn't matter, index [0] = first alphabetically
        h2h_key = '|||'.join(sorted([home, away]))
        home_is_first = sorted([home, away])[0] == home
        # Store win for whichever team won (mapped to [0] = first-alpha team)
        first_won = (home_is_first and home_won) or (not home_is_first and not home_won)
        add_to_pattern(patterns, 'h2h', h2h_key, first_won, mov_weight)

        # Venue: home team's record at this specific venue
        venue = str(game.get('venue', '') or '').strip()
        if venue and len(venue) > 2:
            venue_key = f'{home}|{venue[:30]}'
            add_to_pattern(patterns, 'venue', venue_key, home_won, mov_weight)

        # Day of week: home team's record on this day
        if date and len(date) >= 10:
            try:
                dow = datetime.datetime.strptime(date[:10], '%Y-%m-%d').weekday()
                dow_key = f'{home}|{dow}'
                add_to_pattern(patterns, 'dow', dow_key, home_won)
            except (ValueError, TypeError):
                pass


def prune_patterns(patterns: dict) -> None:
    """
    Removes pattern entries with too few games to be reliable.
    Called after each training year to keep memory usage controlled.
    """
    minimums = {'h2h': MIN_PATTERN_GAMES, 'venue': MIN_PATTERN_GAMES, 'dow': 4}

    for pattern_type, min_games in minimums.items():
        if pattern_type not in patterns:
            continue

        keys_to_delete = []

        for key, val in patterns[pattern_type].items():
            if not isinstance(val, list) or len(val) < 2:
                keys_to_delete.append(key)
                continue

            total = val[0] + val[1]
            if total < min_games:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del patterns[pattern_type][key]


def compress_patterns(patterns: dict) -> None:
    """
    If patterns exceed MAX_PATTERNS_PER_TYPE, keeps only the most
    predictive ones (those with the most extreme win rates).
    More extreme = more signal. Least extreme = closest to 50/50 = noise.
    """
    for pattern_type in ['h2h', 'venue', 'dow']:
        if pattern_type not in patterns:
            continue

        entries = list(patterns[pattern_type].items())

        if len(entries) <= MAX_PATTERNS_PER_TYPE:
            continue

        def extremity_key(item: tuple) -> float:
            val = item[1]
            if not isinstance(val, list) or len(val) < 2:
                return 0.0
            total = val[0] + val[1]
            if total <= 0:
                return 0.0
            win_rate = val[0] / total
            return abs(win_rate - 0.5)

        entries.sort(key=extremity_key, reverse=True)
        patterns[pattern_type] = dict(entries[:MAX_PATTERNS_PER_TYPE])

# ============================================================
# SECTION 16: WEIGHT MANAGEMENT
# ============================================================

def normalize_weights(weights: dict) -> dict:
    """
    Normalizes weights so their sum equals the default weights sum.
    Prevents gradual drift while preserving relative differences.
    Applies clamp_weight to every weight after normalization.
    """
    if not isinstance(weights, dict) or not weights:
        return copy.deepcopy(DEFAULT_WEIGHTS)

    default_sum = sum(DEFAULT_WEIGHTS.values())
    current_sum = sum(v for v in weights.values() if isinstance(v, (int, float)))

    if current_sum <= 0:
        return copy.deepcopy(DEFAULT_WEIGHTS)

    scale = default_sum / current_sum

    return {
        key: clamp_weight(val * scale)
        for key, val in weights.items()
        if isinstance(val, (int, float))
    }


def get_week_number(date_str: str) -> int:
    """
    Returns the ISO week number for a date string.
    Returns 1 on any parsing failure.
    """
    if not date_str:
        return 1

    try:
        dt = datetime.datetime.strptime(date_str[:10], '%Y-%m-%d')
        return dt.isocalendar()[1]
    except (ValueError, TypeError):
        return 1

# ============================================================
# SECTION 17: CORE PREDICTION ENGINE
# predict_game() is the main function that combines all signals.
# Takes all data as parameters - does not read from any global state.
# ============================================================

def predict_game(
    home: str,
    away: str,
    game_date: str,
    venue: str,
    sport: str,
    weights: dict,
    patterns: dict,
    team_history: dict,
    elo_ratings: dict,
    pythagorean: dict,
    standings: dict,
    team_stats: dict,
    injuries_home: Optional[List[dict]] = None,
    injuries_away: Optional[List[dict]] = None,
    weather: Optional[dict] = None,
    odds_parsed: Optional[dict] = None,
    espn_stats: Optional[dict] = None,
    barttorvik: Optional[dict] = None,
    sos: Optional[dict] = None,
    game_time: str = '',
) -> dict:
    """
    Core prediction function. Combines all available signals into a single
    probability estimate. Takes all data as parameters - no global reads.

    Returns dict with:
      home_prob, away_prob, pick, pick_prob, confidence, signals_fired, is_guess
    """

    # Validate required inputs
    if not home or not away or not sport:
        return _default_prediction(home or 'Unknown', away or 'Unknown')

    if not isinstance(weights, dict):
        weights = copy.deepcopy(DEFAULT_WEIGHTS)
    if not isinstance(patterns, dict):
        patterns = {'h2h': {}, 'venue': {}, 'dow': {}}
    if not isinstance(team_history, dict):
        team_history = {}
    if not isinstance(elo_ratings, dict):
        elo_ratings = {}
    if not isinstance(pythagorean, dict):
        pythagorean = {}
    if not isinstance(standings, dict):
        standings = {}
    if not isinstance(team_stats, dict):
        team_stats = {}
    if espn_stats is None:
        espn_stats = {}
    if barttorvik is None:
        barttorvik = {}
    if sos is None:
        sos = {}

    # ----------------------------------------------------------
    # XGBoost fast path — if a trained model is available for this
    # sport, use it. Falls back to weighted-average path below.
    # ----------------------------------------------------------
    if sport in XGB_MODELS and XGB_MODELS[sport] is not None:
        try:
            fv = build_feature_vector(
                home=home, away=away, game_date=game_date,
                venue=venue, sport=sport,
                patterns=patterns, team_history=team_history,
                elo_ratings=elo_ratings, pythagorean=pythagorean,
                standings=standings, team_stats=team_stats,
                espn_stats=espn_stats, barttorvik=barttorvik, sos=sos,
                injuries_home=injuries_home, injuries_away=injuries_away,
                weather=weather, odds_parsed=odds_parsed, game_time=game_time,
            )
            proba       = XGB_MODELS[sport].predict_proba(fv.reshape(1, -1))[0]
            home_prob   = float(clamp_prob(proba[1]))  # proba[1] = P(home_won=1)
            pick_is_home = home_prob >= 0.5
            pick_prob    = max(home_prob, 1.0 - home_prob)
            if pick_prob >= 0.75:
                confidence = 'lock'
            elif pick_prob >= 0.65:
                confidence = 'strong'
            elif pick_prob >= 0.55:
                confidence = 'medium'
            elif pick_prob > 0.50:
                confidence = 'tossup'
            else:
                confidence = 'coinflip'
            return {
                'home_prob':     home_prob,
                'away_prob':     1.0 - home_prob,
                'pick':          home if pick_is_home else away,
                'pick_prob':     pick_prob,
                'confidence':    confidence,
                'signals_fired': ['xgboost'],
                'is_guess':      False
            }
        except Exception as xgb_exc:
            log.warning(f'XGBoost prediction failed for {home} vs {away}: {xgb_exc}')
            # Fall through to weighted-average path

    prob_sum    = 0.0
    weight_sum  = 0.0
    raw_adjustments = 0.0  # Additive signals (not averaged)
    signals_fired = []

    # ----------------------------------------------------------
    # Signal 1: Home Field Advantage
    # Priority: per-venue override > standings-based > sport default
    # ----------------------------------------------------------
    hfa = HFA_DEFAULT.get(sport, 0.575)

    if venue and venue in VENUE_HFA_OVERRIDE:
        hfa = VENUE_HFA_OVERRIDE[venue]
    else:
        home_standing = standings.get(home) if isinstance(standings, dict) else None
        if isinstance(home_standing, dict):
            home_pct = safe_float(home_standing.get('home_win_pct'))
            if home_pct is not None:
                # Dampen toward default to avoid small sample extremes
                hfa = 0.40 + home_pct * 0.38

    hfa_w = weights.get('hfa', 1.5)
    prob_sum   += hfa * hfa_w
    weight_sum += hfa_w
    signals_fired.append('hfa')

    # ----------------------------------------------------------
    # Signal 2: Head-to-Head Historical Record
    # ----------------------------------------------------------
    h2h_key     = '|||'.join(sorted([home, away]))
    h2h_patterns = patterns.get('h2h') if isinstance(patterns, dict) else {}
    h2h_entry   = h2h_patterns.get(h2h_key) if isinstance(h2h_patterns, dict) else None

    if isinstance(h2h_entry, list) and len(h2h_entry) >= 2:
        total = h2h_entry[0] + h2h_entry[1]
        if total >= MIN_PATTERN_GAMES:
            home_is_first = sorted([home, away])[0] == home
            first_wins = h2h_entry[0]
            home_wins  = first_wins if home_is_first else h2h_entry[1]

            # Wilson lower bound for confidence when sample is small
            raw_prob   = home_wins / total
            confidence = wilson_lower_bound(home_wins, total)
            eff_prob   = raw_prob * 0.3 + confidence * 0.7  # Blend raw and conservative

            h2h_w = weights.get('h2h', 1.5)
            prob_sum   += eff_prob * h2h_w
            weight_sum += h2h_w
            signals_fired.append('h2h')

    # ----------------------------------------------------------
    # Signal 3: Venue Record
    # ----------------------------------------------------------
    venue_patterns = patterns.get('venue') if isinstance(patterns, dict) else {}
    if venue and len(venue) > 2 and isinstance(venue_patterns, dict):
        venue_key   = f'{home}|{venue[:30]}'
        venue_entry = venue_patterns.get(venue_key)

        if isinstance(venue_entry, list) and len(venue_entry) >= 2:
            total = venue_entry[0] + venue_entry[1]
            if total >= MIN_PATTERN_GAMES:
                raw_prob   = venue_entry[0] / total
                confidence = wilson_lower_bound(venue_entry[0], total)
                eff_prob   = raw_prob * 0.3 + confidence * 0.7

                venue_w = weights.get('venue', 1.2)
                prob_sum   += eff_prob * venue_w
                weight_sum += venue_w
                signals_fired.append('venue')

    # ----------------------------------------------------------
    # Signal 4: Day of Week
    # ----------------------------------------------------------
    dow_patterns = patterns.get('dow') if isinstance(patterns, dict) else {}
    if game_date and len(game_date) >= 10 and isinstance(dow_patterns, dict):
        try:
            dow     = datetime.datetime.strptime(game_date[:10], '%Y-%m-%d').weekday()
            dow_key = f'{home}|{dow}'
            dow_entry = dow_patterns.get(dow_key)

            if isinstance(dow_entry, list) and len(dow_entry) >= 2:
                total = dow_entry[0] + dow_entry[1]
                if total >= 4:
                    raw_prob   = dow_entry[0] / total
                    confidence = wilson_lower_bound(dow_entry[0], total)
                    eff_prob   = raw_prob * confidence  # Conservative for DOW

                    dow_w = weights.get('dow', 0.8)
                    prob_sum   += eff_prob * dow_w
                    weight_sum += dow_w
                    signals_fired.append('dow')

        except (ValueError, TypeError):
            pass

    # ----------------------------------------------------------
    # Signal 5: Recent Form
    # ----------------------------------------------------------
    home_form = get_recent_form(home, sport, game_date, team_history)
    away_form = get_recent_form(away, sport, game_date, team_history)

    if home_form is not None and away_form is not None:
        form_diff = home_form['score'] - away_form['score']  # Range [-1, 1]
        form_prob = clamp_prob(0.5 + form_diff * 0.5)

        form_w = weights.get('form', 1.8)
        prob_sum   += form_prob * form_w
        weight_sum += form_w
        signals_fired.append('form')

    # ----------------------------------------------------------
    # Signal 6: Pythagorean Expectation
    # Better predictor of future performance than win/loss record.
    # ----------------------------------------------------------
    sport_pythag  = pythagorean.get(sport) if isinstance(pythagorean, dict) else {}
    home_pythag   = sport_pythag.get(home) if isinstance(sport_pythag, dict) else None
    away_pythag   = sport_pythag.get(away) if isinstance(sport_pythag, dict) else None
    pythag_prob   = pythagorean_prob(
        safe_float(home_pythag),
        safe_float(away_pythag)
    )

    if pythag_prob is not None:
        pythag_w = weights.get('pythag', 1.3)
        prob_sum   += pythag_prob * pythag_w
        weight_sum += pythag_w
        signals_fired.append('pythag')

    # ----------------------------------------------------------
    # Signal 7: Elo Rating (all sports — built from historical games)
    # ----------------------------------------------------------
    if isinstance(elo_ratings, dict):
        sport_elos = elo_ratings.get(sport) or {}
        home_elo   = safe_float(sport_elos.get(home)) if isinstance(sport_elos, dict) else None
        away_elo   = safe_float(sport_elos.get(away)) if isinstance(sport_elos, dict) else None

        if home_elo is not None and away_elo is not None:
            ep    = elo_prob(home_elo, away_elo)
            elo_w = weights.get('elo', 1.3)
            prob_sum   += ep * elo_w
            weight_sum += elo_w
            signals_fired.append('elo')

    # ----------------------------------------------------------
    # Signal 8: Current Season Standings (away team road record)
    # ----------------------------------------------------------
    away_standing = standings.get(away) if isinstance(standings, dict) else None
    if isinstance(away_standing, dict):
        away_road_pct = safe_float(away_standing.get('away_win_pct'))
        if away_road_pct is not None:
            # Away team being a strong road team reduces home advantage
            # 0.5 = league average road team; 0.6 = strong road team
            away_road_strength = 0.5 + (away_road_pct - 0.5) * 0.5
            home_advantage_from_standing = 1.0 - away_road_strength

            stand_w = weights.get('standings', 1.1)
            prob_sum   += clamp_prob(home_advantage_from_standing) * stand_w
            weight_sum += stand_w
            signals_fired.append('standings')

    # ----------------------------------------------------------
    # Signal 9: Four Factors (Basketball only)
    # ----------------------------------------------------------
    if sport in ('nba', 'ncaabm', 'ncaabw') and isinstance(team_stats, dict):
        sport_stats  = team_stats.get(sport) if isinstance(team_stats, dict) else {}
        home_s       = sport_stats.get(home) if isinstance(sport_stats, dict) else None
        away_s       = sport_stats.get(away) if isinstance(sport_stats, dict) else None

        home_ff = four_factors_score(home_s)
        away_ff = four_factors_score(away_s)

        if home_ff is not None and away_ff is not None:
            ff_diff = home_ff - away_ff  # Typically in range [-0.3, 0.3]
            ff_prob = clamp_prob(0.5 + ff_diff * 1.5)

            ff_w = weights.get('four_factors', 1.3)
            prob_sum   += ff_prob * ff_w
            weight_sum += ff_w
            signals_fired.append('four_factors')

    # ----------------------------------------------------------
    # Signal 10: Efficiency Differential (general scoring efficiency)
    # ----------------------------------------------------------
    if isinstance(team_stats, dict):
        sport_stats = team_stats.get(sport) or {}
        home_st     = sport_stats.get(home) if isinstance(sport_stats, dict) else None
        away_st     = sport_stats.get(away) if isinstance(sport_stats, dict) else None

        if isinstance(home_st, dict) and isinstance(away_st, dict):
            home_pts = safe_float(home_st.get('pts'))
            away_pts = safe_float(away_st.get('pts'))

            if home_pts is not None and away_pts is not None:
                pts_range = 20.0  # Typical range between best/worst teams
                diff = (home_pts - away_pts) / pts_range
                eff_prob = clamp_prob(0.5 + diff * 0.4)

                eff_w = weights.get('efficiency', 1.4)
                prob_sum   += eff_prob * eff_w
                weight_sum += eff_w
                signals_fired.append('efficiency')

    # ----------------------------------------------------------
    # Signal 11: Scoring Pace / Tempo (sp)
    # Points-per-game differential for NBA; goals-per-game for NHL.
    # Fast-paced teams are favored when their pace advantage is consistent.
    # ----------------------------------------------------------
    if sport in ('nba', 'nhl') and isinstance(team_stats, dict):
        sport_stats = team_stats.get(sport) or {}
        home_st     = sport_stats.get(home) if isinstance(sport_stats, dict) else None
        away_st     = sport_stats.get(away) if isinstance(sport_stats, dict) else None

        if isinstance(home_st, dict) and isinstance(away_st, dict):
            home_pts = safe_float(home_st.get('pts'))
            away_pts = safe_float(away_st.get('pts'))

            if home_pts is not None and away_pts is not None and home_pts > 0 and away_pts > 0:
                # Normalize: NBA typically 105-120 ppg, NHL 2.5-3.5 gpg
                pace_range = 15.0 if sport == 'nba' else 1.0
                diff = (home_pts - away_pts) / pace_range
                sp_prob = clamp_prob(0.5 + diff * 0.35)

                sp_w = weights.get('sp', 1.4)
                prob_sum   += sp_prob * sp_w
                weight_sum += sp_w
                signals_fired.append('sp')

    # ----------------------------------------------------------
    # Signal 12: Turnover Differential
    # NFL: negative plays kill drives; NBA/NCAAB: live-ball turnovers create runs.
    # BallDontLie provides per-game turnover averages for NFL/NBA.
    # ----------------------------------------------------------
    if sport in ('nfl', 'nba', 'ncaabm', 'ncaabw') and isinstance(team_stats, dict):
        sport_stats = team_stats.get(sport) or {}
        home_st     = sport_stats.get(home) if isinstance(sport_stats, dict) else None
        away_st     = sport_stats.get(away) if isinstance(sport_stats, dict) else None

        if isinstance(home_st, dict) and isinstance(away_st, dict):
            home_to = safe_float(home_st.get('turnover'))
            away_to = safe_float(away_st.get('turnover'))

            if home_to is not None and away_to is not None:
                # Fewer turnovers is better — subtract home from away so
                # positive diff = home team turns it over less = advantage
                to_diff = away_to - home_to
                # Typical per-game range: NFL 1-3, NBA 12-18
                to_range = 2.0 if sport == 'nfl' else 5.0
                to_prob = clamp_prob(0.5 + (to_diff / to_range) * 0.30)

                to_w = weights.get('turnover', 1.2)
                prob_sum   += to_prob * to_w
                weight_sum += to_w
                signals_fired.append('turnover')

    # ----------------------------------------------------------
    # Signal 13: QB Passer Rating Differential (NFL only)
    # #1 NFL predictor by a wide margin.
    # ----------------------------------------------------------
    if sport == 'nfl' and isinstance(espn_stats, dict):
        hqb = _get_espn_stat(espn_stats, sport, home, 'passerRating') or \
              _get_espn_stat(espn_stats, sport, home, 'QBRating')
        aqb = _get_espn_stat(espn_stats, sport, away, 'passerRating') or \
              _get_espn_stat(espn_stats, sport, away, 'QBRating')
        if hqb is not None and aqb is not None:
            diff     = (hqb - aqb) / 50.0  # Normalize ~50 point range
            qb_prob  = clamp_prob(0.5 + diff * 0.4)
            qb_w     = weights.get('qb_rating', 2.0)
            prob_sum   += qb_prob * qb_w
            weight_sum += qb_w
            signals_fired.append('qb_rating')

    # ----------------------------------------------------------
    # Signal 14: NBA Net Rating Differential
    # Offensive rating - defensive rating = best single NBA quality metric.
    # ----------------------------------------------------------
    if sport == 'nba' and isinstance(espn_stats, dict):
        hn = _get_espn_stat(espn_stats, sport, home, 'netRating') or \
             _get_espn_stat(espn_stats, sport, home, 'netPoints')
        an = _get_espn_stat(espn_stats, sport, away, 'netRating') or \
             _get_espn_stat(espn_stats, sport, away, 'netPoints')
        if hn is not None and an is not None:
            diff    = (hn - an) / 10.0
            nr_prob = clamp_prob(0.5 + diff * 0.4)
            nr_w    = weights.get('net_rating', 1.8)
            prob_sum   += nr_prob * nr_w
            weight_sum += nr_w
            signals_fired.append('net_rating')

    # ----------------------------------------------------------
    # Signal 15: Back-to-Back (NBA and NHL)
    # Teams playing second game in two nights lose ~5% more often.
    # ----------------------------------------------------------
    if sport in ('nba', 'nhl'):
        btb_home_flag = is_back_to_back(home, sport, game_date, team_history)
        btb_away_flag = is_back_to_back(away, sport, game_date, team_history)
        if btb_home_flag or btb_away_flag:
            btb_adj = 0.0
            if btb_home_flag:
                btb_adj -= 0.045
            if btb_away_flag:
                btb_adj += 0.045
            btb_w = weights.get('back_to_back', 1.1)
            raw_adjustments += btb_adj * btb_w
            signals_fired.append('back_to_back')

    # ----------------------------------------------------------
    # Signal 16: NCAAB Adjusted Efficiency Margin (KenPom equivalent)
    # Best single predictor for college basketball.
    # ----------------------------------------------------------
    if sport in ('ncaabm', 'ncaabw') and isinstance(barttorvik, dict):
        hem = safe_float((barttorvik.get(home) or {}).get('adj_em'))
        aem = safe_float((barttorvik.get(away) or {}).get('adj_em'))
        if hem is not None and aem is not None:
            diff    = (hem - aem) / 20.0
            em_prob = clamp_prob(0.5 + diff * 0.45)
            em_w    = weights.get('efficiency_margin', 1.8)
            prob_sum   += em_prob * em_w
            weight_sum += em_w
            signals_fired.append('efficiency_margin')

        # NCAAB Tempo differential
        ht = safe_float((barttorvik.get(home) or {}).get('tempo'))
        at = safe_float((barttorvik.get(away) or {}).get('tempo'))
        if ht is not None and at is not None:
            tempo_diff = (ht - at) / 5.0  # ~5 possession range
            tempo_prob = clamp_prob(0.5 + tempo_diff * 0.2)
            tm_w       = weights.get('tempo', 0.9)
            prob_sum   += tempo_prob * tm_w
            weight_sum += tm_w
            signals_fired.append('tempo')

        # NCAAB Strength of Schedule from barttorvik
        hs = safe_float((barttorvik.get(home) or {}).get('sos'))
        as_ = safe_float((barttorvik.get(away) or {}).get('sos'))
        if hs is not None and as_ is not None:
            sos_diff = (hs - as_) / 5.0
            sos_prob = clamp_prob(0.5 + sos_diff * 0.25)
            sos_w    = weights.get('sos', 1.2)
            prob_sum   += sos_prob * sos_w
            weight_sum += sos_w
            signals_fired.append('sos')

    # ----------------------------------------------------------
    # Signal 17: NHL Goalie Save % Differential
    # To hockey what QB rating is to football.
    # ----------------------------------------------------------
    if sport == 'nhl' and isinstance(espn_stats, dict):
        hsv = _get_espn_stat(espn_stats, sport, home, 'savePct') or \
              _get_espn_stat(espn_stats, sport, home, 'savePercentage')
        asv = _get_espn_stat(espn_stats, sport, away, 'savePct') or \
              _get_espn_stat(espn_stats, sport, away, 'savePercentage')
        if hsv is not None and asv is not None:
            diff     = (hsv - asv) / 0.05
            sv_prob  = clamp_prob(0.5 + diff * 0.4)
            sv_w     = weights.get('goalie_sv_pct', 1.9)
            prob_sum   += sv_prob * sv_w
            weight_sum += sv_w
            signals_fired.append('goalie_sv_pct')

        # NHL Power Play % differential
        hpp = _get_espn_stat(espn_stats, sport, home, 'powerPlayPct') or \
              _get_espn_stat(espn_stats, sport, home, 'powerPlayPercentage')
        app = _get_espn_stat(espn_stats, sport, away, 'powerPlayPct') or \
              _get_espn_stat(espn_stats, sport, away, 'powerPlayPercentage')
        if hpp is not None and app is not None:
            diff    = (hpp - app) / 10.0
            pp_prob = clamp_prob(0.5 + diff * 0.3)
            pp_w    = weights.get('power_play', 1.3)
            prob_sum   += pp_prob * pp_w
            weight_sum += pp_w
            signals_fired.append('power_play')

    # ----------------------------------------------------------
    # Signal 18: MLB Park Factor
    # Adjusts for run-inflating/deflating ballpark effects.
    # ----------------------------------------------------------
    if sport == 'mlb' and venue:
        pf = get_park_factor(venue)
        if pf != 1.0:
            # >1.0 = hitter's park = slight home advantage boost
            pf_adj = (pf - 1.0) * 0.10
            pf_w   = weights.get('park_factor', 0.8)
            raw_adjustments += pf_adj * pf_w
            signals_fired.append('park_factor')

    # ----------------------------------------------------------
    # Signal 19: NFL Division Game
    # Division rivals are more competitive; upsets more common.
    # ----------------------------------------------------------
    if sport == 'nfl' and is_division_game(home, away):
        div_w = weights.get('division_game', 0.9)
        # Division games reduce home favorite's advantage ~5%
        raw_adjustments -= 0.03 * div_w
        signals_fired.append('division_game')

    # ----------------------------------------------------------
    # Signal 20: NCAAF Rivalry Game
    # Heavy favorites lose more in rivalry games.
    # ----------------------------------------------------------
    if sport == 'ncaaf' and is_rivalry_game(home, away):
        riv_w = weights.get('rivalry_game', 0.8)
        raw_adjustments -= 0.04 * riv_w
        signals_fired.append('rivalry_game')

    # ----------------------------------------------------------
    # Signal 21: Strength of Schedule (computed internally)
    # ----------------------------------------------------------
    if isinstance(sos, dict):
        sport_sos = sos.get(sport) or {}
        hs_val = safe_float(sport_sos.get(home))
        as_val = safe_float(sport_sos.get(away))
        if hs_val is not None and as_val is not None and 'sos' not in signals_fired:
            diff    = (hs_val - as_val) / 0.1
            sos_prob = clamp_prob(0.5 + diff * 0.2)
            sos_w    = weights.get('sos', 1.2)
            prob_sum   += sos_prob * sos_w
            weight_sum += sos_w
            signals_fired.append('sos')

    # ----------------------------------------------------------
    # ADDITIVE ADJUSTMENTS
    # These modify the final probability directly (not averaged in).
    # ----------------------------------------------------------
    rest_home = get_rest_days(home, sport, game_date, team_history)
    rest_away = get_rest_days(away, sport, game_date, team_history)
    rest_adj  = rest_signal(rest_home, rest_away, sport)

    if rest_adj != 0.0:
        rest_w = weights.get('rest', 1.0)
        raw_adjustments += rest_adj * rest_w * 0.5
        signals_fired.append('rest')

    # Bounce-back/letdown
    home_bb = bounce_back_signal(home, sport, game_date, team_history)
    away_bb = bounce_back_signal(away, sport, game_date, team_history)
    bb_diff = home_bb - away_bb
    if abs(bb_diff) > 0.005:
        bb_w = weights.get('bounce_back', 0.6)
        raw_adjustments += bb_diff * bb_w
        signals_fired.append('bounce_back')

    # Travel distance
    if venue:
        travel_miles = calculate_travel_miles(away, venue)
        if travel_miles > 500:
            # Scale: 500 miles = small effect, 3000 miles = full effect
            travel_factor = min(1.0, (travel_miles - 500) / 2500.0)
            travel_adj    = travel_factor * 0.035  # max ~0.035 boost to home
            travel_w      = weights.get('travel', 0.9)
            raw_adjustments += travel_adj * travel_w
            signals_fired.append('travel')

        # Timezone disadvantage
        tz_diff = calculate_timezone_diff(away, venue)
        if abs(tz_diff) >= 2:
            # 2-hour crossing = small, 3+ = significant
            tz_adj    = min(0.025, abs(tz_diff) * 0.008)
            # If away team crosses east (+), they play earlier in body time -> disadvantage
            tz_sign   = 1.0 if tz_diff > 0 else -1.0
            tz_w      = weights.get('timezone', 0.8)
            raw_adjustments += tz_adj * tz_sign * tz_w
            signals_fired.append('timezone')

    # Altitude penalty for visiting teams
    if venue and any(v.lower() in venue.lower() for v in HIGH_ALTITUDE_VENUES):
        alt_w = weights.get('altitude', 0.7)
        raw_adjustments += 0.025 * alt_w
        signals_fired.append('altitude')

    # Cold weather penalty for warm-climate teams
    if sport in ('nfl', 'ncaaf') and weather and isinstance(weather, dict):
        temp = safe_float(weather.get('temp'))
        if temp is not None and temp < 35 and away in WARM_CLIMATE_NFL:
            weather_w = weights.get('weather', 1.0)
            raw_adjustments += 0.04 * weather_w
            signals_fired.append('weather_cold')

    # Weather general impact (outdoor sports)
    if sport in OUTDOOR_SPORTS and weather and isinstance(weather, dict):
        temp   = safe_float(weather.get('temp')) or 65.0
        wind   = safe_float(weather.get('wind')) or 5.0
        precip = safe_float(weather.get('precip')) or 0.0
        snow   = safe_float(weather.get('snow')) or 0.0

        drag = 0.0
        if temp < 32:
            drag += 0.03
        if temp < 20:
            drag += 0.03
        if wind > 15:
            drag += 0.03
        if wind > 25:
            drag += 0.05
        if precip > 50:
            drag += 0.02
        if snow > 0:
            drag += 0.04

        if drag > 0.01:
            weather_w = weights.get('weather', 1.0)
            raw_adjustments += drag * 0.3 * weather_w  # Small directional signal
            if 'weather_cold' not in signals_fired:
                signals_fired.append('weather')

    # Injury impact
    if injuries_home and isinstance(injuries_home, list):
        home_impact = sum(
            inj.get('impact', 0.0)
            for inj in injuries_home
            if isinstance(inj, dict) and inj.get('status') == 'Out'
        )
        if home_impact > 0.01:
            inj_w = weights.get('injury', 1.5)
            raw_adjustments -= home_impact * inj_w * 0.3
            signals_fired.append('injury_home')

    if injuries_away and isinstance(injuries_away, list):
        away_impact = sum(
            inj.get('impact', 0.0)
            for inj in injuries_away
            if isinstance(inj, dict) and inj.get('status') == 'Out'
        )
        if away_impact > 0.01:
            inj_w = weights.get('injury', 1.5)
            raw_adjustments += away_impact * inj_w * 0.3
            signals_fired.append('injury_away')

    # Betting market consensus
    if odds_parsed and isinstance(odds_parsed, dict):
        ml_prob = safe_float(odds_parsed.get('ml_prob'))
        if ml_prob is not None:
            crowd_w = weights.get('crowd', 1.0)
            prob_sum   += ml_prob * crowd_w
            weight_sum += crowd_w
            signals_fired.append('market')

        avg_spread = safe_float(odds_parsed.get('avg_spread'))
        if avg_spread is not None:
            spread_p = spread_to_prob(avg_spread, sport)
            spread_w = weights.get('spread', 1.2)
            prob_sum   += spread_p * spread_w
            weight_sum += spread_w
            signals_fired.append('spread')

    # ----------------------------------------------------------
    # Final probability calculation
    # ----------------------------------------------------------
    if weight_sum > 0:
        base_prob = prob_sum / weight_sum
    else:
        base_prob = HFA_DEFAULT.get(sport, 0.575)

    home_prob = clamp_prob(base_prob + raw_adjustments)

    pick_is_home = home_prob >= 0.5
    pick_prob    = max(home_prob, 1.0 - home_prob)

    # Confidence label
    if pick_prob >= 0.75:
        confidence = 'lock'
    elif pick_prob >= 0.65:
        confidence = 'strong'
    elif pick_prob >= 0.55:
        confidence = 'medium'
    elif pick_prob > 0.50:
        confidence = 'tossup'
    else:
        confidence = 'coinflip'

    # is_guess: only HFA fired, no historical data exists for this matchup
    is_guess = len(signals_fired) <= 1

    return {
        'home_prob':     home_prob,
        'away_prob':     1.0 - home_prob,
        'pick':          home if pick_is_home else away,
        'pick_prob':     pick_prob,
        'confidence':    confidence,
        'signals_fired': signals_fired,
        'is_guess':      is_guess
    }


def _default_prediction(home: str, away: str) -> dict:
    """Returns a neutral prediction when inputs are invalid."""
    return {
        'home_prob':     0.55,
        'away_prob':     0.45,
        'pick':          home,
        'pick_prob':     0.55,
        'confidence':    'coinflip',
        'signals_fired': [],
        'is_guess':      True
    }

# ============================================================
# SECTION 18: WEIGHT OPTIMIZATION
# Walk-forward: tests on unseen data each iteration.
# No hardcoded accuracy ceilings. Stops when converged.
# ============================================================

def optimize_weights_for_year(
    games: List[dict],
    sport: str,
    start_weights: dict,
    patterns: dict,
    team_history: dict,
    elo_ratings: dict,
    pythagorean: dict,
    standings: dict,
    team_stats: dict,
    espn_stats: Optional[dict] = None,
    barttorvik: Optional[dict] = None,
    sos: Optional[dict] = None,
) -> Tuple[dict, float]:
    """
    Optimizes weights on one year of games.
    Returns (optimized_weights, final_accuracy).
    Uses gradient-free weight adjustment: reinforce correct signals, penalize wrong ones.
    No hardcoded accuracy ceiling - stops only when improvement plateaus.
    """
    if not games or len(games) < 10:
        return copy.deepcopy(start_weights), 0.0

    weights  = copy.deepcopy(start_weights)
    prev_acc = 0.0
    plateau  = 0

    # Sort games chronologically
    sorted_games = sorted(
        [g for g in games if isinstance(g, dict)],
        key=lambda g: g.get('date', '')
    )

    accuracy = 0.0  # Track outside loop so return always has final value

    for iteration in range(MAX_TRAIN_ITERATIONS):
        correct = 0

        for game in sorted_games:
            home = game.get('home', '')
            away = game.get('away', '')
            home_won = game.get('home_won')

            if not home or not away or home_won is None:
                continue

            pred = predict_game(
                home=home, away=away,
                game_date=game.get('date', ''),
                venue=game.get('venue', ''),
                sport=sport,
                weights=weights,
                patterns=patterns,
                team_history=team_history,
                elo_ratings=elo_ratings,
                pythagorean=pythagorean,
                standings=standings,
                team_stats=team_stats,
                espn_stats=espn_stats or {},
                barttorvik=barttorvik or {},
                sos=sos or {},
            )

            was_right = (pred['pick'] == home) == bool(home_won)

            if was_right:
                correct += 1

            # Adjust weights for signals that fired this game
            for signal in pred.get('signals_fired', []):
                if signal not in weights:
                    continue
                if was_right:
                    # Reinforce - smaller boost than penalty to prevent overfit
                    weights[signal] = clamp_weight(weights[signal] * (1.0 + LEARNING_RATE * 0.4))
                else:
                    # Penalize - full learning rate
                    weights[signal] = clamp_weight(weights[signal] * (1.0 - LEARNING_RATE))

        # Normalize after each iteration to prevent any single weight dominating
        weights = normalize_weights(weights)

        accuracy = correct / len(sorted_games) if sorted_games else 0.0
        improvement = accuracy - prev_acc

        if iteration > 2:
            if improvement < CONVERGENCE_DELTA:
                plateau += 1
                if plateau >= CONVERGENCE_PLATEAU:
                    log.info(f'  Weights converged at iteration {iteration + 1}, accuracy {accuracy:.3f}')
                    break
            else:
                plateau = 0  # Reset plateau counter on improvement

        prev_acc = accuracy

    return weights, accuracy  # Always return the actual final accuracy, not stale prev_acc


def build_pythagorean_from_games(games: List[dict], sport: str) -> Dict[str, float]:
    """
    Builds Pythagorean expectation ratios from a season of games.
    Returns dict of teamName -> pythagorean ratio [0, 1].
    """
    pts_for:     Dict[str, float] = {}
    pts_against: Dict[str, float] = {}

    for game in games:
        if not isinstance(game, dict):
            continue

        home = game.get('home', '')
        away = game.get('away', '')
        hs   = safe_float(game.get('home_score'))
        as_  = safe_float(game.get('away_score'))

        if not home or not away or hs is None or as_ is None:
            continue

        pts_for[home]     = pts_for.get(home, 0.0) + hs
        pts_against[home] = pts_against.get(home, 0.0) + as_
        pts_for[away]     = pts_for.get(away, 0.0) + as_
        pts_against[away] = pts_against.get(away, 0.0) + hs

    result = {}
    all_teams = set(pts_for.keys()) | set(pts_against.keys())

    for team in all_teams:
        pf = pts_for.get(team, 0.0)
        pa = pts_against.get(team, 0.0)
        result[team] = pythagorean_ratio(pf, pa, sport)

    return result

# ============================================================
# SECTION 19: CALIBRATION TRACKING
# Tracks whether confidence numbers match real outcomes.
# ============================================================

def update_calibration(calibration: dict, sport: str,
                       pick_prob: float, was_correct: bool) -> None:
    """
    Updates calibration data for a sport.
    Tracks wins/losses per confidence bucket (0.5, 0.6, 0.7, 0.8, 0.9).
    """
    if not isinstance(calibration, dict):
        return

    if sport not in calibration:
        calibration[sport] = {}

    # Round to nearest 0.10 to create buckets
    bucket = round(pick_prob * 10) / 10.0
    bucket = max(0.5, min(0.9, bucket))
    key    = f'{bucket:.1f}'

    if key not in calibration[sport]:
        calibration[sport][key] = {'predicted': bucket, 'correct': 0, 'total': 0}

    calibration[sport][key]['total'] += 1
    if was_correct:
        calibration[sport][key]['correct'] += 1


def get_calibration_note(calibration: dict, sport: str, pick_prob: float) -> str:
    """
    Returns a note about historical accuracy at this confidence level.
    Only returns a note if sample size >= 10 and there's a meaningful gap.
    """
    if not isinstance(calibration, dict):
        return ''

    sport_cal = calibration.get(sport)
    if not isinstance(sport_cal, dict):
        return ''

    bucket = round(pick_prob * 10) / 10.0
    key    = f'{bucket:.1f}'
    b      = sport_cal.get(key)

    if not isinstance(b, dict):
        return ''
    if b.get('total', 0) < 10:
        return ''

    actual    = b['correct'] / b['total']
    predicted = b.get('predicted', bucket)
    gap       = actual - predicted

    if abs(gap) < 0.05:
        return ''

    direction = 'historically higher' if gap > 0 else 'historically lower'
    return f' (actual {actual:.0%} - {direction})'

# ============================================================
# SECTION 20: FOUNDATION TRAINING
# Trains all 7 sports year by year from TRAIN_START_YEAR.
# Walk-forward: learns from year N, tests on year N+1.
# No hardcoded ceilings. Tests on unseen data only.
# ============================================================

def run_foundation_training(data: dict) -> dict:
    """
    Main foundation training function.
    Trains weights for all 7 sports using historical data.
    Preserves running record and predictions between training runs.
    Returns updated data dict.
    """
    log.info('=' * 55)
    log.info('EDGE PREDICT - FOUNDATION TRAINING')
    log.info(f'Years: {TRAIN_START_YEAR}-{CURRENT_YEAR}')
    log.info(f'Sports: {", ".join(ALL_SPORTS)}')
    log.info('=' * 55)

    start_time = time.time()

    # Preserve records before resetting training state
    saved_record = copy.deepcopy(data.get('running_record', {}))
    saved_preds  = copy.deepcopy(data.get('predictions', {}))

    # Reset to fresh state
    data = get_empty_data()
    data['running_record'] = saved_record
    data['predictions']    = saved_preds
    data['training_status'] = 'training'

    save_data(data)  # Save "training" status immediately

    # Fetch ESPN stats and Barttorvik once — used by XGBoost feature vectors
    log.info('Fetching ESPN stats for all sports...')
    for sport in ALL_SPORTS:
        es = fetch_espn_stats(sport)
        if es:
            data['espn_stats'][sport] = es
        time.sleep(0.3)

    log.info('Fetching Barttorvik NCAAB efficiency data...')
    btv = fetch_barttorvik()
    if btv:
        data['barttorvik'] = btv

    total_games = 0
    all_games_by_sport: Dict[str, List[dict]] = {s: [] for s in ALL_SPORTS}

    for sport in ALL_SPORTS:
        log.info(f'\n>> TRAINING {sport.upper()} <<')

        current_weights = copy.deepcopy(DEFAULT_WEIGHTS)
        sport_games_total = 0

        # NFL: nflverse has complete data from 1999
        # Other sports: ESPN reliable from 2002 (silently skips empty years)
        # HOLDOUT: exclude the most recent complete season from training.
        # It is tested at the end as a true blind validation.
        # CURRENT_YEAR (2026) is the in-progress season — skip it too.
        holdout_year = CURRENT_YEAR - 1   # 2025 — never trained on
        train_end    = holdout_year - 1   # 2024 — last year included in training
        # NFL: nflverse has complete data from 1999
        # NCAAB: ESPN womens data is sparse before 2008; mens also unreliable pre-2008.
        #        Using 2008 saves ~45 min of empty training time and avoids bad data.
        # All others: ESPN reliable from 2002
        if sport == 'nfl':
            sport_start = 1999
        elif sport in ('ncaabm', 'ncaabw'):
            sport_start = 2008
        else:
            sport_start = 2002
        for year in range(sport_start, train_end + 1):
            log.info(f'  Fetching {sport} {year}...')

            # Primary: best available source (nflverse → CFBD → ESPN → TSDB)
            games = fetch_season(sport, year)

            if not games:
                # Normal for very early years (e.g. ESPN starts ~2002, nflverse 1999)
                # Skip silently for years more than 5 years before ESPN's data starts
                if year < 2003:
                    log.info(f'  {sport} {year}: no data available for this year (expected for pre-2002)')
                else:
                    log.warning(f'  {sport} {year}: no games retrieved — check data source')
                continue

            # ── STEP A: Blind test ────────────────────────────────────────────
            # Test current weights on year N BEFORE incorporating ANY year-N data.
            # Patterns and team history here contain only years before N.
            # This is the genuine anti-hindsight accuracy measurement.
            if sport_games_total >= 50:
                pre_correct = 0
                for game in games:
                    pred = predict_game(
                        home=game.get('home', ''),
                        away=game.get('away', ''),
                        game_date=game.get('date', ''),
                        venue=game.get('venue', ''),
                        sport=sport,
                        weights=current_weights,
                        patterns=data['patterns'],
                        team_history=data['team_history'],
                        elo_ratings=data['elo_ratings'],
                        pythagorean=data['pythagorean'],
                        standings=data['standings'].get(sport, {}),
                        team_stats=data['team_stats'],
                        espn_stats=data.get('espn_stats', {}),
                        barttorvik=data.get('barttorvik', {}),
                        sos=data.get('sos', {}),
                    )
                    home_won = game.get('home_won')
                    if home_won is not None:
                        was_right = (pred['pick'] == game.get('home', '')) == bool(home_won)
                        if was_right:
                            pre_correct += 1

                pre_acc = pre_correct / len(games) if games else 0.0
                log.info(f'  {sport} {year}: {len(games)} games | blind accuracy: {pre_acc:.1%}')
            else:
                log.info(f'  {sport} {year}: {len(games)} games (building baseline)')

            # ── STEP B: Optimize weights ──────────────────────────────────────
            # Optimize weights BEFORE building patterns from this year.
            # Optimizer uses patterns from years 1..N-1 only — no leakage.
            optimized_weights, post_acc = optimize_weights_for_year(
                games=games,
                sport=sport,
                start_weights=current_weights,
                patterns=data['patterns'],
                team_history=data['team_history'],
                elo_ratings=data['elo_ratings'],
                pythagorean=data['pythagorean'],
                standings=data['standings'].get(sport, {}),
                team_stats=data['team_stats'],
                espn_stats=data.get('espn_stats', {}),
                barttorvik=data.get('barttorvik', {}),
                sos=data.get('sos', {}),
            )

            log.info(f'  {sport} {year}: post-train accuracy: {post_acc:.1%}')

            # ── STEP C: Incorporate year N into knowledge base ────────────────
            # NOW add year N patterns, history, and stats so future years can use them.
            # This runs AFTER optimization so it doesn't contaminate this year's training.

            # Build Pythagorean expectation from this year
            pythag_map = build_pythagorean_from_games(games, sport)
            if sport not in data['pythagorean']:
                data['pythagorean'][sport] = {}
            data['pythagorean'][sport].update(pythag_map)

            # Build H2H, venue, day-of-week patterns
            build_patterns_from_games(games, data['patterns'])

            # Update team history (used for form, rest days, bounce-back)
            update_team_history(games, sport, data['team_history'])

            # Update internal Elo ratings from this year's games
            if sport not in data['elo_ratings']:
                data['elo_ratings'][sport] = {}
            data['elo_ratings'][sport] = build_elo_from_games(
                games, sport, data['elo_ratings'][sport]
            )

            # Fetch box score stats if BDN available
            if BALLDONTLIE_KEY and sport in BDN_SPORTS:
                stats = fetch_bdn_team_stats(sport, year)
                if stats:
                    if sport not in data['team_stats']:
                        data['team_stats'][sport] = {}
                    data['team_stats'][sport].update(stats)

            # Collect all games for XGBoost training (Phase 3)
            # Attach ELO and pythagorean snapshots to each game to prevent leakage:
            # a 2005 game should use 2005 ELO/pythagorean values, not 2024 values.
            # deepcopy ensures subsequent years don't mutate this year's snapshot.
            import copy as _copy
            elo_snap    = {sport: _copy.deepcopy(data['elo_ratings'].get(sport, {}))}
            pythag_snap = {sport: _copy.deepcopy(data['pythagorean'].get(sport, {}))}
            for g in games:
                g['elo_snapshot']    = elo_snap
                g['pythag_snapshot'] = pythag_snap
            all_games_by_sport[sport].extend(games)

            current_weights    = optimized_weights
            sport_games_total += len(games)
            total_games        += len(games)

            # Prune and compress patterns periodically to manage memory
            if year % 3 == 0:
                prune_patterns(data['patterns'])
                compress_patterns(data['patterns'])

        # Final validation on holdout year (never seen during training)
        if sport_games_total > 100:
            holdout_games  = fetch_season(sport, holdout_year)

            if holdout_games:
                final_correct = 0
                for game in holdout_games:
                    pred = predict_game(
                        home=game.get('home', ''),
                        away=game.get('away', ''),
                        game_date=game.get('date', ''),
                        venue=game.get('venue', ''),
                        sport=sport,
                        weights=current_weights,
                        patterns=data['patterns'],
                        team_history=data['team_history'],
                        elo_ratings=data['elo_ratings'],
                        pythagorean=data['pythagorean'],
                        standings=data['standings'].get(sport, {}),
                        team_stats=data['team_stats'],
                        espn_stats=data.get('espn_stats', {}),
                        barttorvik=data.get('barttorvik', {}),
                        sos=data.get('sos', {}),
                    )
                    home_won = game.get('home_won')
                    if home_won is not None:
                        was_right = (pred['pick'] == game.get('home', '')) == bool(home_won)
                        if was_right:
                            final_correct += 1

                final_acc = final_correct / len(holdout_games) if holdout_games else 0.0
                log.info(f'  {sport.upper()} HOLDOUT {holdout_year}: {final_acc:.1%} accuracy on {len(holdout_games)} games')
                data['training_record']['accuracy'][sport] = final_acc
            else:
                log.warning(f'  {sport}: no holdout data for {holdout_year}')
        else:
            log.warning(f'  {sport}: limited data ({sport_games_total} games) - predictions will be low confidence')

        data['weights'][sport]                          = current_weights
        data['training_record']['data_collected'][sport] = sport_games_total

        # Save progress after each sport — crash recovery
        log.info(f'  Saving progress after {sport.upper()}...')
        save_data(data)

    # Compute Strength of Schedule from final team history
    log.info('Computing strength of schedule...')
    for sport in ALL_SPORTS:
        sos_map = compute_sos(sport, data['team_history'])
        if sos_map:
            data['sos'][sport] = sos_map

    # Train XGBoost models on all collected historical games (Phase 3)
    if XGBOOST_AVAILABLE:
        log.info('Training XGBoost models...')
        global XGB_MODELS
        XGB_MODELS = train_xgboost_models(all_games_by_sport, data)
        if XGB_MODELS:
            save_xgb_models(XGB_MODELS, XGB_FEATURE_NAMES_DEFAULT)
            data['xgb_trained'] = True
            log.info(f'XGBoost trained for: {list(XGB_MODELS.keys())}')
    else:
        log.warning('XGBoost not available - using weighted-average predictions only')

    # Store historical game records for advanced model training (transformer.py, graph.py).
    # Uses sport-specific limits defined in GAME_STORAGE_LIMITS to balance training
    # quality against data.json file size. Most recent games are kept ([-limit:]).
    log.info('Storing game history for advanced model training...')
    stored_counts = {}
    data['all_games_by_sport'] = {}
    for sport, games in all_games_by_sport.items():
        if not games:
            continue
        limit = GAME_STORAGE_LIMITS.get(sport, 2500)
        data['all_games_by_sport'][sport] = games[-limit:]
        stored_counts[sport] = len(data['all_games_by_sport'][sport])
    for sport, count in stored_counts.items():
        total_available = len(all_games_by_sport[sport])
        log.info(f'  {sport}: stored {count:,} of {total_available:,} games'
                 f' (limit={GAME_STORAGE_LIMITS.get(sport, 2500):,})')

    # Final cleanup
    prune_patterns(data['patterns'])
    compress_patterns(data['patterns'])

    elapsed = time.time() - start_time
    data['training_status']              = 'trained'
    data['last_trained']                 = datetime.datetime.utcnow().isoformat()
    data['training_record']['date']      = data['last_trained']
    data['training_record']['duration_seconds'] = int(elapsed)

    log.info('=' * 55)
    log.info(f'FOUNDATION TRAINING COMPLETE')
    log.info(f'Total games: {total_games}')
    log.info(f'Duration: {int(elapsed // 60)}m {int(elapsed % 60)}s')
    log.info('=' * 55)

    # Train intelligence layers after brain.py training completes
    if MIND_AVAILABLE:
        log.info('[mind] Starting LSTM and Bayesian training...')
        try:
            train_mind(data)
            save_data(data)
            log.info('[mind] Training complete.')
        except Exception as exc:
            log.warning(f'[mind] Training failed (non-fatal): {exc}')

    if SCOUT_AVAILABLE:
        log.info('[scout] Building player and official profiles...')
        try:
            train_scout(data)
            save_data(data)
            log.info('[scout] Training complete.')
        except Exception as exc:
            log.warning(f'[scout] Training failed (non-fatal): {exc}')

    # edge.py has no training phase — runs only at update time

    return data

# ============================================================
# SECTION 21: CBS CONFIDENCE POOL OPTIMIZER
# Simulation-based: runs CBS_SIMS full-week simulations to find
# the point assignment that survives (earns most points) across
# the most possible week outcomes using all trained data.
# Locked games (started or final) keep their pick + point value.
# ============================================================

def generate_cbs_picks(data: dict) -> dict:
    """
    Generates CBS confidence pool picks using Monte Carlo week simulation.

    Algorithm:
    1. Fetch all NFL games this week from ESPN
    2. Separate into LOCKED (started/final) and UNLOCKED (not yet kicked off)
    3. Locked games keep their existing pick and point value from data.json
    4. For unlocked games: run CBS_SIMS simulations of the full week
       - Each sim plays out every game using model probabilities
       - For each possible point assignment of unlocked slots, score the sim
       - The assignment that earns the most total points across all sims wins
    5. Merge locked + newly optimized unlocked assignments
    """
    log.info('Generating CBS confidence pool picks...')
    log.info(f'Running {CBS_SIMS:,} simulations...')

    if data.get('training_status') not in ('trained',):
        log.warning('Cannot generate CBS picks - model not trained')
        return data

    games = fetch_espn_scoreboard('nfl')
    if not games:
        log.warning('No NFL games found for CBS picks')
        return data

    num_games = len(games)
    week_num  = get_week_number(games[0].get('date', '')) if games else 0
    log.info(f'CBS Week {week_num}: {num_games} games')

    # Fetch signals
    odds_data     = fetch_betting_odds('nfl')
    nfl_standings = fetch_espn_standings('nfl')
    if nfl_standings:
        data['standings']['nfl'] = nfl_standings

    # ── Build predictions for every game ─────────────────────────────────────
    game_predictions = []
    existing_locked  = {
        f"{a['home']}|||{a['away']}": a
        for a in (data.get('cbs_picks') or {}).get('assignment', [])
    }

    for game in games:
        status = (game.get('status') or '').lower()
        # Game is locked if ESPN says it's in-progress, halftime, final, or post-game
        is_locked = any(kw in status for kw in (
            'in progress', 'halftime', 'final', 'end', 'game over',
            'inprogress', 'ingame', 'post'
        ))

        weather     = get_weather(game.get('venue_lat'), game.get('venue_lon'),
                                  game.get('venue', ''), game.get('date', ''))
        odds_parsed = parse_odds_for_game(odds_data, game['home'], game['away']) if odds_data else None
        inj_h       = fetch_espn_injuries(game.get('home_id', ''), 'nfl')
        inj_a       = fetch_espn_injuries(game.get('away_id', ''), 'nfl')

        pred = predict_game(
            home=game['home'], away=game['away'],
            game_date=game.get('date', ''), venue=game.get('venue', ''),
            sport='nfl',
            weights=data['weights'].get('nfl', DEFAULT_WEIGHTS),
            patterns=data['patterns'], team_history=data['team_history'],
            elo_ratings=data['elo_ratings'], pythagorean=data['pythagorean'],
            standings=data['standings'].get('nfl', {}),
            team_stats=data['team_stats'],
            injuries_home=inj_h, injuries_away=inj_a,
            weather=weather, odds_parsed=odds_parsed,
            espn_stats=data.get('espn_stats', {}),
            barttorvik=data.get('barttorvik', {}),
            sos=data.get('sos', {}),
            game_time=game.get('game_time', ''),
        )

        # Check if this game was previously locked
        game_key   = f"{game['home']}|||{game['away']}"
        prior_lock = existing_locked.get(game_key)

        if is_locked and prior_lock:
            # Already locked — carry forward exact pick and point value
            game_predictions.append({
                'game': game, 'pred': pred,
                'is_locked': True,
                'locked_pick':  prior_lock['pick'],
                'locked_pts':   prior_lock['point_value'],
                'pick_prob':    prior_lock.get('pick_prob') or pred['pick_prob']
            })
            log.info(f"  LOCKED: {game['away']} @ {game['home']} → {prior_lock['pick']} ({prior_lock['point_value']}pts)")
        else:
            # Unlocked — available for optimization
            pick_prob = pred['pick_prob']
            if pred.get('is_guess'):
                pick_prob = 0.501  # push guesses to bottom
            game_predictions.append({
                'game': game, 'pred': pred,
                'is_locked': False,
                'pick_prob': pick_prob
            })

    # ── Separate locked vs unlocked ───────────────────────────────────────────
    locked   = [gp for gp in game_predictions if gp['is_locked']]
    unlocked = [gp for gp in game_predictions if not gp['is_locked']]

    locked_pts_used  = {gp['locked_pts'] for gp in locked}
    available_pts    = [p for p in range(1, num_games + 1) if p not in locked_pts_used]
    n_unlocked       = len(unlocked)

    log.info(f'  Locked: {len(locked)} games | Unlocked: {n_unlocked} games')
    log.info(f'  Available point slots: {sorted(available_pts)}')

    # ── Simulate full week CBS_SIMS times ─────────────────────────────────────
    # For each sim: play out all games probabilistically, then use
    # the Hungarian algorithm (linear_sum_assignment) for provably optimal
    # point assignment. Track cumulative score per assignment.

    from scipy.optimize import linear_sum_assignment

    assignment_scores: Dict[tuple, float] = {}

    for sim_i in range(CBS_SIMS):
        # Simulate outcomes for all unlocked games
        sim_outcomes: Dict[int, str] = {}
        for i, gp in enumerate(unlocked):
            prob   = gp['pick_prob']
            winner = gp['pred']['pick'] if np.random.random() < prob else (
                gp['game']['away'] if gp['pred']['pick'] == gp['game']['home']
                else gp['game']['home']
            )
            sim_outcomes[i] = winner

        # Build cost matrix: rows=games, cols=point slots
        pts_sorted = sorted(available_pts, reverse=True)
        cost = np.zeros((n_unlocked, n_unlocked))
        for i, gp in enumerate(unlocked):
            for j, pts in enumerate(pts_sorted):
                will_win = 1 if sim_outcomes[i] == gp['pred']['pick'] else 0
                cost[i][j] = will_win * pts

        # Hungarian algorithm: maximize (negate for minimization)
        row_ind, col_ind = linear_sum_assignment(-cost)
        assignment_map   = {row_ind[k]: pts_sorted[col_ind[k]] for k in range(n_unlocked)}

        # Score this sim's assignment
        total = sum(
            assignment_map[i] for i, gp in enumerate(unlocked)
            if sim_outcomes[i] == gp['pred']['pick']
        )
        key = tuple(assignment_map.get(i, 1) for i in range(n_unlocked))
        assignment_scores[key] = assignment_scores.get(key, 0.0) + total

    # ── Pick best assignment ──────────────────────────────────────────────────
    if assignment_scores:
        best_key = max(assignment_scores, key=lambda k: assignment_scores[k])
        best_pts_per_game = list(best_key)
    else:
        # Fallback: sort by prob
        unlocked_sorted = sorted(range(n_unlocked),
                                  key=lambda i: unlocked[i]['pick_prob'], reverse=True)
        pts_sorted_asc  = sorted(available_pts, reverse=True)
        best_pts_per_game = [0] * n_unlocked
        for rank, i in enumerate(unlocked_sorted):
            best_pts_per_game[i] = pts_sorted_asc[rank] if rank < len(pts_sorted_asc) else 1

    # ── Build final assignment list ───────────────────────────────────────────
    assignment = []
    expected_pts = 0.0

    # Add locked games first
    for gp in locked:
        pp = gp['pick_prob']
        assignment.append({
            'point_value':  gp['locked_pts'],
            'home':         gp['game']['home'],
            'away':         gp['game']['away'],
            'pick':         gp['locked_pick'],
            'pick_prob':    pp,
            'display_pct':  f'{pp:.0%}' if pp and pp > 0.502 else '~50%',
            'confidence':   gp['pred']['confidence'],
            'is_guess':     gp['pred'].get('is_guess', False),
            'is_locked':    True,
            'signals':      gp['pred'].get('signals_fired', []),
            'date':         gp['game'].get('date', '')
        })
        expected_pts += (pp or 0.5) * gp['locked_pts']

    # Add unlocked games with simulation-optimized point values
    for i, gp in enumerate(unlocked):
        pt  = best_pts_per_game[i] if i < len(best_pts_per_game) else 1
        pp  = gp['pick_prob']
        is_guess = gp['pred'].get('is_guess', False) or pp <= 0.502
        assignment.append({
            'point_value':  pt,
            'home':         gp['game']['home'],
            'away':         gp['game']['away'],
            'pick':         gp['pred']['pick'],
            'pick_prob':    pp if not is_guess else None,
            'display_pct':  f'{pp:.0%}' if not is_guess else '~50%',
            'confidence':   gp['pred']['confidence'],
            'is_guess':     is_guess,
            'is_locked':    False,
            'signals':      gp['pred'].get('signals_fired', []),
            'date':         gp['game'].get('date', '')
        })
        expected_pts += pp * pt

    # Sort by point value desc for display
    assignment.sort(key=lambda x: x['point_value'], reverse=True)
    max_possible = sum(range(1, num_games + 1))

    data['cbs_picks'] = {
        'generated_at':        datetime.datetime.utcnow().isoformat(),
        'week_number':         week_num,
        'num_games':           num_games,
        'max_possible_points': max_possible,
        'expected_points':     round(expected_pts, 1),
        'simulations_run':     CBS_SIMS,
        'assignment':          assignment
    }

    log.info(f'CBS picks: {num_games} games | {len(locked)} locked | expected {expected_pts:.1f}/{max_possible}pts')
    return data

# ============================================================
# SECTION 22: MARCH MADNESS BRACKET OPTIMIZER
# Simulates the full 64-team tournament MONTE_CARLO_SIMS times.
# Scores each simulated bracket using CBS round multipliers:
#   Round of 64: 1pt, Round of 32: 2pt, Sweet 16: 4pt,
#   Elite 8: 8pt, Final Four: 16pt, Championship: 32pt
# Picks the bracket that maximizes CBS points across all sims.
# Uses trained H2H/Pythagorean data blended with seed history.
# Seeds must be present or bracket generation is skipped.
# Separate runs for men's and women's.
# ============================================================

# CBS scoring multipliers by round (0-indexed: round 0 = Round of 64)
CBS_ROUND_PTS = [1, 2, 4, 8, 16, 32]

# Standard bracket structure: 4 regions × 16 teams each
# Seeds 1-16 in each region, first round pairings: 1v16, 2v15, 3v14, 4v13, 5v12, 6v11, 7v10, 8v9
FIRST_ROUND_PAIRINGS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]


def _get_team_prob(team_a: str, seed_a: int, team_b: str, seed_b: int,
                   sport: str, data: dict) -> float:
    """
    Returns probability that team_a beats team_b.
    Blends trained model data with historical seed matchup rates.
    """
    pred = predict_game(
        home=team_a, away=team_b,
        game_date='', venue='',
        sport=sport,
        weights=data['weights'].get(sport, DEFAULT_WEIGHTS),
        patterns=data['patterns'],
        team_history=data['team_history'],
        elo_ratings=data['elo_ratings'],
        pythagorean=data['pythagorean'],
        standings=data['standings'].get(sport, {}),
        team_stats=data['team_stats'],
        espn_stats=data.get('espn_stats', {}),
        barttorvik=data.get('barttorvik', {}),
        sos=data.get('sos', {}),
    )
    model_prob = pred['home_prob']

    # Blend with seed-based probability (model 60%, seeds 40%)
    seed_prob = _get_seed_probability(seed_a, seed_b)
    if seed_prob is not None:
        return model_prob * 0.6 + seed_prob * 0.4
    return model_prob


def _fetch_tournament_field(sport_key: str) -> Optional[List[dict]]:
    """
    Fetches the tournament bracket from ESPN.
    Returns list of team dicts with name, seed, region if seeds are out.
    Returns None if tournament bracket not yet available.

    ESPN bracket endpoint provides the full field once seeds are announced.
    Falls back to scoreboard if bracket endpoint not available.
    """
    path = ESPN_PATHS.get(sport_key)
    if not path:
        return None

    # Try ESPN bracket endpoint first
    bracket_data = fetch_json(
        f'{ESPN_BASE}/{path}/tournament/bracket',
        params={'limit': 100},
        max_retries=2, timeout=20
    )

    if bracket_data:
        teams = _parse_espn_bracket(bracket_data)
        if teams and len(teams) >= 32:
            log.info(f'Tournament field ({sport_key}): {len(teams)} teams via bracket endpoint')
            return teams

    # Fallback: scoreboard with groups param to check if tournament games exist
    params = {'limit': 200, 'groups': '50' if sport_key == 'ncaabm' else '49'}
    scoreboard = fetch_json(f'{ESPN_BASE}/{path}/scoreboard', params=params)
    if not scoreboard:
        return None

    # Check if any events have seed data (indicates tournament is happening)
    events = scoreboard.get('events', [])
    has_seeds = any(
        safe_int((c.get('curatedRank') or {}).get('current'))
        for e in events if isinstance(e, dict)
        for comp in (e.get('competitions') or [{}])
        for c in (comp.get('competitors') or [])
        if isinstance(c, dict)
    )

    if not has_seeds:
        log.info(f'No seed data found for {sport_key} — tournament not yet seeded')
        return None

    # Parse teams from scoreboard events
    teams = []
    seen  = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        for comp in (event.get('competitions') or []):
            for c in (comp.get('competitors') or []):
                if not isinstance(c, dict):
                    continue
                team = c.get('team') or {}
                name = team.get('displayName', '').strip()
                seed = safe_int((c.get('curatedRank') or {}).get('current')) or 0
                region = str(c.get('conferenceId', '') or '')
                if name and name not in seen:
                    seen.add(name)
                    teams.append({'name': name, 'seed': seed, 'region': region})

    return teams if teams else None


def _parse_espn_bracket(data: dict) -> List[dict]:
    """Parses ESPN bracket API response into team list."""
    teams = []
    seen  = set()
    for region in (data.get('bracket') or {}).get('regions', []):
        region_name = region.get('name', '')
        for team_data in (region.get('teams') or []):
            if not isinstance(team_data, dict):
                continue
            name = (team_data.get('team') or {}).get('displayName', '').strip()
            seed = safe_int(team_data.get('seed')) or 0
            if name and name not in seen:
                seen.add(name)
                teams.append({'name': name, 'seed': seed, 'region': region_name})
    return teams


def generate_march_bracket(data: dict, gender: str) -> dict:
    """
    Generates an optimized March Madness bracket using Monte Carlo simulation.
    gender: 'mens' or 'womens'

    1. Checks if seeds are available — if not, sets no_seeds flag and returns.
    2. Fetches the full 64-team field with seeds and regions.
    3. Runs MONTE_CARLO_SIMS full-tournament simulations using trained model.
    4. Scores each sim with CBS round multipliers (1/2/4/8/16/32).
    5. For each game, picks the team that appeared in winning brackets most often
       weighted by CBS points earned when that pick was correct.
    6. Outputs round-by-round results optimized for CBS pool points.
    """
    sport_key  = 'ncaabm' if gender == 'mens' else 'ncaabw'
    result_key = 'march_mens' if gender == 'mens' else 'march_womens'

    log.info(f'Generating March Madness bracket ({gender})...')

    # ── Check for seeds ───────────────────────────────────────────────────────
    field = _fetch_tournament_field(sport_key)

    if not field:
        log.info(f'  Seeds not yet available for {sport_key}')
        data[result_key] = {
            'generated_at':      datetime.datetime.utcnow().isoformat(),
            'seeds_available':   False,
            'message':           'Seeds not yet released. Check back after Selection Sunday.',
            'recommended_bracket': [],
            'rounds':            {},
            'upset_alerts':      [],
            'simulations_run':   0
        }
        return data

    log.info(f'  Field: {len(field)} teams seeded | Running {MONTE_CARLO_SIMS:,} simulations...')

    # ── Group teams by region, build first-round matchups ─────────────────────
    # Group by region
    by_region: Dict[str, List[dict]] = {}
    for team in field:
        r = team.get('region', 'Unknown')
        if r not in by_region:
            by_region[r] = []
        by_region[r].append(team)

    # Sort each region by seed
    for r in by_region:
        by_region[r].sort(key=lambda t: t['seed'])

    # If we don't have 4 clean regions of 16, use all teams sorted by seed
    all_teams_sorted = sorted(field, key=lambda t: t['seed'])

    # Build first round: pair 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15 per region
    first_round_games = []
    regions_used      = [r for r in by_region if len(by_region[r]) >= 8]

    if len(regions_used) >= 4:
        for region in regions_used[:4]:
            region_teams = by_region[region][:16]
            seed_map = {t['seed']: t for t in region_teams}
            for s1, s2 in FIRST_ROUND_PAIRINGS:
                t1 = seed_map.get(s1)
                t2 = seed_map.get(s2)
                if t1 and t2:
                    first_round_games.append((t1, t2, region))
    else:
        # Fallback: pair teams sequentially by seed across all regions
        pairs = list(zip(all_teams_sorted[:32], reversed(all_teams_sorted[:32])))
        for t1, t2 in pairs:
            first_round_games.append((t1, t2, 'Tournament'))

    if not first_round_games:
        log.warning(f'  Could not build bracket structure for {sport_key}')
        return data

    log.info(f'  First round: {len(first_round_games)} games across {len(set(g[2] for g in first_round_games))} regions')

    # ── Pre-compute win probabilities for all possible matchups ───────────────
    # Cache to avoid re-running predict_game thousands of times
    prob_cache: Dict[str, float] = {}

    def get_prob(t_a: dict, t_b: dict) -> float:
        key = f"{t_a['name']}|||{t_b['name']}"
        if key not in prob_cache:
            prob_cache[key] = _get_team_prob(
                t_a['name'], t_a['seed'],
                t_b['name'], t_b['seed'],
                sport_key, data
            )
        return prob_cache[key]

    # ── Monte Carlo simulation ────────────────────────────────────────────────
    # Track for each (team_name, round) how many CBS points that team earned
    # when correctly predicted in that round, summed across all simulations.
    # Final bracket picks the team with highest weighted CBS points per slot.

    # round_team_pts[round][matchup_key][team] = total CBS points earned
    # matchup_key = sorted tuple of original first-round seeds/teams
    n_rounds = 6  # 64→32→16→8→4→2→1

    # For each sim, track who wins each round
    # team_round_wins[team_name][round] = count of sims where this team reached this round and won
    team_round_wins: Dict[str, List[int]] = {}
    for t in field:
        team_round_wins[t['name']] = [0] * n_rounds

    # Also track the specific bracket (who plays who in later rounds)
    # bracket_picks[round][matchup_idx] = {team: wins}
    bracket_picks: List[Dict[int, Dict[str, int]]] = [{} for _ in range(n_rounds)]

    for sim_i in range(MONTE_CARLO_SIMS):
        # Run full tournament for this sim
        current_round_teams = [(t1, t2) for t1, t2, _ in first_round_games]

        for round_idx in range(n_rounds):
            if not current_round_teams:
                break

            next_round = []
            matchup_idx = 0

            for t1, t2 in current_round_teams:
                prob_t1_wins = get_prob(t1, t2)
                winner = t1 if np.random.random() < prob_t1_wins else t2

                # Track this win
                team_round_wins[winner['name']][round_idx] += 1

                # Track bracket picks for this specific matchup
                mkey = matchup_idx
                if mkey not in bracket_picks[round_idx]:
                    bracket_picks[round_idx][mkey] = {}
                bracket_picks[round_idx][mkey][winner['name']] =                     bracket_picks[round_idx][mkey].get(winner['name'], 0) + 1

                next_round.append(winner)
                matchup_idx += 1

            # Pair winners for next round
            current_round_teams = list(zip(next_round[::2], next_round[1::2]))

    # ── Build recommended bracket ─────────────────────────────────────────────
    # For each matchup in each round, pick the team that won more simulations
    # weighted by CBS points for that round.

    rounds_output: Dict[str, List[dict]] = {}
    upset_alerts: List[dict] = []
    recommended_bracket: List[dict] = []  # flat list for backward compat

    round_names = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final Four', 'Championship']

    # Rebuild bracket deterministically using picks from simulation majority
    sim_bracket = [(t1, t2) for t1, t2, _ in first_round_games]
    first_round_regions = {i: r for i, (_, _, r) in enumerate(first_round_games)}

    for round_idx in range(n_rounds):
        if not sim_bracket:
            break

        round_name    = round_names[round_idx] if round_idx < len(round_names) else f'Round {round_idx+1}'
        round_entries = []
        next_bracket  = []
        pts_this_round = CBS_ROUND_PTS[round_idx] if round_idx < len(CBS_ROUND_PTS) else 32

        for matchup_idx, (t1, t2) in enumerate(sim_bracket):
            wins_t1 = team_round_wins[t1['name']][round_idx]
            wins_t2 = team_round_wins[t2['name']][round_idx]
            total   = wins_t1 + wins_t2

            # Pick the team that won this specific matchup more often in sims
            mp = bracket_picks[round_idx].get(matchup_idx, {})
            mp_t1 = mp.get(t1['name'], wins_t1)
            mp_t2 = mp.get(t2['name'], wins_t2)
            mp_total = mp_t1 + mp_t2

            if mp_total > 0:
                pick = t1 if mp_t1 >= mp_t2 else t2
                sim_pct = max(mp_t1, mp_t2) / mp_total
            else:
                pick = t1 if wins_t1 >= wins_t2 else t2
                sim_pct = max(wins_t1, wins_t2) / max(total, 1)

            loser = t2 if pick == t1 else t1

            entry = {
                'round':        round_name,
                'round_idx':    round_idx,
                'team1':        t1['name'],
                'team1_seed':   t1['seed'],
                'team2':        t2['name'],
                'team2_seed':   t2['seed'],
                'pick':         pick['name'],
                'pick_seed':    pick['seed'],
                'round_pts':    pts_this_round,
                'sim_pct':      round(sim_pct, 3),
                'region':       first_round_regions.get(matchup_idx // max(1, 2**round_idx), '')
            }
            round_entries.append(entry)
            recommended_bracket.append({
                'home': t1['name'], 'away': t2['name'],
                'home_seed': t1['seed'], 'away_seed': t2['seed'],
                'recommended': pick['name'], 'sim_confidence': sim_pct,
                'home_prob': get_prob(t1, t2), 'round': round_name
            })

            # Upset alert: underdog (higher seed number) is our pick
            seed_diff = abs(t1['seed'] - t2['seed'])
            if seed_diff >= 4 and pick['seed'] > min(t1['seed'], t2['seed']):
                upset_alerts.append({
                    'round':    round_name,
                    'pick':     pick['name'],
                    'seed':     pick['seed'],
                    'against':  loser['name'],
                    'message':  f"UPSET PICK: ({pick['seed']}) {pick['name']} over ({loser['seed']}) {loser['name']} — {sim_pct:.0%} of sims"
                })

            next_bracket.append(pick)

        rounds_output[round_name] = round_entries
        sim_bracket = list(zip(next_bracket[::2], next_bracket[1::2]))

    # ── Log summary ───────────────────────────────────────────────────────────
    champ_entries = rounds_output.get('Championship', [])
    champ = champ_entries[0]['pick'] if champ_entries else 'Unknown'
    ff_entries = rounds_output.get('Final Four', [])
    ff_teams = [e['pick'] for e in ff_entries]
    log.info(f'  Champion pick: {champ}')
    log.info(f'  Final Four: {", ".join(ff_teams)}')
    log.info(f'  Upset alerts: {len(upset_alerts)}')

    data[result_key] = {
        'generated_at':        datetime.datetime.utcnow().isoformat(),
        'seeds_available':     True,
        'simulations_run':     MONTE_CARLO_SIMS,
        'field_size':          len(field),
        'rounds':              rounds_output,
        'recommended_bracket': recommended_bracket,
        'upset_alerts':        upset_alerts,
        'champion_pick':       champ,
        'final_four':          ff_teams
    }

    return data


def _get_seed_probability(home_seed: int, away_seed: int) -> Optional[float]:
    """
    Returns historical win probability for the better-seeded team.
    Returns None if seeds are unknown (0) or equal.
    """
    if not home_seed or not away_seed or home_seed == away_seed:
        return None

    better_seed  = min(home_seed, away_seed)
    worse_seed   = max(home_seed, away_seed)
    better_is_home = home_seed < away_seed

    prob_better_wins = SEED_UPSET_RATES.get((better_seed, worse_seed))

    if prob_better_wins is None:
        return None

    # Return probability that HOME TEAM wins
    return prob_better_wins if better_is_home else (1.0 - prob_better_wins)

# ============================================================
# SECTION 23: DAILY UPDATE
# Fetches current schedules, standings, predictions for all sports.
# Runs on a schedule set in train.yml.
# ============================================================

def run_daily_update(data: dict) -> dict:
    """
    Updates predictions for all sports using current schedules.
    Also auto-verifies past pending predictions against actual results.
    """
    if data.get('training_status') not in ('trained',):
        log.warning('Skipping daily update - model not trained')
        return data

    log.info('Running daily update...')

    # Step 1: Auto-verify pending predictions
    data = _auto_verify_predictions(data)

    # Run edge intelligence (line movement + sentiment) before prediction generation
    if EDGE_AVAILABLE:
        log.info('[edge] Fetching market and sentiment signals...')
        try:
            update_edge(data)
            log.info('[edge] Complete.')
        except Exception as exc:
            log.warning(f'[edge] Update failed (non-fatal): {exc}')

    # Run scout intelligence (lineup adjustments + official assignments)
    if SCOUT_AVAILABLE:
        log.info('[scout] Fetching lineup and official data...')
        try:
            update_scout(data)
            log.info('[scout] Complete.')
        except Exception as exc:
            log.warning(f'[scout] Update failed (non-fatal): {exc}')

    # Step 2: Fetch standings for all sports
    for sport in ALL_SPORTS:
        log.info(f'Fetching standings: {sport}')
        standings = fetch_espn_standings(sport)
        if standings:
            data['standings'][sport] = standings
        time.sleep(0.5)  # Rate limit

    # Step 3: Generate predictions for all sports
    odds_cache: Dict[str, Optional[List[dict]]] = {}

    for sport in ALL_SPORTS:
        log.info(f'Generating predictions: {sport}')

        games = fetch_espn_scoreboard(sport)
        if not games:
            log.info(f'  No upcoming games for {sport}')
            continue

        # Gate: skip predictions for sports outside their regular season window.
        # Prevents spring training, preseason, and off-season games from being predicted.
        _today_mmdd = datetime.date.today().strftime('%m-%d')
        _start_str, _end_str = SEASON_WINDOWS.get(sport, ('01-01', '12-31'))
        if sport in CROSS_YEAR_SPORTS:
            _in_season = (_today_mmdd >= _start_str) or (_today_mmdd <= _end_str)
        else:
            _in_season = _start_str <= _today_mmdd <= _end_str
        if not _in_season:
            log.info(f'  {sport} is out of regular season ({_start_str}–{_end_str}) — skipping predictions')
            continue

        # Cache odds per sport to avoid multiple API calls
        if sport not in odds_cache:
            odds_cache[sport] = fetch_betting_odds(sport)

        sport_predictions = []

        for game in games:
            weather = None
            if sport in OUTDOOR_SPORTS:
                weather = get_weather(
                    game.get('venue_lat'),
                    game.get('venue_lon'),
                    game.get('venue', ''),
                    game.get('date', '')
                )

            odds_parsed = None
            if odds_cache.get(sport):
                odds_parsed = parse_odds_for_game(
                    odds_cache[sport], game['home'], game['away']
                )

            pred = predict_game(
                home=game['home'],
                away=game['away'],
                game_date=game.get('date', ''),
                venue=game.get('venue', ''),
                sport=sport,
                weights=data['weights'].get(sport, DEFAULT_WEIGHTS),
                patterns=data['patterns'],
                team_history=data['team_history'],
                elo_ratings=data['elo_ratings'],
                pythagorean=data['pythagorean'],
                standings=data['standings'].get(sport, {}),
                team_stats=data['team_stats'],
                weather=weather,
                odds_parsed=odds_parsed,
                espn_stats=data.get('espn_stats', {}),
                barttorvik=data.get('barttorvik', {}),
                sos=data.get('sos', {}),
                game_time=game.get('game_time', ''),
            )

            cal_note = get_calibration_note(
                data['calibration'], sport, pred['pick_prob']
            )

            game_date_str = game.get('date', '')
            game_id       = game.get('id') or f"{sport}_{game['home']}_{game['away']}_{game_date_str[:10]}"

            # --- Intelligence layer: refine raw probability ---
            raw_prob = pred['home_prob']
            win_prob = raw_prob

            # Step 1: Scout — lineup and official adjustments
            if SCOUT_AVAILABLE:
                try:
                    lineup_adj   = data.get('scout', {}).get('lineup_adjustments', {}).get(str(game_id), {})
                    official_adj = data.get('scout', {}).get('official_assignments', {}).get(str(game_id), {})
                    scout_delta  = lineup_adj.get('home_adjustment', 0.0)
                    off_impact   = official_adj.get('impact', 0.0)
                    off_dir      = official_adj.get('impact_direction', 'neutral')
                    if off_dir == 'home':
                        scout_delta += off_impact
                    elif off_dir == 'away':
                        scout_delta -= off_impact
                    win_prob = max(0.02, min(0.98, win_prob + scout_delta))
                except Exception:
                    pass  # Scout failed silently

            # Step 2: Edge — sharp money, sentiment, and prediction market adjustments
            if EDGE_AVAILABLE:
                try:
                    sharp_signal      = data.get('edge', {}).get('sharp_signals', {}).get(str(game_id), {})
                    sentiment_signal  = data.get('edge', {}).get('sentiment', {}).get(str(game_id), {})
                    pm_signal         = data.get('edge', {}).get('prediction_markets', {}).get(str(game_id), {})
                    edge_delta = (
                        sharp_signal.get('prob_delta', 0.0) +
                        sentiment_signal.get('prob_adjustment', 0.0) +
                        pm_signal.get('prob_adjustment', 0.0)
                    )
                    win_prob = max(0.02, min(0.98, win_prob + edge_delta))
                except Exception:
                    pass  # Edge failed silently

            # Step 3: Mind — calibration + momentum refinement (final word)
            refined_prob      = win_prob
            confidence_lower  = None
            confidence_upper  = None
            momentum_score    = None
            if MIND_AVAILABLE:
                try:
                    mind_result       = mind_refine(game, win_prob, data, sport)
                    refined_prob      = mind_result['refined_prob']
                    confidence_lower  = mind_result['confidence_lower']
                    confidence_upper  = mind_result['confidence_upper']
                    momentum_score    = mind_result['momentum_score']
                except Exception:
                    refined_prob = win_prob  # Mind failed silently

            sport_predictions.append({
                'id':                  game_id,
                'home':                game['home'],
                'away':                game['away'],
                'date':                game_date_str,
                'venue':               game.get('venue', ''),
                'pick':                pred['pick'],
                'home_prob':           refined_prob,
                'win_probability':     refined_prob,
                'raw_win_probability': raw_prob,
                'pick_prob':           pred['pick_prob'],
                'confidence':          pred['confidence'],
                'confidence_lower':    confidence_lower,
                'confidence_upper':    confidence_upper,
                'momentum_score':      momentum_score,
                'is_guess':            pred['is_guess'],
                'signals':             pred.get('signals_fired', []),
                'cal_note':            cal_note,
                'status':              'pending',
                'generated_at':        datetime.datetime.utcnow().isoformat(),
            })

        data['predictions'][sport] = sport_predictions
        log.info(f'  {sport}: {len(sport_predictions)} predictions generated')
        time.sleep(0.3)  # Rate limit between sports

    # Step 4: Generate betting picks
    data = _generate_betting_picks(data, odds_cache)

    # Update mind calibration with newly resolved predictions
    if MIND_AVAILABLE:
        log.info('[mind] Updating calibration with resolved predictions...')
        try:
            update_mind(data)
            log.info('[mind] Calibration updated.')
        except Exception as exc:
            log.warning(f'[mind] Update failed (non-fatal): {exc}')

    # Step 5: Incremental XGBoost refit if buffer has enough new games
    if data.get('xgb_trained') and XGBOOST_AVAILABLE:
        data = retrain_xgboost_incremental(data)

    # Step 6: Refresh ESPN stats and Barttorvik for fresh signal data
    log.info('Refreshing ESPN stats...')
    for sport in ALL_SPORTS:
        es = fetch_espn_stats(sport)
        if es:
            data['espn_stats'][sport] = es
        time.sleep(0.2)

    btv = fetch_barttorvik()
    if btv:
        data['barttorvik'] = btv

    data['last_daily_update'] = datetime.datetime.utcnow().isoformat()
    log.info('Daily update complete')

    return data


def _auto_verify_predictions(data: dict) -> dict:
    """
    Automatically verifies pending predictions against completed game results.
    Updates running record and calibration with actual outcomes.
    """
    now    = datetime.datetime.utcnow()
    cutoff = now - datetime.timedelta(hours=6)  # Give games time to complete

    for sport in ALL_SPORTS:
        sport_preds = data['predictions'].get(sport, [])
        if not isinstance(sport_preds, list):
            continue

        # Find pending predictions old enough to have finished
        pending = [
            p for p in sport_preds
            if isinstance(p, dict)
            and p.get('status') == 'pending'
            and p.get('date', '')
        ]

        if not pending:
            continue

        # Group by date to minimize API calls
        by_date: Dict[str, List[dict]] = {}
        for pred in pending:
            pred_date = pred.get('date', '')[:10]
            if not pred_date:
                continue

            try:
                pred_dt = datetime.datetime.strptime(pred_date, '%Y-%m-%d')
            except (ValueError, TypeError):
                continue

            # Only check games that should be finished
            if pred_dt > cutoff.replace(tzinfo=None):
                continue

            date_key = pred_date.replace('-', '')  # YYYYMMDD
            if date_key not in by_date:
                by_date[date_key] = []
            by_date[date_key].append(pred)

        for date_key, date_preds in by_date.items():
            actual_results = fetch_espn_past_scores(sport, date_key)
            if not actual_results:
                continue

            for pred in date_preds:
                home = pred.get('home', '')
                away = pred.get('away', '')

                # Find matching actual result — use fuzzy name match to handle
                # ESPN occasionally returning shortened names in past scores
                actual = next(
                    (r for r in actual_results
                     if _names_match(r.get('home', ''), home)
                     and _names_match(r.get('away', ''), away)),
                    None
                )

                if not actual:
                    continue

                actual_home_won = actual.get('home_won', False)
                pick_is_home    = pred.get('pick') == home
                was_correct     = pick_is_home == actual_home_won

                pred['status']         = 'correct' if was_correct else 'wrong'
                pred['auto_verified']  = True
                pred['actual_result']  = 'home_won' if actual_home_won else 'away_won'

                # Update running record
                if sport not in data['running_record']:
                    data['running_record'][sport] = {'wins': 0, 'losses': 0}

                if was_correct:
                    data['running_record'][sport]['wins'] += 1
                else:
                    data['running_record'][sport]['losses'] += 1

                # Update calibration
                pick_prob = safe_float(pred.get('pick_prob')) or 0.5
                update_calibration(data['calibration'], sport, pick_prob, was_correct)

                # Phase 4: Buffer resolved game for incremental XGBoost refit
                if data.get('xgb_trained'):
                    try:
                        fv = build_feature_vector(
                            home=home, away=away,
                            game_date=pred.get('date', ''),
                            venue=pred.get('venue', ''),
                            sport=sport,
                            patterns=data.get('patterns', {}),
                            team_history=data.get('team_history', {}),
                            elo_ratings=data.get('elo_ratings', {}),
                            pythagorean=data.get('pythagorean', {}),
                            standings=data.get('standings', {}).get(sport, {}),
                            team_stats=data.get('team_stats', {}),
                            espn_stats=data.get('espn_stats', {}),
                            barttorvik=data.get('barttorvik', {}),
                            sos=data.get('sos', {}),
                        )
                        if sport not in data['training_buffer']:
                            data['training_buffer'][sport] = []
                        data['training_buffer'][sport].append({
                            'features':  fv.tolist(),
                            'home_won':  actual_home_won,
                            'date':      pred.get('date', ''),
                        })
                    except Exception:
                        pass

                # Update closing line value if we have odds
                # (tracks whether pick was on right side of where line ended up)
                if pred.get('signals') and 'spread' in pred.get('signals', []):
                    data['closing_line_values'].append({
                        'sport':       sport,
                        'pick':        pred['pick'],
                        'was_correct': was_correct,
                        'pick_prob':   pick_prob,
                        'date':        pred.get('date', '')
                    })

    # Expire predictions that are still pending and more than 7 days old.
    # These are games ESPN will never resolve (cancelled, postponed, wrong teams/dates).
    # Marked 'expired' so they are excluded from both wins and losses in the running record.
    seven_days_ago = now - datetime.timedelta(days=7)
    for sport in ALL_SPORTS:
        sport_preds = data['predictions'].get(sport, [])
        if not isinstance(sport_preds, list):
            continue
        for pred in sport_preds:
            if not isinstance(pred, dict):
                continue
            if pred.get('status') != 'pending':
                continue
            pred_date = pred.get('date', '')[:10]
            if not pred_date:
                continue
            try:
                pred_dt = datetime.datetime.strptime(pred_date, '%Y-%m-%d')
                if pred_dt < seven_days_ago:
                    pred['status'] = 'expired'
                    log.info(
                        f'Expired stale prediction: {pred.get("home")} vs '
                        f'{pred.get("away")} ({pred_date})'
                    )
            except (ValueError, TypeError):
                continue

    return data


def _generate_betting_picks(data: dict, odds_cache: dict) -> dict:
    """
    Generates comprehensive betting picks across ALL sports.
    - Safe bet:   highest confidence pick (pick_prob >= 0.68) with moneyline
    - Value bet:  largest model_prob - market_implied_prob gap (>= 0.08)
    - Longshot:   model gives 35-45%, moneyline +220 or better, positive EV
    - Parlay:     2-4 games at 65%+, real payout after vig, combined probability
    """
    all_pending = []

    for sport in ALL_SPORTS:
        preds = data['predictions'].get(sport, [])
        odds_data = odds_cache.get(sport)

        for p in preds:
            if not isinstance(p, dict) or p.get('status') != 'pending':
                continue

            pick_prob = safe_float(p.get('pick_prob')) or 0.5

            # Find moneyline for this pick from odds
            ml_odds = None
            market_prob = None
            if odds_data:
                parsed = parse_odds_for_game(odds_data, p.get('home', ''), p.get('away', ''))
                if parsed:
                    market_prob = safe_float(parsed.get('ml_prob'))
                    # Reconstruct approximate American moneyline from market_prob
                    if market_prob is not None and market_prob > 0:
                        pick_is_home = p.get('pick') == p.get('home')
                        prob_for_pick = market_prob if pick_is_home else 1.0 - market_prob
                        if prob_for_pick >= 0.5:
                            ml_odds = -round((prob_for_pick / (1.0 - prob_for_pick)) * 100)
                        else:
                            ml_odds = round(((1.0 - prob_for_pick) / prob_for_pick) * 100)

            all_pending.append({
                **p,
                'sport':       sport,
                'ml_odds':     ml_odds,
                'market_prob': market_prob,
            })

    if not all_pending:
        return data

    # ── Safe bet ────────────────────────────────────────────────────────────
    safe_candidates = [
        p for p in all_pending
        if (safe_float(p.get('pick_prob')) or 0) >= 0.68 and p.get('ml_odds') is not None
    ]
    safe_candidates.sort(key=lambda p: safe_float(p.get('pick_prob')) or 0, reverse=True)
    safe_out = _format_betting_pick_v2(safe_candidates[0]) if safe_candidates else None

    # ── Value bet ────────────────────────────────────────────────────────────
    # True value = model probability significantly higher than market
    value_candidates = []
    for p in all_pending:
        mp = safe_float(p.get('pick_prob')) or 0
        mkt = safe_float(p.get('market_prob'))
        if mkt is None:
            continue
        pick_is_home = p.get('pick') == p.get('home')
        mkt_for_pick = mkt if pick_is_home else 1.0 - mkt
        edge = mp - mkt_for_pick
        if edge >= 0.08:
            value_candidates.append({**p, 'edge': edge})
    value_candidates.sort(key=lambda p: p.get('edge', 0), reverse=True)
    value_out = _format_betting_pick_v2(value_candidates[0]) if value_candidates else None

    # ── Longshot ─────────────────────────────────────────────────────────────
    # Model says 35-45%, but moneyline +220 or better → positive EV
    longshot_candidates = []
    for p in all_pending:
        mp   = safe_float(p.get('pick_prob')) or 0
        ml   = p.get('ml_odds')
        if ml is None or ml <= 0:
            continue  # Need positive moneyline (underdog)
        if not (0.35 <= mp <= 0.45):
            continue
        decimal_odds = 1.0 + ml / 100.0
        ev = mp * decimal_odds - 1.0
        if ev > 0 and ml >= 220:
            longshot_candidates.append({**p, 'ev': ev, 'decimal_odds': decimal_odds})
    longshot_candidates.sort(key=lambda p: p.get('ev', 0), reverse=True)
    longshot_out = _format_betting_pick_v2(longshot_candidates[0], stake=10.0) if longshot_candidates else None

    # ── Parlay ───────────────────────────────────────────────────────────────
    parlay_legs = [
        p for p in all_pending
        if (safe_float(p.get('pick_prob')) or 0) >= 0.65 and p.get('ml_odds') is not None
    ][:4]

    parlay_out = None
    if len(parlay_legs) >= 2:
        combined_true = 1.0
        for leg in parlay_legs:
            combined_true *= (safe_float(leg.get('pick_prob')) or 0.5)

        # Calculate true payout from individual moneylines (multiplicative decimal odds)
        parlay_decimal = 1.0
        for leg in parlay_legs:
            ml = leg.get('ml_odds', -110)
            dec = (1.0 + ml / 100.0) if ml > 0 else (1.0 + 100.0 / abs(ml))
            parlay_decimal *= dec

        stake     = 10.0
        vig_factor = 0.92  # Approximate vig deduction for parlays
        net_payout = stake * parlay_decimal * vig_factor - stake

        parlay_out = {
            'legs':          [_format_betting_pick_v2(p) for p in parlay_legs],
            'combined_prob': round(combined_true, 4),
            'bet_10_wins':   round(net_payout, 2),
            'true_odds':     f'+{round(parlay_decimal * 100 * vig_factor - 100)}',
            'note':          f'{len(parlay_legs)}-leg parlay. All legs must win.',
        }

    data['betting_picks'] = {
        'generated_at': datetime.datetime.utcnow().isoformat(),
        'safe_bet':     safe_out,
        'value_bet':    value_out,
        'longshot':     longshot_out,
        'parlay':       parlay_out,
    }

    return data


def moneyline_to_payout(american_odds: float, stake: float = 10.0) -> float:
    """Calculates net profit from american odds and stake."""
    if american_odds > 0:
        return stake * (american_odds / 100.0)
    elif american_odds < 0:
        return stake * (100.0 / abs(american_odds))
    return 0.0


def kelly_criterion(model_prob: float, american_odds: float) -> float:
    """
    Computes Kelly fraction: optimal fraction of bankroll to bet.
    Returns fraction between 0 and 0.25 (capped at quarter-Kelly for safety).
    """
    if american_odds > 0:
        b = american_odds / 100.0
    else:
        b = 100.0 / abs(american_odds)
    p = model_prob
    q = 1.0 - p
    if b <= 0:
        return 0.0
    kelly = (p * b - q) / b
    return max(0.0, min(0.25, kelly))  # Quarter-Kelly cap


def _format_betting_pick_v2(pred: dict, stake: float = 10.0) -> dict:
    """Formats a prediction for the betting tab with dollar amounts and Kelly."""
    if not isinstance(pred, dict):
        return {}

    ml   = pred.get('ml_odds')
    pp   = safe_float(pred.get('pick_prob')) or 0.5
    out  = {
        'home':       pred.get('home', ''),
        'away':       pred.get('away', ''),
        'sport':      pred.get('sport', ''),
        'pick':       pred.get('pick', ''),
        'pick_prob':  pp,
        'confidence': pred.get('confidence', ''),
        'signals':    pred.get('signals', []),
        'date':       pred.get('date', ''),
        'ml_odds':    ml,
        'edge':       round(pred.get('edge', 0.0), 3),
        'ev':         round(pred.get('ev', 0.0), 3),
    }
    if ml is not None:
        out['bet_10_wins']   = round(moneyline_to_payout(ml, stake), 2)
        out['kelly_pct']     = round(kelly_criterion(pp, ml) * 100, 1)
    return out


# Legacy wrapper so existing code doesn't break
def _format_betting_pick(pred: dict) -> dict:
    return _format_betting_pick_v2(pred)

# ============================================================
# SECTION 24: MAIN ENTRY POINT
# Called by GitHub Actions with a mode argument.
# ============================================================

def main() -> None:
    """
    Main entry point. Accepts command line argument for mode:
      --mode train    : Full foundation training (slow, 1-2 hours)
      --mode update   : Daily update - predictions, auto-verify (fast, minutes)
      --mode cbs      : Generate CBS confidence pool picks
      --mode march    : Generate March Madness brackets
    """
    parser = argparse.ArgumentParser(description='EDGE Predict Brain')
    parser.add_argument(
        '--mode',
        choices=['train', 'train_models', 'update', 'cbs', 'march'],
        default='update',
        help='Operation mode'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Run Optuna hyperparameter optimization during training (~2 extra hours)'
    )
    args = parser.parse_args()

    log.info(f'EDGE Predict v{APP_VERSION} starting in mode: {args.mode}')
    log.info(f'BallDontLie key: {"set" if BALLDONTLIE_KEY else "NOT SET"}')
    log.info(f'Odds API key:    {"set" if ODDS_API_KEY else "NOT SET"}')
    log.info(f'OpenWeather key: {"set" if OPENWEATHER_KEY else "NOT SET"}')
    log.info(f'XGBoost:         {"available" if XGBOOST_AVAILABLE else "NOT AVAILABLE (install xgboost)"}')

    # Load persisted XGBoost models (if they exist) before anything else
    load_xgb_models()

    data = load_data()

    try:
        if args.mode == 'train':
            data = run_foundation_training(data)
            if getattr(args, 'tune', False):
                if TUNE_AVAILABLE:
                    log.info('[tune] Starting Optuna hyperparameter optimization...')
                    try:
                        run_tune(data)
                        save_data(data)
                        log.info('[tune] Optimization complete.')
                    except Exception as exc:
                        log.warning(f'[tune] Failed (non-fatal): {exc}')
                else:
                    log.warning('[tune] tune.py not found — skipping')

        elif args.mode == 'train_models':
            log.info('[brain] train_models — training advanced architectures')
            if data.get('training_status') != 'trained':
                log.error('[brain] Foundation not complete. Run train_foundation.yml first.')
                sys.exit(1)

            if TRANSFORMER_AVAILABLE:
                log.info('[transformer] Training...')
                try:
                    train_transformer(data)
                    save_data(data)
                    log.info('[transformer] Done.')
                except Exception as exc:
                    log.warning(f'[transformer] Failed (non-fatal): {exc}')
            else:
                log.warning('[transformer] transformer.py not found — skipping')

            if GRAPH_AVAILABLE:
                log.info('[graph] Training GNN...')
                try:
                    train_graph(data)
                    save_data(data)
                    log.info('[graph] Done.')
                except Exception as exc:
                    log.warning(f'[graph] Failed (non-fatal): {exc}')
            else:
                log.warning('[graph] graph.py not found — skipping')

            if STACK_AVAILABLE:
                log.info('[stack] Training meta-learner...')
                try:
                    train_stack(data)
                    save_data(data)
                    log.info('[stack] Done.')
                except Exception as exc:
                    log.warning(f'[stack] Failed (non-fatal): {exc}')
            else:
                log.warning('[stack] stack.py not found — skipping')

            if EXPLAIN_AVAILABLE:
                log.info('[explain] Initializing SHAP explainers...')
                try:
                    initialize_explain(data)
                    save_data(data)
                    log.info('[explain] Done.')
                except Exception as exc:
                    log.warning(f'[explain] Failed (non-fatal): {exc}')
            else:
                log.warning('[explain] explain.py not found — skipping')

            if SIMULATE_AVAILABLE:
                log.info('[simulate] Training drive/possession models...')
                try:
                    train_simulate(data)
                    save_data(data)
                    log.info('[simulate] Done.')
                except Exception as exc:
                    log.warning(f'[simulate] Failed (non-fatal): {exc}')
            else:
                log.warning('[simulate] simulate.py not found — skipping')

            log.info('[brain] train_models complete.')

        elif args.mode == 'update':
            data = run_daily_update(data)
            if AUDIT_AVAILABLE:
                try:
                    run_audit(data)
                except Exception as exc:
                    log.warning(f'[audit] Failed (non-fatal): {exc}')
            if BASELINE_AVAILABLE:
                try:
                    run_baseline(data)
                except Exception as exc:
                    log.warning(f'[baseline] Failed (non-fatal): {exc}')
            if TOURNAMENT_AVAILABLE:
                try:
                    run_tournament(data)
                except Exception as exc:
                    log.warning(f'[tournament] Failed (non-fatal): {exc}')
            if PORTFOLIO_AVAILABLE:
                try:
                    run_portfolio(data)
                except Exception as exc:
                    log.warning(f'[portfolio] Failed (non-fatal): {exc}')

        elif args.mode == 'cbs':
            if CBS_AVAILABLE:
                run_cbs(data)
            else:
                log.warning('cbs.py not found — falling back to built-in CBS optimizer')
                data = generate_cbs_picks(data)

        elif args.mode == 'march':
            if MARCH_AVAILABLE:
                run_march(data)
            else:
                log.warning('march.py not found — falling back to built-in bracket generator')
                data = generate_march_bracket(data, 'mens')
                data = generate_march_bracket(data, 'womens')

    except Exception as exc:
        log.error(f'Unhandled exception in mode {args.mode}: {exc}')
        log.error(traceback.format_exc())
        data['training_status'] = 'error'
        data['last_error']      = str(exc)

    finally:
        saved = save_data(data)
        if not saved:
            log.error('CRITICAL: Failed to save data after run')
            sys.exit(1)

    log.info('Run complete')


if __name__ == '__main__':
    main()
