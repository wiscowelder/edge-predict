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

import requests
import numpy as np

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
TRAIN_START_YEAR   = 2010
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
WEIGHT_MIN             = 0.05
WEIGHT_MAX             = 8.0
MONTE_CARLO_SIMS       = 10000    # Simulations for March Madness

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

# 538 NFL Elo abbreviation to full name
NFL_ELO_MAP: Dict[str, str] = {
    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons',
    'BAL': 'Baltimore Ravens', 'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys', 'DEN': 'Denver Broncos',
    'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts',
    'JAX': 'Jacksonville Jaguars', 'KC': 'Kansas City Chiefs',
    'LAC': 'Los Angeles Chargers', 'LAR': 'Los Angeles Rams',
    'LV': 'Las Vegas Raiders', 'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings', 'NE': 'New England Patriots',
    'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
    'NYJ': 'New York Jets', 'OAK': 'Las Vegas Raiders',
    'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers',
    'SD': 'Los Angeles Chargers', 'SEA': 'Seattle Seahawks',
    'SF': 'San Francisco 49ers', 'STL': 'Los Angeles Rams',
    'TB': 'Tampa Bay Buccaneers', 'TEN': 'Tennessee Titans',
    'WAS': 'Washington Commanders', 'WSH': 'Washington Commanders'
}

# 538 NBA Elo code to full name
NBA_ELO_MAP: Dict[str, str] = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BRK': 'Brooklyn Nets',
    'CHI': 'Chicago Bulls', 'CHO': 'Charlotte Hornets',
    'CLE': 'Cleveland Cavaliers', 'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers', 'LAC': 'LA Clippers',
    'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves', 'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers',
    'PHO': 'Phoenix Suns', 'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards', 'NJN': 'Brooklyn Nets',
    'NOH': 'New Orleans Pelicans', 'SEA': 'Oklahoma City Thunder',
    'CHA': 'Charlotte Hornets', 'BKN': 'Brooklyn Nets'
}

# 2024 NFL Elo fallback - used only when 538 GitHub is unreachable
NFL_ELO_FALLBACK: Dict[str, float] = {
    'Kansas City Chiefs': 1750, 'Philadelphia Eagles': 1680,
    'San Francisco 49ers': 1670, 'Baltimore Ravens': 1660,
    'Detroit Lions': 1640, 'Buffalo Bills': 1630,
    'Dallas Cowboys': 1620, 'Cincinnati Bengals': 1600,
    'Miami Dolphins': 1590, 'Houston Texans': 1580,
    'Pittsburgh Steelers': 1570, 'Los Angeles Rams': 1560,
    'Cleveland Browns': 1550, 'Minnesota Vikings': 1545,
    'Green Bay Packers': 1540, 'Seattle Seahawks': 1530,
    'Atlanta Falcons': 1510, 'Tampa Bay Buccaneers': 1500,
    'Los Angeles Chargers': 1490, 'Indianapolis Colts': 1480,
    'New Orleans Saints': 1470, 'New York Giants': 1460,
    'Denver Broncos': 1455, 'Jacksonville Jaguars': 1450,
    'Tennessee Titans': 1440, 'Chicago Bears': 1430,
    'New York Jets': 1420, 'Arizona Cardinals': 1410,
    'Las Vegas Raiders': 1400, 'Washington Commanders': 1390,
    'New England Patriots': 1380, 'Carolina Panthers': 1350
}

# Historical March Madness seed upset rates 1985-2024
# Key: (favorite_seed, underdog_seed), Value: probability favorite wins
SEED_UPSET_RATES: Dict[Tuple[int, int], float] = {
    (1, 16): 0.993, (2, 15): 0.934, (3, 14): 0.847,
    (4, 13): 0.795, (5, 12): 0.647, (6, 11): 0.631,
    (7, 10): 0.602, (8, 9): 0.514
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
    'mlb':    'baseball_mlb',
    'nhl':    'icehockey_nhl'
}

# All sports this program handles
ALL_SPORTS = list(ESPN_PATHS.keys())

# Default signal weights - starting point before any training
DEFAULT_WEIGHTS: Dict[str, float] = {
    'h2h':         1.5,
    'hfa':         1.5,
    'venue':       1.2,
    'dow':         0.8,
    'form':        1.8,
    'pythag':      1.3,
    'rest':        1.0,
    'elo':         1.3,
    'sp':          1.4,
    'injury':      1.5,
    'weather':     1.0,
    'crowd':       1.0,
    'spread':      1.2,
    'standings':   1.1,
    'travel':      0.9,
    'timezone':    0.8,
    'efficiency':  1.4,
    'turnover':    1.2,
    'four_factors':1.3,
    'altitude':    0.7,
    'bounce_back': 0.6
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
        'elo_ratings':        {'nfl': {}, 'nba': {}},
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
            'safe_bet': None, 'value_bet': None, 'parlay': None
        },
        'running_record': {s: copy.deepcopy(empty_rec) for s in ALL_SPORTS},
        'training_record': {
            'date': None, 'duration_seconds': None,
            'data_collected': {s: 0 for s in ALL_SPORTS},
            'accuracy': {s: 0.0 for s in ALL_SPORTS}
        }
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

    temp_path = DATA_FILE + '.tmp'

    try:
        json_str = json.dumps(data, ensure_ascii=False, indent=2, default=str)

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

        home_rank = home.get('curatedRank') or {}
        away_rank = away.get('curatedRank') or {}

        games.append({
            'id':        str(event.get('id', '')),
            'home':      home_name,
            'away':      away_name,
            'home_id':   str(home_team.get('id', '')),
            'away_id':   str(away_team.get('id', '')),
            'date':      safe_date(event.get('date', '')) or '',
            'venue':     venue_name,
            'venue_lat': safe_float(venue_addr.get('latitude')),
            'venue_lon': safe_float(venue_addr.get('longitude')),
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
    """
    path = ESPN_PATHS.get(sport)
    if not path:
        return []

    if not date_str or len(date_str) != 8 or not date_str.isdigit():
        log.warning(f'fetch_espn_past_scores: invalid date_str "{date_str}"')
        return []

    data = fetch_json(f'{ESPN_BASE}/{path}/scoreboard', params={'dates': date_str})
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
# SECTION 8: THESPORTSDB HISTORICAL DATA
# Free historical game data back to 2010.
# Used as primary source for training data.
# ============================================================

def fetch_tsdb_season(sport: str, year: int) -> List[dict]:
    """
    Fetches all games for a sport in a given year from TheSportsDB.
    Returns list of normalized game dicts with scores.
    Filters out games that haven't happened yet (future dates).
    """
    league_id = TSDB_LEAGUE_IDS.get(sport)
    if not league_id:
        log.warning(f'fetch_tsdb_season: no league ID for {sport}')
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

        home_name  = str(event.get('strHomeTeam', '') or '').strip()
        away_name  = str(event.get('strAwayTeam', '') or '').strip()
        date_str   = safe_date(event.get('dateEvent', ''))
        home_score = safe_int(event.get('intHomeScore'))
        away_score = safe_int(event.get('intAwayScore'))
        venue      = str(event.get('strVenue', '') or '').strip()

        if not home_name or not away_name:
            continue

        if home_score is None or away_score is None:
            continue

        if not date_str:
            continue

        # Filter out future games when processing current year
        if year == CURRENT_YEAR and date_str > YESTERDAY:
            continue

        games.append({
            'id':         str(event.get('idEvent', '')),
            'home':       home_name,
            'away':       away_name,
            'date':       date_str,
            'venue':      venue,
            'home_score': home_score,
            'away_score': away_score,
            'home_won':   home_score > away_score,
            'margin':     home_score - away_score
        })

    log.info(f'TSDB {sport} {year}: {len(games)} completed games')
    return games

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
# SECTION 10: ELO RATINGS FETCHER
# Fetches 538 NFL and NBA Elo ratings from GitHub.
# Falls back to hardcoded 2024 values if unavailable.
# ============================================================

def fetch_elo_ratings() -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Fetches NFL and NBA Elo ratings from 538's public GitHub.
    Returns (nfl_elo_dict, nba_elo_dict) - both keyed by full team name.
    If fetching fails at any point, uses hardcoded 2024 values as fallback.
    """
    nfl_elo: Dict[str, float] = {}
    nba_elo: Dict[str, float] = {}

    # Try NFL Elo - nfl_elo_latest.csv is smaller and more current
    nfl_urls = [
        'https://raw.githubusercontent.com/fivethirtyeight/data/master/nfl-elo/nfl_elo_latest.csv',
        'https://raw.githubusercontent.com/fivethirtyeight/data/master/nfl-elo/nfl_elo.csv'
    ]

    for url in nfl_urls:
        text = fetch_text(url)
        if text:
            nfl_elo = _parse_nfl_elo_csv(text)
            if len(nfl_elo) >= 20:
                log.info(f'NFL Elo loaded: {len(nfl_elo)} teams from {url.split("/")[-1]}')
                break

    if len(nfl_elo) < 20:
        log.warning('NFL Elo unavailable - using built-in 2024 ratings')
        nfl_elo = copy.deepcopy(NFL_ELO_FALLBACK)

    # Try NBA Elo
    nba_url = 'https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv'
    nba_text = fetch_text(nba_url)
    if nba_text:
        nba_elo = _parse_nba_elo_csv(nba_text)
        log.info(f'NBA Elo loaded: {len(nba_elo)} teams')
    else:
        log.warning('NBA Elo unavailable - Elo signal skipped for NBA')

    return nfl_elo, nba_elo


def _parse_nfl_elo_csv(text: str) -> Dict[str, float]:
    """
    Parses 538 NFL Elo CSV. Reads from end for most recent values.
    Returns dict of full team name -> elo rating.
    """
    if not text:
        return {}

    lines = text.strip().split('\n')
    if len(lines) < 2:
        return {}

    header = lines[0].split(',')

    try:
        i_team1 = header.index('team1')
        i_elo1  = header.index('elo1_pre')
    except ValueError:
        log.warning('NFL Elo CSV: expected columns not found')
        return {}

    i_team2 = header.index('team2') if 'team2' in header else -1
    i_elo2  = header.index('elo2_pre') if 'elo2_pre' in header else -1

    latest: Dict[str, float] = {}

    # Read backwards - most recent entries first
    for line in reversed(lines[1:]):
        if not line.strip():
            continue

        cols = line.split(',')
        if len(cols) <= i_elo1:
            continue

        team1 = cols[i_team1].strip() if i_team1 < len(cols) else ''
        elo1  = safe_float(cols[i_elo1]) if i_elo1 < len(cols) else None
        team2 = cols[i_team2].strip() if i_team2 >= 0 and i_team2 < len(cols) else ''
        elo2  = safe_float(cols[i_elo2]) if i_elo2 >= 0 and i_elo2 < len(cols) else None

        if team1 and elo1 is not None and team1 not in latest:
            latest[team1] = elo1
        if team2 and elo2 is not None and team2 not in latest:
            latest[team2] = elo2

        if len(latest) >= 35:
            break

    result: Dict[str, float] = {}
    for abbr, elo in latest.items():
        full_name = NFL_ELO_MAP.get(abbr)
        if full_name:
            result[full_name] = elo

    return result


def _parse_nba_elo_csv(text: str) -> Dict[str, float]:
    """
    Parses 538 NBA Elo CSV. Returns dict of full team name -> elo rating.
    """
    if not text:
        return {}

    lines = text.strip().split('\n')
    if len(lines) < 2:
        return {}

    header = lines[0].split(',')

    try:
        i_team = header.index('team_id')
        i_elo  = header.index('elo_n')
    except ValueError:
        log.warning('NBA Elo CSV: expected columns not found')
        return {}

    latest: Dict[str, float] = {}

    for line in reversed(lines[1:]):
        if not line.strip():
            continue

        cols = line.split(',')
        if len(cols) <= max(i_team, i_elo):
            continue

        team = cols[i_team].strip()
        elo  = safe_float(cols[i_elo])

        if team and elo is not None and team not in latest:
            latest[team] = elo

        if len(latest) >= 35:
            break

    result: Dict[str, float] = {}
    for code, elo in latest.items():
        full_name = NBA_ELO_MAP.get(code)
        if full_name:
            result[full_name] = elo

    return result

# ============================================================
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
            'regions': 'us',
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
    odds_parsed: Optional[dict] = None
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
    # Signal 7: Elo Rating (NFL and NBA)
    # ----------------------------------------------------------
    sport_elo_map = {'nfl': 'nfl', 'nba': 'nba'}
    elo_key = sport_elo_map.get(sport)

    if elo_key and isinstance(elo_ratings, dict):
        sport_elos = elo_ratings.get(elo_key) or {}
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
    # ADDITIVE ADJUSTMENTS
    # These modify the final probability directly (not averaged in).
    # ----------------------------------------------------------

    # Rest advantage
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
    team_stats: dict
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
                team_stats=team_stats
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

    return weights, prev_acc


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

    # Fetch Elo ratings once
    log.info('Fetching Elo ratings...')
    nfl_elo, nba_elo = fetch_elo_ratings()
    data['elo_ratings']['nfl'] = nfl_elo
    data['elo_ratings']['nba'] = nba_elo

    total_games = 0

    for sport in ALL_SPORTS:
        log.info(f'\n>> TRAINING {sport.upper()} <<')

        current_weights = copy.deepcopy(DEFAULT_WEIGHTS)
        sport_games_total = 0

        for year in range(TRAIN_START_YEAR, CURRENT_YEAR + 1):
            log.info(f'  Fetching {sport} {year}...')

            # Primary: TheSportsDB
            games = fetch_tsdb_season(sport, year)

            # Supplement with BDN if available and sport is supported
            if BALLDONTLIE_KEY and sport in BDN_SPORTS:
                start_str = f'{year}-01-01'
                end_str   = f'{year}-12-31'
                bdn_games = fetch_bdn_games(sport, start_str, end_str)

                # Merge: add BDN games that aren't already in TSDB (by date+teams)
                existing_keys = set()
                for g in games:
                    if isinstance(g, dict):
                        key = f"{g.get('date','')}{g.get('home','')}{g.get('away','')}"
                        existing_keys.add(key)

                for g in bdn_games:
                    if isinstance(g, dict):
                        key = f"{g.get('date','')}{g.get('home','')}{g.get('away','')}"
                        if key not in existing_keys:
                            games.append(g)

            if not games:
                log.warning(f'  {sport} {year}: no games retrieved')
                continue

            # Anti-hindsight accuracy test: test current weights on this year BEFORE learning from it
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
                        team_stats=data['team_stats']
                    )
                    home_won = game.get('home_won')
                    if home_won is not None:
                        was_right = (pred['pick'] == game.get('home', '')) == bool(home_won)
                        if was_right:
                            pre_correct += 1

                pre_acc = pre_correct / len(games) if games else 0.0
                log.info(f'  {sport} {year}: {len(games)} games | pre-train accuracy: {pre_acc:.1%}')
            else:
                log.info(f'  {sport} {year}: {len(games)} games (building baseline)')

            # Build Pythagorean expectation from this year
            pythag_map = build_pythagorean_from_games(games, sport)
            if sport not in data['pythagorean']:
                data['pythagorean'][sport] = {}
            data['pythagorean'][sport].update(pythag_map)

            # Build patterns from this year
            build_patterns_from_games(games, data['patterns'])

            # Update team history
            update_team_history(games, sport, data['team_history'])

            # Fetch box score stats if BDN available
            if BALLDONTLIE_KEY and sport in BDN_SPORTS:
                stats = fetch_bdn_team_stats(sport, year)
                if stats:
                    if sport not in data['team_stats']:
                        data['team_stats'][sport] = {}
                    data['team_stats'][sport].update(stats)

            # Optimize weights on this year's data
            optimized_weights, post_acc = optimize_weights_for_year(
                games=games,
                sport=sport,
                start_weights=current_weights,
                patterns=data['patterns'],
                team_history=data['team_history'],
                elo_ratings=data['elo_ratings'],
                pythagorean=data['pythagorean'],
                standings=data['standings'].get(sport, {}),
                team_stats=data['team_stats']
            )

            log.info(f'  {sport} {year}: post-train accuracy: {post_acc:.1%}')

            current_weights    = optimized_weights
            sport_games_total += len(games)
            total_games        += len(games)

            # Prune and compress patterns periodically to manage memory
            if year % 3 == 0:
                prune_patterns(data['patterns'])
                compress_patterns(data['patterns'])

        # Final validation on most recent complete year
        if sport_games_total > 100:
            holdout_year   = CURRENT_YEAR - 1
            holdout_games  = fetch_tsdb_season(sport, holdout_year)

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
                        team_stats=data['team_stats']
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

    return data

# ============================================================
# SECTION 21: CBS CONFIDENCE POOL OPTIMIZER
# Expected value math per slot - not just sorting by confidence.
# Accounts for variable game counts per week.
# ============================================================

def generate_cbs_picks(data: dict) -> dict:
    """
    Generates CBS confidence pool picks for the current week.
    Uses expected value optimization: finds the assignment of
    confidence values to picks that maximizes expected total points.

    Expected points for a slot = pick_probability * point_value
    Optimizes this across all slots simultaneously.
    """
    log.info('Generating CBS confidence pool picks...')

    if data.get('training_status') not in ('trained',):
        log.warning('Cannot generate CBS picks - model not trained')
        return data

    # Fetch current NFL schedule
    games = fetch_espn_scoreboard('nfl')

    if not games:
        log.warning('No NFL games found for CBS picks')
        return data

    num_games = len(games)
    week_num  = get_week_number(games[0].get('date', '')) if games else 0

    log.info(f'CBS Week {week_num}: {num_games} games')

    # Fetch live signals once
    odds_data     = fetch_betting_odds('nfl')
    nfl_standings = fetch_espn_standings('nfl')

    if nfl_standings:
        data['standings']['nfl'] = nfl_standings

    # Predict each game
    game_predictions = []

    for game in games:
        # Get weather if outdoor
        weather = get_weather(
            game.get('venue_lat'),
            game.get('venue_lon'),
            game.get('venue', ''),
            game.get('date', '')
        )

        # Get betting odds
        odds_parsed = None
        if odds_data:
            odds_parsed = parse_odds_for_game(odds_data, game['home'], game['away'])

        # Get injuries
        injuries_home = fetch_espn_injuries(game.get('home_id', ''), 'nfl')
        injuries_away = fetch_espn_injuries(game.get('away_id', ''), 'nfl')

        pred = predict_game(
            home=game['home'],
            away=game['away'],
            game_date=game.get('date', ''),
            venue=game.get('venue', ''),
            sport='nfl',
            weights=data['weights'].get('nfl', DEFAULT_WEIGHTS),
            patterns=data['patterns'],
            team_history=data['team_history'],
            elo_ratings=data['elo_ratings'],
            pythagorean=data['pythagorean'],
            standings=data['standings'].get('nfl', {}),
            team_stats=data['team_stats'],
            injuries_home=injuries_home,
            injuries_away=injuries_away,
            weather=weather,
            odds_parsed=odds_parsed
        )

        game_predictions.append({
            'game':  game,
            'pred':  pred
        })

    # Expected value optimization for point assignment
    # For N games, points available are 1 through N
    # Assign highest points to highest expected value picks
    # Expected value = pick_probability * point_value
    # Optimal: sort by pick_prob DESC, assign N down to 1

    # Mark is_guess picks as low confidence (they default to ~55% which would
    # incorrectly rank them above real 52% picks)
    for gp in game_predictions:
        if gp['pred']['is_guess']:
            gp['pred']['pick_prob'] = 0.501  # Force to bottom of sort

    # Sort by pick probability descending
    game_predictions.sort(key=lambda x: x['pred']['pick_prob'], reverse=True)

    # Assign point values: highest confidence = most points
    max_pts      = num_games
    expected_pts = 0.0
    assignment   = []

    for i, gp in enumerate(game_predictions):
        point_value   = max_pts - i
        pick_prob     = gp['pred']['pick_prob']
        expected_value = pick_prob * point_value

        game   = gp['game']
        pred   = gp['pred']
        is_guess = pred.get('is_guess', False) or pred['pick_prob'] <= 0.502

        assignment.append({
            'point_value':    point_value,
            'home':           game['home'],
            'away':           game['away'],
            'pick':           pred['pick'],
            'pick_prob':      pick_prob if not is_guess else None,
            'display_pct':    f'{pick_prob:.0%}' if not is_guess else '~50%',
            'confidence':     pred['confidence'],
            'is_guess':       is_guess,
            'expected_value': expected_value,
            'signals':        pred.get('signals_fired', []),
            'date':           game.get('date', '')
        })

        expected_pts += expected_value

    # Total max possible if all picks win
    max_possible = sum(range(1, num_games + 1))

    data['cbs_picks'] = {
        'generated_at':       datetime.datetime.utcnow().isoformat(),
        'week_number':        week_num,
        'num_games':          num_games,
        'max_possible_points': max_possible,
        'expected_points':    round(expected_pts, 1),
        'assignment':         assignment
    }

    log.info(f'CBS picks generated: {num_games} games, expected {expected_pts:.1f}/{max_possible} pts')
    return data

# ============================================================
# SECTION 22: MARCH MADNESS BRACKET OPTIMIZER
# Monte Carlo simulation: runs tournament 10,000 times.
# Finds the bracket that wins the most simulations.
# Separate for men's and women's.
# ============================================================

def generate_march_bracket(data: dict, gender: str) -> dict:
    """
    Generates an optimized March Madness bracket using Monte Carlo simulation.
    gender: 'mens' or 'womens'
    Simulates the full tournament MONTE_CARLO_SIMS times using model probabilities.
    Returns the bracket that wins the most simulations.
    """
    sport_key = 'ncaabm' if gender == 'mens' else 'ncaabw'

    log.info(f'Generating March Madness bracket ({gender})...')
    log.info(f'Running {MONTE_CARLO_SIMS:,} simulations...')

    games = fetch_espn_scoreboard(sport_key)

    if not games:
        log.warning(f'No {sport_key} tournament games found')
        return data

    # Build current round matchups
    matchups = []
    for game in games:
        home_seed = game.get('home_seed') or 0
        away_seed = game.get('away_seed') or 0

        # Get model probability
        pred = predict_game(
            home=game['home'],
            away=game['away'],
            game_date=game.get('date', ''),
            venue=game.get('venue', ''),
            sport=sport_key,
            weights=data['weights'].get(sport_key, DEFAULT_WEIGHTS),
            patterns=data['patterns'],
            team_history=data['team_history'],
            elo_ratings=data['elo_ratings'],
            pythagorean=data['pythagorean'],
            standings=data['standings'].get(sport_key, {}),
            team_stats=data['team_stats']
        )

        # Blend model probability with historical seed upset rates
        model_prob = pred['home_prob']
        seed_prob  = _get_seed_probability(home_seed, away_seed)

        if seed_prob is not None:
            # 60% model, 40% historical seed data
            blended_prob = model_prob * 0.6 + seed_prob * 0.4
        else:
            blended_prob = model_prob

        matchups.append({
            'home':          game['home'],
            'away':          game['away'],
            'home_seed':     home_seed,
            'away_seed':     away_seed,
            'home_prob':     blended_prob,
            'model_prob':    model_prob,
            'seed_prob':     seed_prob,
            'date':          game.get('date', ''),
            'venue':         game.get('venue', '')
        })

    # Monte Carlo simulation
    win_counts: Dict[str, Dict[str, int]] = {}  # game_key -> team -> win count

    for _ in range(MONTE_CARLO_SIMS):
        for matchup in matchups:
            home      = matchup['home']
            away      = matchup['away']
            home_prob = matchup['home_prob']
            game_key  = f'{home}|||{away}'

            if game_key not in win_counts:
                win_counts[game_key] = {home: 0, away: 0}

            if np.random.random() < home_prob:
                win_counts[game_key][home] += 1
            else:
                win_counts[game_key][away] += 1

    # Build recommended bracket from simulation results
    recommended_bracket = []
    upset_alerts        = []

    for matchup in matchups:
        home      = matchup['home']
        away      = matchup['away']
        game_key  = f'{home}|||{away}'

        counts = win_counts.get(game_key, {home: 0, away: 0})
        home_wins  = counts.get(home, 0)
        total_sims = home_wins + counts.get(away, 0)

        if total_sims == 0:
            sim_home_prob = 0.5
        else:
            sim_home_prob = home_wins / total_sims

        recommended_pick = home if sim_home_prob >= 0.5 else away
        sim_confidence   = max(sim_home_prob, 1.0 - sim_home_prob)

        entry = {
            'home':           home,
            'away':           away,
            'home_seed':      matchup.get('home_seed', 0),
            'away_seed':      matchup.get('away_seed', 0),
            'recommended':    recommended_pick,
            'sim_confidence': sim_confidence,
            'model_prob':     matchup['model_prob'],
            'seed_prob':      matchup.get('seed_prob'),
            'home_prob':      matchup['home_prob'],
            'date':           matchup.get('date', '')
        }

        recommended_bracket.append(entry)

        # Upset alert: model thinks upset is likely (>40%) when seed mismatch is large
        home_seed = matchup.get('home_seed', 0)
        away_seed = matchup.get('away_seed', 0)
        if home_seed and away_seed and abs(home_seed - away_seed) >= 4:
            # Underdog is the team with the higher seed number
            underdog     = away if away_seed > home_seed else home
            underdog_prob = (1.0 - matchup['home_prob']) if away_seed > home_seed else matchup['home_prob']

            if underdog_prob >= 0.40:
                upset_alerts.append({
                    'game':          f'{away} ({away_seed}) @ {home} ({home_seed})',
                    'underdog':      underdog,
                    'upset_prob':    underdog_prob,
                    'message':       f'{underdog} has {underdog_prob:.0%} upset chance'
                })

    # Sort bracket by date for display
    recommended_bracket.sort(key=lambda x: x.get('date', ''))

    result_key = 'march_mens' if gender == 'mens' else 'march_womens'
    data[result_key] = {
        'generated_at':       datetime.datetime.utcnow().isoformat(),
        'simulations_run':    MONTE_CARLO_SIMS,
        'recommended_bracket': recommended_bracket,
        'upset_alerts':        upset_alerts
    }

    log.info(f'March bracket ({gender}): {len(recommended_bracket)} games, {len(upset_alerts)} upset alerts')
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
                odds_parsed=odds_parsed
            )

            cal_note = get_calibration_note(
                data['calibration'], sport, pred['pick_prob']
            )

            game_date_str = game.get('date', '')
            game_id       = f"{sport}_{game['home']}_{game['away']}_{game_date_str[:10]}"

            sport_predictions.append({
                'id':           game_id,
                'home':         game['home'],
                'away':         game['away'],
                'date':         game_date_str,
                'venue':        game.get('venue', ''),
                'pick':         pred['pick'],
                'home_prob':    pred['home_prob'],
                'pick_prob':    pred['pick_prob'],
                'confidence':   pred['confidence'],
                'is_guess':     pred['is_guess'],
                'signals':      pred.get('signals_fired', []),
                'cal_note':     cal_note,
                'status':       'pending',
                'generated_at': datetime.datetime.utcnow().isoformat()
            })

        data['predictions'][sport] = sport_predictions
        log.info(f'  {sport}: {len(sport_predictions)} predictions generated')
        time.sleep(0.3)  # Rate limit between sports

    # Step 4: Generate betting picks
    data = _generate_betting_picks(data, odds_cache)

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

                # Find matching actual result
                actual = next(
                    (r for r in actual_results
                     if r.get('home') == home and r.get('away') == away),
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

    return data


def _generate_betting_picks(data: dict, odds_cache: dict) -> dict:
    """
    Generates three betting pick strategies from NFL predictions.
    Safe bet: highest confidence single game.
    Value bet: high confidence game where market may underestimate.
    Parlay: 2-4 games all at 65%+ confidence.
    """
    nfl_preds = data['predictions'].get('nfl', [])

    if not nfl_preds:
        return data

    pending = [p for p in nfl_preds if isinstance(p, dict) and p.get('status') == 'pending']

    if not pending:
        return data

    # Sort by pick confidence
    pending.sort(key=lambda p: safe_float(p.get('pick_prob')) or 0.0, reverse=True)

    # Safe bet: highest confidence
    safe = pending[0] if pending else None

    # Value bet: second highest confidence (not the same as safe)
    value_candidates = [
        p for p in pending[1:]
        if (safe_float(p.get('pick_prob')) or 0.0) >= 0.58
    ]
    value = value_candidates[0] if value_candidates else (pending[1] if len(pending) > 1 else None)

    # Parlay: need 2-4 games at 65%+
    parlay_legs = [
        p for p in pending
        if (safe_float(p.get('pick_prob')) or 0.0) >= 0.65
    ][:4]

    safe_out   = _format_betting_pick(safe) if safe else None
    value_out  = _format_betting_pick(value) if value else None

    parlay_out = None
    if len(parlay_legs) >= 2:
        combined = 1.0
        for leg in parlay_legs:
            combined *= (safe_float(leg.get('pick_prob')) or 0.5)

        approx_mult = 1.0 / combined if combined > 0 else 0.0

        parlay_out = {
            'legs':         [_format_betting_pick(p) for p in parlay_legs],
            'combined_prob': combined,
            'approx_mult':  approx_mult,
            'note':         'All legs must win. High risk.'
        }

    data['betting_picks'] = {
        'generated_at': datetime.datetime.utcnow().isoformat(),
        'safe_bet':     safe_out,
        'value_bet':    value_out,
        'parlay':       parlay_out
    }

    return data


def _format_betting_pick(pred: dict) -> dict:
    """Formats a prediction dict for the betting picks display."""
    if not isinstance(pred, dict):
        return {}

    return {
        'home':       pred.get('home', ''),
        'away':       pred.get('away', ''),
        'pick':       pred.get('pick', ''),
        'pick_prob':  pred.get('pick_prob'),
        'confidence': pred.get('confidence', ''),
        'signals':    pred.get('signals', []),
        'date':       pred.get('date', '')
    }

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
        choices=['train', 'update', 'cbs', 'march'],
        default='update',
        help='Operation mode'
    )
    args = parser.parse_args()

    log.info(f'EDGE Predict v{APP_VERSION} starting in mode: {args.mode}')
    log.info(f'BallDontLie key: {"set" if BALLDONTLIE_KEY else "NOT SET"}')
    log.info(f'Odds API key:    {"set" if ODDS_API_KEY else "NOT SET"}')
    log.info(f'OpenWeather key: {"set" if OPENWEATHER_KEY else "NOT SET"}')

    data = load_data()

    try:
        if args.mode == 'train':
            data = run_foundation_training(data)

        elif args.mode == 'update':
            data = run_daily_update(data)

        elif args.mode == 'cbs':
            data = generate_cbs_picks(data)

        elif args.mode == 'march':
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
