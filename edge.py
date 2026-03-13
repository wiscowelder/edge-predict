#!/usr/bin/env python3
"""
EDGE Predict — edge.py
Market and sentiment intelligence layer.
Collects line movement signals and text sentiment signals.
Runs during update workflow, writes to data['edge'], never raises.

Section 1  — Imports and constants
Section 2  — data.json keys managed by edge.py
Section 3  — Odds API client
Section 4  — Sharp money detection
Section 5  — Line movement classifier
Section 6  — Reddit scraper
Section 7  — Nitter/Twitter scraper
Section 8  — ESPN news feed parser
Section 9  — Sentiment scoring engine
Section 10 — Injury report language classifier
Section 11 — Press conference / coach quote analyzer
Section 12 — Sentiment aggregation
Section 13 — Signal export
Section 14 — Update entry point
Section 15 — Rate limiting, retry logic, error handling
"""

# ============================================================
# SECTION 1: IMPORTS AND CONSTANTS
# ============================================================
import os
import re
import sys
import json
import math
import time
import copy
import logging
import datetime
import traceback
import xml.etree.ElementTree as ET
from typing import Optional, Dict, List, Tuple, Any

import requests

log = logging.getLogger(__name__)

ODDS_API_KEY  = os.environ.get('ODDS_API_KEY', '')
ODDS_BASE     = 'https://api.the-odds-api.com/v4/sports'
ESPN_BASE     = 'https://site.api.espn.com/apis/site/v2/sports'
ESPN_RSS_BASE = 'https://www.espn.com/espn/rss'
REDDIT_BASE   = 'https://old.reddit.com'

ESPN_PATHS = {
    'nfl':    'football/nfl',
    'nba':    'basketball/nba',
    'mlb':    'baseball/mlb',
    'nhl':    'hockey/nhl',
    'ncaaf':  'football/college-football',
    'ncaabm': 'basketball/mens-college-basketball',
    'ncaabw': 'basketball/womens-college-basketball',
}
ALL_SPORTS = list(ESPN_PATHS.keys())

ODDS_SPORT_KEYS = {
    'nfl':    'americanfootball_nfl',
    'ncaaf':  'americanfootball_ncaaf',
    'nba':    'basketball_nba',
    'ncaabm': 'basketball_ncaab',
    'ncaabw': 'basketball_ncaawb',
    'mlb':    'baseball_mlb',
    'nhl':    'icehockey_nhl',
}

REDDIT_SUBS = {
    'nfl':    'nfl',
    'nba':    'nba',
    'mlb':    'baseball',
    'nhl':    'hockey',
    'ncaaf':  'CFB',
    'ncaabm': 'CollegeBasketball',
    'ncaabw': 'CollegeBasketball',
}

ESPN_INJURY_PATHS = {
    'nfl':    'football/nfl/injuries',
    'nba':    'basketball/nba/injuries',
    'mlb':    'baseball/mlb/injuries',
    'nhl':    'hockey/nhl/injuries',
    'ncaabm': 'basketball/mens-college-basketball/injuries',
    'ncaabw': 'basketball/womens-college-basketball/injuries',
}

# Nitter mirrors — tried in order, first success used
NITTER_MIRRORS = [
    'https://nitter.net',
    'https://nitter.it',
    'https://nitter.privacydev.net',
]

# Rate limits: maximum requests per update run per source
RATE_LIMITS = {
    'odds_api':   8,
    'reddit':     60,
    'nitter':     30,
    'espn_rss':   20,
    'espn_inj':   7,
    'polymarket': 5,
}

# Maximum probability adjustment from prediction markets
PREDICTION_MARKET_MAX_IMPACT = 0.05

# Minimum resolved samples required before adaptive multiplier adjusts away from 1.0
SIGNAL_MIN_SAMPLE = 20

# Sentiment probability adjustment cap per sport
SENTIMENT_MAX_IMPACT = {
    'nfl': 0.07, 'nba': 0.05, 'mlb': 0.04, 'nhl': 0.05,
    'ncaaf': 0.06, 'ncaabm': 0.07, 'ncaabw': 0.07,
}

# Steam move detection thresholds
STEAM_MOVE_SPREAD_PTS = 2.0     # Points in spread that counts as steam
STEAM_MOVE_ML_CENTS   = 20      # Cents in moneyline
STEAM_MOVE_WINDOW_HRS = 2       # Time window for steam detection

# Sharp probability deltas per signal type
SHARP_DELTA = {
    'steam_move':    0.10,
    'rlm':           0.08,
    'key_number':    0.04,
    'consensus':     0.00,  # Informational only
    'freeze':        0.02,
}

# Teams with chronic public betting bias
PUBLIC_TEAMS = {
    'nfl':    ['dallas cowboys', 'new england patriots', 'green bay packers', 'kansas city chiefs'],
    'nba':    ['los angeles lakers', 'golden state warriors', 'boston celtics'],
    'mlb':    ['new york yankees', 'los angeles dodgers', 'chicago cubs'],
    'nhl':    ['toronto maple leafs', 'montreal canadiens'],
    'ncaabm': ['duke blue devils', 'kentucky wildcats', 'kansas jayhawks', 'north carolina tar heels'],
    'ncaabw': [],
    'ncaaf':  ['alabama crimson tide', 'ohio state buckeyes', 'georgia bulldogs'],
}

# Key numbers in NFL spread betting
NFL_KEY_NUMBERS = {3, 7, 10, 14, 6, 4, 17}

# Reporter accounts to monitor by sport
REPORTER_ACCOUNTS = {
    'nfl':    ['adamschefter', 'rapsheet', 'TomPelissero'],
    'nba':    ['wojespn', 'ShamsCharania'],
    'mlb':    ['JeffPassan', 'Feinsand'],
    'nhl':    ['PierreVLeBrun', 'TSNBobMcKenzie'],
}

# ============================================================
# SECTION 2: DATA.JSON KEYS MANAGED BY EDGE.PY
# ============================================================

def get_edge_default() -> dict:
    """Returns default structure for data['edge']."""
    return {
        'version': '1.0',
        'last_updated': None,
        'line_history': {},
        'sharp_signals': {},
        'sentiment': {},
        'prediction_markets': {},
        'signal_accuracy': {
            '_processed_ids': [],
        },
        'reporter_accounts': copy.deepcopy(REPORTER_ACCOUNTS),
        'public_teams': copy.deepcopy(PUBLIC_TEAMS),
        'rate_limit_state': {},
        'cache': {},
        'error_log': [],
    }


def ensure_edge_keys(data: dict) -> None:
    """Ensures data['edge'] exists with all required keys."""
    if 'edge' not in data or not isinstance(data.get('edge'), dict):
        data['edge'] = get_edge_default()
        return
    default = get_edge_default()
    for k, v in default.items():
        if k not in data['edge']:
            data['edge'][k] = copy.deepcopy(v)
    # Ensure signal_accuracy always has _processed_ids list
    if not isinstance(data['edge']['signal_accuracy'].get('_processed_ids'), list):
        data['edge']['signal_accuracy']['_processed_ids'] = []


# ============================================================
# SECTION 3: ODDS API CLIENT
# ============================================================

def _fetch_odds_lines(sport: str, req_counter: dict) -> List[dict]:
    """
    Fetches current odds lines from The Odds API for a sport.
    Returns list of game odds dicts or empty list if key missing / rate exceeded.
    """
    if not ODDS_API_KEY:
        return []

    if req_counter.get('odds_api', 0) >= RATE_LIMITS['odds_api']:
        return []

    odds_key = ODDS_SPORT_KEYS.get(sport)
    if not odds_key:
        return []

    try:
        url = f'{ODDS_BASE}/{odds_key}/odds'
        params = {
            'apiKey':    ODDS_API_KEY,
            'regions':   'us',
            'markets':   'h2h,spreads',
            'oddsFormat': 'american',
            'dateFormat': 'iso',
        }
        resp = _http_get(url, params=params, timeout=20)
        req_counter['odds_api'] = req_counter.get('odds_api', 0) + 1

        if resp is None:
            return []

        return resp if isinstance(resp, list) else []
    except Exception as exc:
        log.debug(f'[edge] Odds API error for {sport}: {exc}')
        return []


def _parse_odds_lines(raw_odds: List[dict]) -> Dict[str, dict]:
    """
    Parses raw Odds API response into a dict keyed by ESPN-compatible game string.
    Returns dict: {home_team|away_team: {spread, moneyline, total, bookmakers}}
    """
    parsed = {}

    for game in raw_odds:
        if not isinstance(game, dict):
            continue

        espn_id    = game.get('id', '')
        home_team  = (game.get('home_team') or '').lower().strip()
        away_team  = (game.get('away_team') or '').lower().strip()
        game_time  = game.get('commence_time', '')

        if not home_team or not away_team:
            continue

        spread_values = []
        ml_values     = []
        total_values  = []

        for bookmaker in (game.get('bookmakers') or []):
            for market in (bookmaker.get('markets') or []):
                mkey = market.get('key', '')
                outcomes = market.get('outcomes') or []

                if mkey == 'spreads':
                    for o in outcomes:
                        if (o.get('name') or '').lower() == home_team:
                            spread_values.append(float(o.get('point', 0)))

                elif mkey == 'h2h':
                    home_price = next(
                        (float(o.get('price', 0)) for o in outcomes
                         if (o.get('name') or '').lower() == home_team), None
                    )
                    if home_price is not None:
                        ml_values.append(home_price)

                elif mkey == 'totals':
                    for o in outcomes:
                        total_values.append(float(o.get('point', 0) or 0))

        avg_spread = sum(spread_values) / len(spread_values) if spread_values else None
        avg_ml     = sum(ml_values)     / len(ml_values)     if ml_values     else None
        avg_total  = sum(total_values)  / len(total_values)  if total_values  else None

        key = f'{home_team}|{away_team}'
        parsed[key] = {
            'espn_id':      espn_id,
            'home_team':    home_team,
            'away_team':    away_team,
            'game_time':    game_time,
            'spread':       avg_spread,
            'moneyline':    avg_ml,
            'total':        avg_total,
            'n_books':      len(game.get('bookmakers') or []),
            'fetched_at':   datetime.datetime.utcnow().isoformat(),
        }

    return parsed


# ============================================================
# SECTION 4: SHARP MONEY DETECTION ALGORITHM
# ============================================================

def detect_sharp_money(game_id: str, current_line: dict,
                        line_history: dict, sport: str,
                        public_teams: dict) -> dict:
    """
    Analyzes line history for a game to detect sharp money signals.
    Returns a sharp_signal dict with type, confidence, prob_delta.
    """
    result = {
        'game_id':        game_id,
        'sport':          sport,
        'signal_type':    None,
        'prob_delta':     0.0,
        'confidence':     0.0,
        'toward_home':    None,
        'notes':          [],
    }

    history = line_history.get(str(game_id), {})
    snapshots = history.get('snapshots', [])

    if not snapshots or not current_line:
        return result

    opening = history.get('opening_spread')
    current_spread = current_line.get('spread')
    current_ml     = current_line.get('moneyline')

    if opening is None or current_spread is None:
        return result

    total_movement = current_spread - opening
    result['total_movement'] = total_movement
    toward_home = total_movement < 0  # Spread moved in home team's favor
    result['toward_home'] = toward_home

    home_team = (history.get('home_team') or '').lower()
    away_team = (history.get('away_team') or '').lower()

    # --- Signal 1: Steam Move ---
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        curr = snapshots[i]

        prev_spread = prev.get('spread')
        curr_spread = curr.get('spread')
        if prev_spread is None or curr_spread is None:
            continue

        move_magnitude = abs(curr_spread - prev_spread)

        try:
            prev_ts = datetime.datetime.fromisoformat(prev.get('timestamp', ''))
            curr_ts = datetime.datetime.fromisoformat(curr.get('timestamp', ''))
            hours   = (curr_ts - prev_ts).total_seconds() / 3600
        except (ValueError, TypeError):
            hours = 24

        if move_magnitude >= STEAM_MOVE_SPREAD_PTS and hours <= STEAM_MOVE_WINDOW_HRS:
            result['signal_type'] = 'steam_move'
            result['confidence']  = min(0.90, 0.60 + move_magnitude * 0.10)
            delta = SHARP_DELTA['steam_move']
            result['prob_delta'] = delta if toward_home else -delta
            result['notes'].append(
                f'Steam move: {move_magnitude:.1f} pts in {hours:.1f}hrs'
            )
            return result

    # --- Signal 2: Reverse Line Movement ---
    pub_teams_sport = public_teams.get(sport, [])
    home_is_public  = home_team in pub_teams_sport
    away_is_public  = away_team in pub_teams_sport

    if home_is_public and total_movement > 0.5:
        # Line moved against public home team = sharp on away
        result['signal_type'] = 'rlm'
        result['confidence']  = 0.68
        result['prob_delta']  = -SHARP_DELTA['rlm']  # Against home
        result['toward_home'] = False
        result['notes'].append(
            f'RLM: public on {home_team} but line moved {total_movement:+.1f}'
        )
        return result

    if away_is_public and total_movement < -0.5:
        # Line moved against public away team = sharp on home
        result['signal_type'] = 'rlm'
        result['confidence']  = 0.68
        result['prob_delta']  = SHARP_DELTA['rlm']
        result['toward_home'] = True
        result['notes'].append(
            f'RLM: public on {away_team} but line moved {total_movement:+.1f}'
        )
        return result

    # --- Signal 4: Key Number Cross (NFL only) ---
    if sport == 'nfl' and opening is not None and current_spread is not None:
        for kn in NFL_KEY_NUMBERS:
            crossed = (
                (opening < kn and current_spread >= kn) or
                (opening > kn and current_spread <= kn) or
                (opening < -kn and current_spread >= -kn) or
                (opening > -kn and current_spread <= -kn)
            )
            if crossed:
                result['signal_type'] = 'key_number'
                result['confidence']  = 0.55
                delta = SHARP_DELTA['key_number']
                result['prob_delta']  = delta if toward_home else -delta
                result['notes'].append(f'Key number {kn} crossed')
                return result

    # --- Signal 5: Freeze (line locked despite public team in game) ---
    # Public money normally moves lines toward the public team.
    # If the line is frozen across the last 3+ snapshots, sharps are
    # counteracting public action — signal is AGAINST the public team.
    pub_teams_sport_freeze = public_teams.get(sport, [])
    home_is_public_freeze  = home_team in pub_teams_sport_freeze
    away_is_public_freeze  = away_team in pub_teams_sport_freeze

    if (home_is_public_freeze or away_is_public_freeze) and len(snapshots) >= 3:
        recent_snapshots = snapshots[-3:]
        recent_spreads   = [s.get('spread') for s in recent_snapshots
                            if s.get('spread') is not None]
        if len(recent_spreads) >= 3:
            freeze_range = max(recent_spreads) - min(recent_spreads)
            if freeze_range < 0.25:
                result['signal_type'] = 'freeze'
                result['confidence']  = 0.50
                if home_is_public_freeze:
                    # Sharps fading the public home team
                    result['toward_home'] = False
                    result['prob_delta']  = -SHARP_DELTA['freeze']
                    result['notes'].append(
                        f'Freeze: line locked at {recent_spreads[-1]} '
                        f'despite public home team {home_team}'
                    )
                else:
                    # Sharps fading the public away team
                    result['toward_home'] = True
                    result['prob_delta']  = SHARP_DELTA['freeze']
                    result['notes'].append(
                        f'Freeze: line locked at {recent_spreads[-1]} '
                        f'despite public away team {away_team}'
                    )
                return result

    # --- Signal 6: Consensus Line ---
    n_books = current_line.get('n_books', 0)
    if n_books >= 7 and abs(total_movement) < 0.5:
        result['signal_type'] = 'consensus'
        result['confidence']  = 0.75
        result['prob_delta']  = 0.0  # Informational, no prob adjustment
        result['notes'].append(f'Consensus line: {n_books} books agree')

    return result


# ============================================================
# SECTION 5: LINE MOVEMENT CLASSIFIER
# ============================================================

def update_line_history(data: dict, game_id: str, sport: str,
                         home_team: str, away_team: str,
                         current_line: dict, game_date: str) -> None:
    """Updates line_history in data['edge'] with current snapshot."""
    ensure_edge_keys(data)

    line_history = data['edge']['line_history']
    gid = str(game_id)

    if gid not in line_history:
        line_history[gid] = {
            'sport':         sport,
            'home_team':     home_team.lower(),
            'away_team':     away_team.lower(),
            'game_date':     game_date,
            'snapshots':     [],
            'opening_spread': current_line.get('spread'),
            'current_spread': current_line.get('spread'),
            'opening_ml':    current_line.get('moneyline'),
            'total_movement': 0.0,
        }

    entry = line_history[gid]

    # Add snapshot
    snapshot = {
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'spread':    current_line.get('spread'),
        'total':     current_line.get('total'),
        'home_ml':   current_line.get('moneyline'),
    }
    entry['snapshots'].append(snapshot)

    # Keep last 20 snapshots
    if len(entry['snapshots']) > 20:
        entry['snapshots'] = entry['snapshots'][-20:]

    # Update current values
    if current_line.get('spread') is not None:
        entry['current_spread'] = current_line.get('spread')
        if entry.get('opening_spread') is not None:
            entry['total_movement'] = entry['current_spread'] - entry['opening_spread']


def _ml_to_prob(ml: float) -> float:
    """Converts American moneyline to implied probability."""
    if ml is None:
        return 0.5
    try:
        ml = float(ml)
        if ml > 0:
            return 100.0 / (ml + 100.0)
        else:
            return abs(ml) / (abs(ml) + 100.0)
    except (ValueError, TypeError):
        return 0.5


# ============================================================
# SECTION 6: REDDIT SCRAPER
# ============================================================

def _fetch_reddit_posts(sport: str, team_name: str, req_counter: dict) -> List[dict]:
    """
    Fetches recent Reddit posts mentioning a team from sport-specific subreddit.
    Uses old.reddit.com JSON endpoint (no API key required).
    Returns list of post dicts with title, selftext, score.
    """
    if req_counter.get('reddit', 0) >= RATE_LIMITS['reddit']:
        return []

    sub = REDDIT_SUBS.get(sport, 'sports')
    # Limit team name to first two words for broader matches
    query = ' '.join(team_name.split()[:2])

    url = f'{REDDIT_BASE}/r/{sub}/search.json'
    params = {'q': query, 'sort': 'new', 'limit': '15', 'restrict_sr': '1', 't': 'week'}

    try:
        resp = _http_get_raw(url, params=params, headers={'User-Agent': 'edge-predict/1.0'})
        req_counter['reddit'] = req_counter.get('reddit', 0) + 1
        time.sleep(2)  # 1 req per 2s per Reddit guidelines

        if not resp:
            return []

        data = json.loads(resp)
        posts = []
        for child in (data.get('data') or {}).get('children', []):
            post = child.get('data') or {}
            if not post:
                continue
            posts.append({
                'title':    post.get('title', ''),
                'body':     post.get('selftext', ''),
                'score':    int(post.get('score', 0)),
                'created':  post.get('created_utc', 0),
                'url':      post.get('url', ''),
            })
        return posts

    except Exception as exc:
        log.debug(f'[edge] Reddit error for {sport}/{team_name}: {exc}')
        return []


# ============================================================
# SECTION 7: NITTER/TWITTER SCRAPER
# ============================================================

def _fetch_nitter_tweets(account: str, req_counter: dict) -> List[str]:
    """
    Fetches recent tweets from a reporter account via Nitter mirrors.
    Returns list of tweet text strings.
    """
    if req_counter.get('nitter', 0) >= RATE_LIMITS['nitter']:
        return []

    for mirror in NITTER_MIRRORS:
        try:
            url = f'{mirror}/{account}/rss'
            resp = _http_get_raw(url, headers={'User-Agent': 'edge-predict/1.0'}, timeout=15)
            req_counter['nitter'] = req_counter.get('nitter', 0) + 1
            time.sleep(3)

            if not resp:
                continue

            # Parse RSS XML
            texts = []
            try:
                root = ET.fromstring(resp)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                # Try Atom format first
                for entry in root.findall('.//item'):
                    title = entry.findtext('title') or ''
                    desc  = entry.findtext('description') or ''
                    # Strip HTML tags
                    clean = re.sub(r'<[^>]+>', ' ', desc + ' ' + title)
                    clean = re.sub(r'\s+', ' ', clean).strip()
                    if clean:
                        texts.append(clean)
            except ET.ParseError:
                # Try plain text extraction
                texts = re.findall(r'<title>(.*?)</title>', resp, re.DOTALL)[1:]
                texts = [re.sub(r'<[^>]+>', '', t).strip() for t in texts if t.strip()]

            return texts[:20]

        except Exception:
            continue

    return []


# ============================================================
# SECTION 8: ESPN NEWS FEED PARSER
# ============================================================

def _fetch_espn_rss(sport: str, req_counter: dict,
                    cache: dict, cache_ttl_hours: int = 12) -> List[str]:
    """
    Fetches ESPN RSS news feed for a sport.
    Returns list of news item text strings.
    Caches results for cache_ttl_hours to avoid duplicate fetches.
    """
    cache_key = f'espn_rss_{sport}'

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        try:
            fetched_at = datetime.datetime.fromisoformat(cached.get('fetched_at', ''))
            age_hours  = (datetime.datetime.utcnow() - fetched_at).total_seconds() / 3600
            if age_hours < cache_ttl_hours:
                return cached.get('items', [])
        except (ValueError, TypeError):
            pass

    if req_counter.get('espn_rss', 0) >= RATE_LIMITS['espn_rss']:
        return []

    rss_sport = {
        'nfl': 'nfl', 'nba': 'nba', 'mlb': 'mlb', 'nhl': 'nhl',
        'ncaaf': 'college-football', 'ncaabm': 'mens-college-basketball',
        'ncaabw': 'womens-college-basketball',
    }.get(sport, sport)

    url = f'{ESPN_RSS_BASE}/{rss_sport}/news'

    try:
        resp = _http_get_raw(url, timeout=20)
        req_counter['espn_rss'] = req_counter.get('espn_rss', 0) + 1
        time.sleep(2)

        if not resp:
            return []

        # Parse RSS
        items = []
        try:
            root = ET.fromstring(resp)
            for item in root.findall('.//item'):
                title = item.findtext('title') or ''
                desc  = item.findtext('description') or ''
                clean = re.sub(r'<[^>]+>', ' ', title + ' ' + desc)
                clean = re.sub(r'\s+', ' ', clean).strip()
                if clean:
                    items.append(clean)
        except ET.ParseError:
            items = re.findall(r'<title>(.*?)</title>', resp, re.DOTALL)[1:30]
            items = [re.sub(r'<[^>]+>', '', i).strip() for i in items if i.strip()]

        # Cache result
        cache[cache_key] = {
            'fetched_at': datetime.datetime.utcnow().isoformat(),
            'items': items[:50],
        }

        return items[:50]

    except Exception as exc:
        log.debug(f'[edge] ESPN RSS error for {sport}: {exc}')
        return []


def _fetch_espn_injuries_text(sport: str, req_counter: dict,
                               cache: dict, cache_ttl_hours: int = 6) -> List[str]:
    """
    Fetches ESPN injury report page text for a sport.
    Returns list of injury text strings.
    """
    cache_key = f'espn_inj_{sport}'
    cached = cache.get(cache_key)
    if cached:
        try:
            fetched_at = datetime.datetime.fromisoformat(cached.get('fetched_at', ''))
            age_hours  = (datetime.datetime.utcnow() - fetched_at).total_seconds() / 3600
            if age_hours < cache_ttl_hours:
                return cached.get('items', [])
        except (ValueError, TypeError):
            pass

    if req_counter.get('espn_inj', 0) >= RATE_LIMITS['espn_inj']:
        return []

    inj_path = ESPN_INJURY_PATHS.get(sport)
    if not inj_path:
        return []

    url = f'https://www.espn.com/{inj_path}'

    try:
        resp = _http_get_raw(url, headers={'User-Agent': 'edge-predict/1.0'}, timeout=20)
        req_counter['espn_inj'] = req_counter.get('espn_inj', 0) + 1
        time.sleep(2)

        if not resp:
            return []

        # Extract text content - look for injury-related text blocks
        # Strip all HTML tags and extract readable text chunks
        text = re.sub(r'<script[^>]*>.*?</script>', ' ', resp, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Extract sentences that look like injury reports
        injury_keywords = [
            'injured', 'injury', 'out', 'questionable', 'doubtful', 'day-to-day',
            'placed on', 'activated', 'practiced', 'limited', 'did not practice',
            'full practice', 'cleared', 'surgery', 'return', 'listed'
        ]
        sentences = re.split(r'[.!?]', text)
        injury_items = []
        for s in sentences:
            s = s.strip()
            if len(s) < 15 or len(s) > 300:
                continue
            if any(kw in s.lower() for kw in injury_keywords):
                injury_items.append(s)

        items = injury_items[:40]
        cache[cache_key] = {
            'fetched_at': datetime.datetime.utcnow().isoformat(),
            'items': items,
        }
        return items

    except Exception as exc:
        log.debug(f'[edge] ESPN injuries error for {sport}: {exc}')
        return []


# ============================================================
# SECTION 9: SENTIMENT SCORING ENGINE
# ============================================================

INJURY_LEXICON = {
    'full participant':              +0.90,
    'full practice':                 +0.85,
    'no injury designation':         +0.95,
    'cleared':                       +0.90,
    'activated':                     +0.80,
    'questionable but expected':     +0.70,
    'limited but practiced':         +0.50,
    'probable':                      +0.85,
    'day-to-day':                    +0.30,
    'day to day':                    +0.30,
    'returned to practice':          +0.50,
    'game-time decision':            +0.20,
    'game time decision':            +0.20,

    'out':                           -0.95,
    'placed on ir':                  -1.00,
    'placed on the injured reserve': -1.00,
    'did not practice':              -0.70,
    'did not participate':           -0.70,
    'doubtful':                      -0.80,
    'ruled out':                     -0.95,
    'not expected to play':          -0.85,
    'will not play':                 -0.90,
    'will see how he feels':         -0.50,
    'will see how she feels':        -0.50,
    'limited':                       -0.30,
    'questionable':                  -0.20,
    'non-contact':                   -0.40,
    'missed practice':               -0.50,
    'worse than initially':          -0.90,
    'worse than originally':         -0.90,
    'surgery':                       -0.85,
}

TEAM_MORALE_LEXICON = {
    'trade demand':           -0.70,
    'wants out':              -0.65,
    'locker room':            -0.40,
    'chemistry issues':       -0.50,
    'frustrated':             -0.40,
    'motivated':              +0.40,
    'statement game':         +0.30,
    'revenge game':           +0.40,
    'embarrassed':            +0.30,
    'rallied':                +0.35,
    'energized':              +0.35,
    'focused':                +0.25,
    'distraction':            -0.45,
    'feuding':                -0.55,
    'unhappy':                -0.40,
}

COACHING_LEXICON = {
    'head coach fired':       -0.60,
    'coaching change':        -0.30,
    'interim head coach':     -0.45,
    'confident in':           +0.30,
    'expects to play':        +0.70,
    'should be available':    +0.65,
    'will be ready':          +0.70,
    'not concerned':          +0.40,
    'preparing as normal':    +0.40,
    'game plan limited':      -0.35,
}

COMBINED_LEXICON = {**INJURY_LEXICON, **TEAM_MORALE_LEXICON, **COACHING_LEXICON}

# Patterns that override individual word scores
OVERRIDE_PATTERNS = [
    (r'worse than (initially|originally) thought', -0.90),
    (r'(coach|head coach) .{0,50} (confident|expect|should|will) .{0,30} (play|available|ready)', +0.65),
    (r'not (concerned|worried) .{0,20} (week|game|Sunday|Monday|Friday)', +0.35),
    (r'full (participant|practice) .{0,20} (wednesday|thursday|friday)', +0.80),
    (r'did not practice (all|this) week', -0.75),
    (r'placed on (the )?injured reserve', -1.00),
    (r'season.ending (injury|surgery)', -1.00),
    (r'ruled out for (season|year)', -1.00),
]


def score_text_sentiment(text: str, team_name: str = '') -> float:
    """
    Scores text for sentiment relevant to a team.
    Returns float in [-1, 1]. 0 = neutral.
    Checks override patterns first, then lexicon.
    """
    if not text:
        return 0.0

    text_lower = text.lower()
    total_score = 0.0
    n_matches   = 0

    # Check override patterns first
    for pattern, score in OVERRIDE_PATTERNS:
        if re.search(pattern, text_lower):
            total_score += score
            n_matches   += 1

    if n_matches > 0:
        return max(-1.0, min(1.0, total_score / n_matches))

    # Lexicon scan
    for phrase, score in sorted(COMBINED_LEXICON.items(), key=lambda x: -len(x[0])):
        if phrase in text_lower:
            total_score += score
            n_matches   += 1

    if n_matches == 0:
        return 0.0

    return max(-1.0, min(1.0, total_score / n_matches))


def score_texts_for_team(texts: List[str], team_name: str) -> float:
    """
    Scores a list of text items for a specific team.
    Only scores items that mention the team.
    Returns average sentiment score [-1, 1].
    """
    if not texts or not team_name:
        return 0.0

    team_words = team_name.lower().split()
    team_keywords = team_words[-2:] if len(team_words) >= 2 else team_words  # Last 2 words usually identify team

    relevant_scores = []

    for text in texts:
        if not text:
            continue
        text_lower = text.lower()
        # Only score text that mentions this team
        if not any(kw in text_lower for kw in team_keywords):
            continue
        score = score_text_sentiment(text, team_name)
        relevant_scores.append(score)

    if not relevant_scores:
        return 0.0

    return sum(relevant_scores) / len(relevant_scores)


# ============================================================
# SECTION 10: INJURY REPORT LANGUAGE CLASSIFIER
# ============================================================

def parse_injury_designation(text: str) -> float:
    """
    Classifies injury report language and returns availability probability [0,1].
    Higher = more likely to play. 0.5 = standard 'questionable'.
    """
    text_lower = text.lower()

    # Hard rules first (high confidence patterns)
    if any(p in text_lower for p in ['ruled out', 'will not play', 'placed on ir', 'out for season']):
        return 0.02

    if any(p in text_lower for p in ['doubtful', 'highly unlikely', 'not expected to play']):
        # Check if there's a mitigating factor
        if 'returned to practice' in text_lower or 'game-time decision' in text_lower:
            return 0.25
        return 0.12

    if any(p in text_lower for p in ['full participant', 'full practice', 'no injury designation', 'cleared']):
        return 0.95

    if 'did not practice all week' in text_lower or 'missed all practice' in text_lower:
        return 0.12

    # NFL-specific: combine practice + designation
    full_practice_days = len(re.findall(r'full (participant|practice)', text_lower))
    limited_days       = len(re.findall(r'limited', text_lower))
    dnp_days           = len(re.findall(r'did not (practice|participate)', text_lower))

    if 'questionable' in text_lower:
        if full_practice_days >= 2:
            return 0.72   # Questionable but practiced fully
        if dnp_days >= 2:
            return 0.18   # Questionable and barely practiced
        return 0.45        # Standard questionable

    if 'probable' in text_lower:
        return 0.87

    if 'limited' in text_lower and 'questionable' not in text_lower:
        return 0.65

    if 'game-time decision' in text_lower or 'game time decision' in text_lower:
        return 0.50

    if 'day-to-day' in text_lower or 'day to day' in text_lower:
        return 0.70

    # Default: slight uncertainty
    return 0.60


# ============================================================
# SECTION 11: PRESS CONFERENCE / COACH QUOTE ANALYZER
# ============================================================

def analyze_coach_quotes(texts: List[str], team_name: str) -> float:
    """
    Analyzes coach or press conference quotes for a team.
    Returns sentiment score [-1, 1].
    """
    if not texts:
        return 0.0

    team_lower = team_name.lower()
    team_words = team_lower.split()
    lookup_words = team_words[-2:] if len(team_words) >= 2 else team_words

    coach_patterns = [
        (r'(optimistic|positive|confident|good news)', +0.50),
        (r'(concerned|worried|serious|significant)', -0.50),
        (r'(day-to-day|game-time|wait and see)', -0.15),
        (r'(surgery|season.ending)', -0.90),
        (r'(full go|full speed|ready to go)', +0.80),
        (r'(limited role|reduced role|snap count)', -0.40),
    ]

    scores = []
    for text in texts:
        text_lower = text.lower()
        if not any(w in text_lower for w in lookup_words):
            continue
        for pattern, score in coach_patterns:
            if re.search(pattern, text_lower):
                scores.append(score)
                break

    return sum(scores) / len(scores) if scores else 0.0


# ============================================================
# SECTION 12: SENTIMENT AGGREGATION
# ============================================================

def compute_sentiment_package(
    game_id: str,
    home_team: str,
    away_team: str,
    sport: str,
    all_texts: Dict[str, List[str]],
    injury_texts: List[str],
    source_multipliers: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Aggregates all text signals into a sentiment package for a game.
    Returns dict with home/away sentiment scores and prob_adjustment.

    source_multipliers: optional dict of source name → weight multiplier.
    Keys: 'injury', 'reporter', 'news', 'social'. Defaults to 1.0 for
    any missing key. Applied before renormalisation so relative weights
    are preserved even when some sources have no data.
    """
    mults = source_multipliers or {}
    m_inj      = max(0.0, mults.get('injury',   1.0))
    m_reporter = max(0.0, mults.get('reporter',  1.0))
    m_news     = max(0.0, mults.get('news',      1.0))
    m_social   = max(0.0, mults.get('social',    1.0))

    # Score from different sources
    home_inj   = score_texts_for_team(injury_texts, home_team)
    away_inj   = score_texts_for_team(injury_texts, away_team)

    home_news  = score_texts_for_team(all_texts.get('espn_rss', []), home_team)
    away_news  = score_texts_for_team(all_texts.get('espn_rss', []), away_team)

    home_social = score_texts_for_team(all_texts.get('reddit', []), home_team)
    away_social = score_texts_for_team(all_texts.get('reddit', []), away_team)

    home_reporter = score_texts_for_team(all_texts.get('reporter', []), home_team)
    away_reporter = score_texts_for_team(all_texts.get('reporter', []), away_team)

    # Base weights before multiplier application
    base_weights = [
        (home_inj,      away_inj,      0.40, m_inj),
        (home_reporter, away_reporter, 0.30, m_reporter),
        (home_news,     away_news,     0.20, m_news),
        (home_social,   away_social,   0.10, m_social),
    ]

    # Weighted aggregate with adaptive multipliers applied and renormalised
    def weighted_avg_pair(scores_home, scores_away, weights_mults):
        total_w = sum(bw * m for _, _, bw, m in weights_mults
                      if bw * m > 0)
        if total_w == 0:
            return 0.0, 0.0
        h = sum(sh * bw * m for sh, _, bw, m in weights_mults) / total_w
        a = sum(sa * bw * m for _, sa, bw, m in weights_mults) / total_w
        return h, a

    home_sentiment, away_sentiment = weighted_avg_pair(
        None, None, base_weights
    )

    sentiment_delta  = home_sentiment - away_sentiment
    max_impact       = SENTIMENT_MAX_IMPACT.get(sport, 0.05)

    # tanh ensures bounded, smoothly diminishing adjustment
    prob_adjustment = math.tanh(sentiment_delta) * max_impact

    n_sources = sum(1 for lst in all_texts.values() if lst) + (1 if injury_texts else 0)
    confidence = min(0.90, 0.40 + n_sources * 0.08)

    key_signals = []
    if home_inj < -0.5:
        key_signals.append(f'{home_team} significant injury signal')
    if away_inj < -0.5:
        key_signals.append(f'{away_team} significant injury signal')
    if home_inj > 0.7:
        key_signals.append(f'{home_team} fully healthy signal')
    if away_inj > 0.7:
        key_signals.append(f'{away_team} fully healthy signal')

    return {
        'game_id':               str(game_id),
        'home_team_sentiment':   round(home_sentiment, 4),
        'away_team_sentiment':   round(away_sentiment, 4),
        'sentiment_delta':       round(sentiment_delta, 4),
        'home_injury_score':     round(home_inj, 4),
        'away_injury_score':     round(away_inj, 4),
        'home_morale_score':     round(home_news, 4),
        'away_morale_score':     round(away_news, 4),
        'home_reporter_score':   round(home_reporter, 4),
        'away_reporter_score':   round(away_reporter, 4),
        'home_social_score':     round(home_social, 4),
        'away_social_score':     round(away_social, 4),
        'sources_counted':       n_sources,
        'confidence':            round(confidence, 4),
        'key_signals':           key_signals,
        'prob_adjustment':       round(prob_adjustment, 4),
        'fetched_at':            datetime.datetime.utcnow().isoformat(),
    }


# ============================================================
# SECTION 12b: ADAPTIVE SIGNAL WEIGHT LEARNING (Fix E)
# ============================================================

def get_adaptive_multiplier(data: dict, sport: str, signal_type: str) -> float:
    """
    Returns the learned accuracy multiplier for a signal type in a sport.
    Returns 1.0 (neutral) until SIGNAL_MIN_SAMPLE resolved predictions exist.
    """
    entry = (data.get('edge', {})
               .get('signal_accuracy', {})
               .get(sport, {})
               .get(signal_type, {}))
    return entry.get('multiplier', 1.0)


def evaluate_resolved_signals(data: dict) -> None:
    """
    Scans all resolved predictions and scores each sharp/sentiment signal
    that contributed to that prediction as correct or incorrect.

    A signal is correct when its directional prediction (toward_home or
    prob_adjustment > 0 meaning toward home) matches the actual game outcome.

    Only processes each game_id once (tracked via _processed_ids).
    Caps _processed_ids at 500 entries to bound memory growth.
    """
    ensure_edge_keys(data)

    sig_acc     = data['edge']['signal_accuracy']
    processed   = sig_acc.get('_processed_ids', [])
    processed_set = set(processed)

    sharp_signals = data['edge'].get('sharp_signals', {})
    sentiment     = data['edge'].get('sentiment', {})

    for sport in ALL_SPORTS:
        preds = data.get('predictions', {}).get(sport, [])
        for pred in preds:
            if not isinstance(pred, dict):
                continue

            status = pred.get('status', '')
            if status not in ('correct', 'wrong'):
                continue

            game_id = str(pred.get('id', ''))
            if not game_id or game_id in processed_set:
                continue

            home = (pred.get('home') or '').lower().strip()
            away = (pred.get('away') or '').lower().strip()
            pick = (pred.get('pick') or '').lower().strip()

            # Determine actual outcome: did home win?
            if pick == home:
                home_won = (status == 'correct')
            elif pick == away:
                home_won = (status == 'wrong')
            else:
                # pick doesn't cleanly match home or away — skip
                processed_set.add(game_id)
                processed.append(game_id)
                continue

            # --- Evaluate sharp signal ---
            sharp = sharp_signals.get(game_id, {})
            sharp_type     = sharp.get('signal_type')
            sharp_toward   = sharp.get('toward_home')

            if sharp_type and sharp_type != 'consensus' and sharp_toward is not None:
                sharp_correct = (
                    (sharp_toward is True  and home_won) or
                    (sharp_toward is False and not home_won)
                )
                sport_acc = sig_acc.setdefault(sport, {})
                entry     = sport_acc.setdefault(sharp_type, {'correct': 0, 'total': 0, 'multiplier': 1.0})
                entry['total']   += 1
                entry['correct'] += 1 if sharp_correct else 0

            # --- Evaluate sentiment signal ---
            sent = sentiment.get(game_id, {})
            prob_adj = sent.get('prob_adjustment', 0.0)

            if prob_adj != 0.0:
                sent_toward_home = (prob_adj > 0.0)
                sent_correct = (
                    (sent_toward_home and home_won) or
                    (not sent_toward_home and not home_won)
                )
                sport_acc = sig_acc.setdefault(sport, {})
                entry     = sport_acc.setdefault('sentiment', {'correct': 0, 'total': 0, 'multiplier': 1.0})
                entry['total']   += 1
                entry['correct'] += 1 if sent_correct else 0

            processed_set.add(game_id)
            processed.append(game_id)

    # Cap processed list to avoid unbounded growth
    if len(processed) > 500:
        processed = processed[-500:]

    sig_acc['_processed_ids'] = processed


def compute_adaptive_multipliers(data: dict) -> None:
    """
    Recalculates accuracy multipliers for all signal types that have
    reached SIGNAL_MIN_SAMPLE resolved predictions.

    Multiplier formula:
        accuracy  = correct / total
        raw       = (accuracy - 0.5) / 0.5 * 1.5
        multiplier = clamp(raw, 0.0, 1.5)

    This means:
        50% accuracy (random) → multiplier 0.0 (signal disabled)
        75% accuracy          → multiplier 0.75
        100% accuracy         → multiplier 1.5 (max boost)

    Multiplier stays at 1.0 (unchanged) until min sample is reached.
    """
    ensure_edge_keys(data)
    sig_acc = data['edge']['signal_accuracy']

    for sport, signals in sig_acc.items():
        if sport == '_processed_ids' or not isinstance(signals, dict):
            continue
        for signal_type, entry in signals.items():
            if not isinstance(entry, dict):
                continue
            total = entry.get('total', 0)
            if total < SIGNAL_MIN_SAMPLE:
                entry['multiplier'] = 1.0  # Not enough data — stay neutral
                continue
            accuracy   = entry.get('correct', 0) / total
            raw        = (accuracy - 0.5) / 0.5 * 1.5
            multiplier = max(0.0, min(1.5, raw))
            entry['multiplier'] = round(multiplier, 4)


# ============================================================
# SECTION 12c: POLYMARKET PREDICTION MARKET INTEGRATION (Fix F)
# ============================================================

POLYMARKET_MARKETS_URL = 'https://gamma-api.polymarket.com/markets'

# Series tickers Polymarket uses for sports we track
POLYMARKET_SERIES_TICKERS = {
    'nfl':    'nfl',
    'nba':    'nba',
    'mlb':    'mlb',
    'nhl':    'nhl',
    'ncaaf':  'ncaaf',
    'ncaabm': 'ncaab',
    'ncaabw': 'ncaaw',
}


def _normalize_team_name(name: str) -> str:
    """Lowercases and strips punctuation for fuzzy team name matching."""
    name = name.lower()
    name = re.sub(r"[^\w\s]", '', name)
    return name.strip()


def _match_team_in_outcome(outcome: str, team_name: str) -> bool:
    """
    Returns True if outcome string plausibly refers to team_name.
    Checks:
      1. Normalized outcome is substring of normalized team name
      2. Normalized team name is substring of normalized outcome
      3. Any word in outcome (len >= 4) appears in normalized team name
    All checks are case-insensitive.
    """
    norm_outcome = _normalize_team_name(outcome)
    norm_team    = _normalize_team_name(team_name)

    if not norm_outcome or not norm_team:
        return False

    if norm_outcome in norm_team:
        return True
    if norm_team in norm_outcome:
        return True

    # Word-level match — skip very short words to avoid false positives
    for word in norm_outcome.split():
        if len(word) >= 4 and word in norm_team:
            return True

    return False


def fetch_polymarket_probs(
    upcoming_games: List[dict],
    req_counter: dict,
    cache: dict,
) -> Dict[str, dict]:
    """
    Fetches active Polymarket sports markets and matches them to upcoming games.

    Returns dict keyed by game_id:
        {
            'home_prob':       float,   # implied win probability for home team
            'away_prob':       float,   # implied win probability for away team
            'prob_adjustment': float,   # tanh-scaled delta vs 0.5 baseline
            'market_title':    str,
            'liquidity':       float,
            'fetched_at':      str,
        }

    Only markets with liquidity > 0 and exactly two outcomes (binary win/loss)
    are used. Markets that cannot be unambiguously matched to a game are skipped.
    """
    if req_counter.get('polymarket', 0) >= RATE_LIMITS['polymarket']:
        return {}

    # Build index of upcoming games for fast lookup
    # Key: frozenset of normalized home + away words for matching
    game_index: List[dict] = []
    for g in upcoming_games:
        home = g.get('home', '')
        away = g.get('away', '')
        if home and away:
            game_index.append({
                'game_id': g.get('id') or f"{g['sport']}_{home.lower()}_{away.lower()}_{g.get('date','')[:10]}",
                'sport':   g['sport'],
                'home':    home,
                'away':    away,
            })

    if not game_index:
        return {}

    results: Dict[str, dict] = {}

    # Fetch one page of active, non-closed markets. Polymarket doesn't expose
    # a reliable sport filter on the gamma API, so we fetch a broad set and
    # filter client-side by outcome matching.
    # limit=500 keeps us within 1–2 requests while covering most sports markets.
    params = {
        'active':  'true',
        'closed':  'false',
        'limit':   '500',
    }

    try:
        raw = _http_get_raw(POLYMARKET_MARKETS_URL, params=params, timeout=25)
        req_counter['polymarket'] = req_counter.get('polymarket', 0) + 1

        if not raw:
            return {}

        markets = json.loads(raw)
        if not isinstance(markets, list):
            return {}

    except Exception as exc:
        log.debug(f'[edge] Polymarket fetch error: {exc}')
        return {}

    for market in markets:
        if not isinstance(market, dict):
            continue

        # Skip markets with no liquidity — stale or never traded
        try:
            liquidity = float(market.get('liquidityNum') or market.get('liquidity') or 0)
        except (ValueError, TypeError):
            liquidity = 0.0
        if liquidity <= 0:
            continue

        # Parse outcomes and prices — both are JSON-encoded strings in the API
        try:
            outcomes_raw     = market.get('outcomes', '[]')
            out_prices_raw   = market.get('outcomePrices', '[]')
            outcomes         = json.loads(outcomes_raw)   if isinstance(outcomes_raw,   str) else outcomes_raw
            outcome_prices   = json.loads(out_prices_raw) if isinstance(out_prices_raw, str) else out_prices_raw
        except (json.JSONDecodeError, TypeError):
            continue

        # Only binary markets (exactly 2 outcomes) are meaningful for win prob
        if not isinstance(outcomes, list) or len(outcomes) != 2:
            continue
        if not isinstance(outcome_prices, list) or len(outcome_prices) != 2:
            continue

        try:
            prob_0 = float(outcome_prices[0])
            prob_1 = float(outcome_prices[1])
        except (ValueError, TypeError):
            continue

        # Prices must sum to ~1.0 and be valid probabilities
        if not (0.0 <= prob_0 <= 1.0 and 0.0 <= prob_1 <= 1.0):
            continue
        if abs(prob_0 + prob_1 - 1.0) > 0.10:
            continue

        title = market.get('question') or market.get('title') or ''

        # Try to match this market to an upcoming game
        for game in game_index:
            home = game['home']
            away = game['away']
            game_id = game['game_id']

            if game_id in results:
                continue  # Already matched from a higher-liquidity market

            # Determine which outcome index corresponds to which team.
            # Both teams must match different outcomes — no partial credit.
            home_idx = None
            away_idx = None

            for i, outcome in enumerate(outcomes):
                if _match_team_in_outcome(outcome, home):
                    home_idx = i
                elif _match_team_in_outcome(outcome, away):
                    away_idx = i

            # Require unambiguous two-sided match
            if home_idx is None or away_idx is None:
                continue
            if home_idx == away_idx:
                continue

            home_prob = prob_0 if home_idx == 0 else prob_1
            away_prob = prob_0 if away_idx == 0 else prob_1

            # prob_adjustment: positive = market favors home, bounded by tanh
            prob_adjustment = round(
                math.tanh(home_prob - away_prob) * PREDICTION_MARKET_MAX_IMPACT,
                4
            )

            results[game_id] = {
                'home_prob':       round(home_prob, 4),
                'away_prob':       round(away_prob, 4),
                'prob_adjustment': prob_adjustment,
                'market_title':    title,
                'liquidity':       round(liquidity, 2),
                'source':          'polymarket',
                'fetched_at':      datetime.datetime.utcnow().isoformat(),
            }
            break  # One match per market

    log.info(f'[edge] Polymarket: matched {len(results)} markets')
    return results


# ============================================================
# SECTION 13: SIGNAL EXPORT
# ============================================================

def export_sharp_signal(data: dict, game_id: str, sharp_result: dict) -> None:
    """Writes sharp money signal to data['edge']['sharp_signals']."""
    ensure_edge_keys(data)
    if sharp_result.get('signal_type'):
        data['edge']['sharp_signals'][str(game_id)] = sharp_result


def export_sentiment(data: dict, game_id: str, sentiment_pkg: dict) -> None:
    """Writes sentiment package to data['edge']['sentiment']."""
    ensure_edge_keys(data)
    data['edge']['sentiment'][str(game_id)] = sentiment_pkg


def _cleanup_old_signals(data: dict, days_back: int = 7) -> None:
    """Removes edge signals for games older than days_back days."""
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')

    for section in ('line_history', 'sharp_signals', 'sentiment', 'prediction_markets'):
        store = data['edge'].get(section, {})
        old_keys = []
        for gid, entry in list(store.items()):
            if isinstance(entry, dict):
                date = entry.get('game_date', entry.get('fetched_at', ''))
                if date and date[:10] < cutoff:
                    old_keys.append(gid)
        for k in old_keys:
            store.pop(k, None)


# ============================================================
# SECTION 14: UPDATE ENTRY POINT
# ============================================================

def update_edge(data: dict) -> None:
    """
    Daily update run for edge.py.
    1. Evaluates resolved predictions to update signal accuracy learning
    2. Recalculates adaptive multipliers from accumulated accuracy data
    3. Fetches odds lines and detects sharp money signals (with multipliers applied)
    4. Fetches Polymarket prediction market probabilities
    5. Fetches text from ESPN RSS, Reddit, and reporter accounts
    6. Scores sentiment for all upcoming games (with adaptive source multipliers)
    7. Writes all signals to data['edge']

    Called by brain.py --mode update after _auto_verify_predictions().
    Never raises — all errors logged to data['edge']['error_log'].
    """
    ensure_edge_keys(data)

    req_counter: Dict[str, int] = {}
    cache = data['edge'].get('cache', {})
    error_log = data['edge'].get('error_log', [])

    log.info('[edge] Starting update...')

    # Step 1: Update signal accuracy from resolved predictions
    try:
        evaluate_resolved_signals(data)
        compute_adaptive_multipliers(data)
        log.info('[edge] Adaptive learning updated.')
    except Exception as exc:
        msg = f'[edge] Adaptive learning error: {exc}'
        log.warning(msg)
        error_log.append({'time': datetime.datetime.utcnow().isoformat(), 'error': msg})

    # Get upcoming games from brain.py's scoreboard data (already in data)
    # We read from data['predictions'] to find game IDs and teams
    upcoming_games: List[dict] = []
    for sport in ALL_SPORTS:
        preds = data.get('predictions', {}).get(sport, [])
        for p in preds:
            if isinstance(p, dict) and p.get('status') == 'pending':
                upcoming_games.append({
                    'id':    p.get('id', ''),
                    'sport': sport,
                    'home':  p.get('home', ''),
                    'away':  p.get('away', ''),
                    'date':  p.get('date', ''),
                })

    # --- PART 1: Line movement intelligence ---
    if ODDS_API_KEY:
        log.info('[edge] Fetching odds lines...')
        for sport in ALL_SPORTS:
            try:
                raw_odds = _fetch_odds_lines(sport, req_counter)
                if not raw_odds:
                    continue

                parsed_lines = _parse_odds_lines(raw_odds)

                for game in upcoming_games:
                    if game['sport'] != sport:
                        continue

                    home_lower = game['home'].lower()
                    away_lower = game['away'].lower()
                    line_key   = f'{home_lower}|{away_lower}'
                    line_data  = parsed_lines.get(line_key)

                    if not line_data:
                        # Try reverse
                        rev_key = f'{away_lower}|{home_lower}'
                        line_data = parsed_lines.get(rev_key)

                    if not line_data:
                        continue

                    game_id = game.get('id') or f"{sport}_{home_lower}_{away_lower}_{game['date'][:10]}"

                    # Update line history
                    update_line_history(
                        data, game_id, sport,
                        game['home'], game['away'],
                        line_data, game.get('date', '')
                    )

                    # Detect sharp signal
                    sharp = detect_sharp_money(
                        game_id,
                        line_data,
                        data['edge']['line_history'],
                        sport,
                        data['edge'].get('public_teams', PUBLIC_TEAMS),
                    )

                    # Apply adaptive multiplier to prob_delta (not to confidence)
                    if sharp.get('signal_type') and sharp['signal_type'] != 'consensus':
                        multiplier = get_adaptive_multiplier(
                            data, sport, sharp['signal_type']
                        )
                        sharp['prob_delta'] = round(
                            sharp.get('prob_delta', 0.0) * multiplier, 4
                        )
                        sharp['adaptive_multiplier'] = multiplier

                    export_sharp_signal(data, game_id, sharp)

            except Exception as exc:
                msg = f'[edge] Odds processing error for {sport}: {exc}'
                log.warning(msg)
                error_log.append({'time': datetime.datetime.utcnow().isoformat(), 'error': msg})
    else:
        log.info('[edge] ODDS_API_KEY not set — skipping line movement')

    # --- PART 2: Prediction market probabilities (Polymarket) ---
    log.info('[edge] Fetching Polymarket probabilities...')
    try:
        pm_results = fetch_polymarket_probs(upcoming_games, req_counter, cache)
        for game_id, pm_data in pm_results.items():
            data['edge']['prediction_markets'][str(game_id)] = pm_data
    except Exception as exc:
        msg = f'[edge] Polymarket error: {exc}'
        log.warning(msg)
        error_log.append({'time': datetime.datetime.utcnow().isoformat(), 'error': msg})

    # --- PART 3: Sentiment intelligence ---
    log.info('[edge] Gathering sentiment signals...')

    # Pre-fetch sport-level resources (RSS feeds, injury reports)
    sport_injury_texts: Dict[str, List[str]] = {}
    sport_rss_texts:    Dict[str, List[str]] = {}

    for sport in ALL_SPORTS:
        # Check if any upcoming games exist for this sport
        sport_games = [g for g in upcoming_games if g['sport'] == sport]
        if not sport_games:
            continue

        try:
            sport_rss_texts[sport]    = _fetch_espn_rss(sport, req_counter, cache)
        except Exception:
            sport_rss_texts[sport] = []

        try:
            sport_injury_texts[sport] = _fetch_espn_injuries_text(sport, req_counter, cache)
        except Exception:
            sport_injury_texts[sport] = []

    # Fetch reporter tweets once per sport
    reporter_texts: Dict[str, List[str]] = {}
    for sport in ALL_SPORTS:
        sport_games = [g for g in upcoming_games if g['sport'] == sport]
        if not sport_games:
            continue

        tweets = []
        for account in data['edge'].get('reporter_accounts', {}).get(sport, []):
            try:
                new_tweets = _fetch_nitter_tweets(account, req_counter)
                tweets.extend(new_tweets)
            except Exception:
                pass
        reporter_texts[sport] = tweets

    # Build source multipliers from sentiment signal accuracy
    def _build_source_multipliers(sport: str) -> Dict[str, float]:
        """Returns per-source adaptive multipliers for sentiment scoring."""
        sig_acc   = data['edge'].get('signal_accuracy', {})
        sent_acc  = sig_acc.get(sport, {}).get('sentiment', {})
        # Sentiment accuracy applies uniformly across sources for now.
        # Individual source tracking would require separate signal types.
        m = sent_acc.get('multiplier', 1.0)
        return {
            'injury':   m,
            'reporter': m,
            'news':     m,
            'social':   m,
        }

    # Per-game Reddit + sentiment
    processed_games = set()
    for game in upcoming_games:
        sport    = game['sport']
        home     = game['home']
        away     = game['away']
        game_id  = game.get('id') or f"{sport}_{home.lower()}_{away.lower()}_{game['date'][:10]}"

        if game_id in processed_games:
            continue
        processed_games.add(game_id)

        try:
            # Reddit posts for this game's teams
            reddit_texts = []
            if req_counter.get('reddit', 0) < RATE_LIMITS['reddit']:
                for team in [home, away]:
                    posts = _fetch_reddit_posts(sport, team, req_counter)
                    reddit_texts.extend([
                        p.get('title', '') + ' ' + p.get('body', '')
                        for p in posts
                    ])

            all_texts = {
                'espn_rss': sport_rss_texts.get(sport, []),
                'reddit':   reddit_texts,
                'reporter': reporter_texts.get(sport, []),
            }

            source_mults = _build_source_multipliers(sport)

            sentiment_pkg = compute_sentiment_package(
                game_id, home, away, sport,
                all_texts,
                sport_injury_texts.get(sport, []),
                source_multipliers=source_mults,
            )

            export_sentiment(data, game_id, sentiment_pkg)

        except Exception as exc:
            msg = f'[edge] Sentiment error for {home} vs {away}: {exc}'
            log.debug(msg)
            error_log.append({'time': datetime.datetime.utcnow().isoformat(), 'error': msg})

    # Cleanup and save
    _cleanup_old_signals(data)
    data['edge']['cache']        = cache
    data['edge']['error_log']    = error_log[-200:]  # Keep last 200 errors
    data['edge']['last_updated'] = datetime.datetime.utcnow().isoformat()

    n_sharp      = len([v for v in data['edge']['sharp_signals'].values()
                        if v.get('signal_type')])
    n_sentiment  = len(data['edge']['sentiment'])
    n_pm         = len(data['edge']['prediction_markets'])
    log.info(
        f'[edge] Complete. Sharp signals: {n_sharp}, '
        f'Sentiment: {n_sentiment}, Prediction markets: {n_pm}'
    )


# ============================================================
# SECTION 15: RATE LIMITING, RETRY LOGIC, ERROR HANDLING
# ============================================================

BACKOFF_DELAYS = [2, 5, 15, 30]


def _http_get(url: str, params: Optional[Dict] = None,
              headers: Optional[Dict] = None, timeout: int = 30) -> Optional[Any]:
    """GET request returning parsed JSON or None. Includes retry with backoff."""
    for attempt in range(3):
        try:
            resp = requests.get(
                url,
                params=params or {},
                headers=headers or {},
                timeout=timeout,
            )
            if resp.status_code == 200:
                try:
                    return resp.json()
                except Exception:
                    return None
            if resp.status_code == 429:
                time.sleep(60)
                continue
            if resp.status_code >= 500 and attempt < 2:
                time.sleep(BACKOFF_DELAYS[attempt])
                continue
            return None
        except requests.Timeout:
            if attempt < 2:
                time.sleep(BACKOFF_DELAYS[attempt])
            continue
        except Exception:
            return None
    return None


def _http_get_raw(url: str, params: Optional[Dict] = None,
                  headers: Optional[Dict] = None, timeout: int = 30) -> Optional[str]:
    """GET request returning raw text or None."""
    for attempt in range(3):
        try:
            resp = requests.get(
                url,
                params=params or {},
                headers=headers or {'User-Agent': 'edge-predict/1.0'},
                timeout=timeout,
            )
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 429:
                time.sleep(60)
                continue
            if resp.status_code >= 500 and attempt < 2:
                time.sleep(BACKOFF_DELAYS[attempt])
                continue
            return None
        except requests.Timeout:
            if attempt < 2:
                time.sleep(BACKOFF_DELAYS[attempt])
            continue
        except Exception:
            return None
    return None
