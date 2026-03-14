#!/usr/bin/env python3
"""
EDGE Predict — scout.py
Player-level impact modeling and referee/umpire tendency profiling.
Runs during train and update workflows, writes to data['scout'].
Never raises — all errors logged silently.

Section 1  — Imports and constants
Section 2  — data.json keys managed by scout.py
Section 3  — ESPN depth chart fetcher and parser
Section 4  — Player historical stats fetcher
Section 5  — Player Win Impact (PWI) calculator
Section 6  — Lineup impact model
Section 7  — Matchup-specific player performance
Section 8  — Player availability adjuster
Section 9  — Referee/umpire roster fetcher
Section 10 — Referee tendency profiler (NFL/NBA)
Section 11 — Umpire strike zone profiler (MLB)
Section 12 — Official impact calculator
Section 13 — Combined scout signal export
Section 14 — Training entry point
Section 15 — Update entry point
Section 16 — Rate limiting and error handling
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
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import requests

log = logging.getLogger(__name__)

ESPN_BASE      = 'https://site.api.espn.com/apis/site/v2/sports'
ESPN_API_BASE  = 'https://site.api.espn.com/apis/site/v2/sports'
MLB_STATS_BASE = 'https://statsapi.mlb.com/api/v1'

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

# Injury designation to availability probability mapping
DESIGNATION_PROB = {
    'out':                    0.02,
    'ir':                     0.01,
    'injured reserve':        0.01,
    'doubtful':               0.12,
    'questionable':           0.45,
    'probable':               0.87,
    'limited':                0.70,
    'full participant':       0.97,
    'full':                   0.95,
    'day-to-day':             0.72,
    'day to day':             0.72,
    'game-time decision':     0.55,
    'game time decision':     0.55,
    'not expected to play':   0.05,
    'will not play':          0.03,
}

# Positional PWI fallbacks when insufficient game data exists
POSITIONAL_PWI_FALLBACK = {
    'nfl': {
        'QB':        0.18, 'QB_backup_to_starter': 0.08,
        'LT':        0.05, 'WR':  0.04, 'RB':  0.03,
        'CB':        0.04, 'DE':  0.04, 'EDGE': 0.04,
        'OT':        0.03, 'C':   0.03, 'DT':  0.03,
        'LB':        0.03, 'S':   0.02, 'TE':  0.03,
        'default':   0.02,
    },
    'nba': {
        'PG_star':        0.14, 'SG_star': 0.12, 'SF_star': 0.13,
        'PF_star':        0.11, 'C_star':  0.11,
        'PG':             0.07, 'SG':      0.06, 'SF':     0.06,
        'PF':             0.05, 'C':       0.06,
        'starter_avg':    0.06, 'rotation': 0.02,
        'default':        0.01,
    },
    'mlb': {
        'SP1':       0.08, 'SP2': 0.05, 'SP3': 0.04,
        'RP_closer': 0.04, 'C':   0.03, 'CF':  0.03,
        'SS':        0.03, '2B':  0.02, '3B':  0.02,
        '1B':        0.02, 'LF':  0.02, 'RF':  0.02,
        'DH':        0.02, 'default': 0.01,
    },
    'nhl': {
        'G':          0.12, 'C_star': 0.10, 'LW_star': 0.09,
        'RW_star':    0.09, 'D_star':  0.07,
        'C':          0.05, 'LW': 0.04, 'RW': 0.04, 'D': 0.03,
        'default':    0.02,
    },
    'ncaabm': {
        'leading_scorer':  0.12, 'primary_ball_handler': 0.10,
        'starter_avg':     0.05, 'default': 0.01,
    },
    'ncaabw': {
        'leading_scorer':  0.13, 'primary_ball_handler': 0.11,
        'starter_avg':     0.06, 'default': 0.01,
    },
    'ncaaf': {
        'QB':     0.15, 'RB': 0.05, 'WR': 0.04,
        'LT':     0.04, 'DE': 0.04, 'default': 0.02,
    },
}

# Cap on maximum lineup adjustment per team
MAX_LINEUP_ADJUSTMENT = 0.25
MAX_OFFICIAL_IMPACT   = 0.04

# Minimum games to compute a player-specific PWI
MIN_GAMES_FOR_PWI = 10

# Rate limits per update run
RATE_LIMITS = {
    'espn_depth':   7,   # One per sport
    'espn_player':  80,  # Per individual player
    'espn_game':    40,  # Per upcoming game for officials
    'mlb_officials': 2,
}

BACKOFF_DELAYS = [2, 5, 15]

# ============================================================
# SECTION 2: DATA.JSON KEYS MANAGED BY SCOUT.PY
# ============================================================

def get_scout_default() -> dict:
    """Returns default structure for data['scout']."""
    return {
        'version': '1.0',
        'last_updated': None,
        'players': {},
        'officials': {s: {} for s in ALL_SPORTS},
        'team_style': {},
        'lineup_adjustments': {},
        'official_assignments': {},
        'team_stats_cache': {},
        'cache': {},
        'error_log': [],
    }


def ensure_scout_keys(data: dict) -> None:
    """Ensures data['scout'] has all required keys."""
    if 'scout' not in data or not isinstance(data.get('scout'), dict):
        data['scout'] = get_scout_default()
        return
    default = get_scout_default()
    for k, v in default.items():
        if k not in data['scout']:
            data['scout'][k] = copy.deepcopy(v)
    # Ensure all sports exist in officials
    for sport in ALL_SPORTS:
        data['scout']['officials'].setdefault(sport, {})


# ============================================================
# SECTION 3: ESPN DEPTH CHART FETCHER AND PARSER
# ============================================================

def fetch_depth_chart(sport: str, team_id: str, req_counter: dict) -> List[dict]:
    """
    Fetches ESPN depth chart for a team.
    Returns list of player dicts: {player_id, name, position, depth_rank, team_id}
    """
    if req_counter.get('espn_depth', 0) >= RATE_LIMITS['espn_depth']:
        return []
    if not team_id:
        return []

    path = ESPN_PATHS.get(sport, '')
    url  = f'{ESPN_API_BASE}/{path}/teams/{team_id}/depthchart'

    try:
        resp = _http_get(url, timeout=20)
        req_counter['espn_depth'] = req_counter.get('espn_depth', 0) + 1
        time.sleep(1.0)

        if not resp:
            return []

        players = []
        for position_group in (resp.get('positionGroups') or []):
            pos_abbr = (position_group.get('abbreviation') or '').upper()
            for position in (position_group.get('positions') or []):
                pos_name = (position.get('abbreviation') or pos_abbr).upper()
                for athlete in (position.get('athletes') or []):
                    athlete_obj = athlete.get('athlete') or {}
                    player_id   = str(athlete_obj.get('id', ''))
                    player_name = (athlete_obj.get('displayName') or '').strip()
                    rank        = int(athlete.get('rank', 99))
                    status_obj  = athlete_obj.get('status') or {}
                    status      = (status_obj.get('type') or {}).get('name', 'active')

                    if not player_id or not player_name:
                        continue

                    players.append({
                        'player_id':   player_id,
                        'name':        player_name,
                        'position':    pos_name,
                        'depth_rank':  rank,
                        'team_id':     team_id,
                        'sport':       sport,
                        'status':      status.lower(),
                    })

        return players

    except Exception as exc:
        log.debug(f'[scout] Depth chart error {sport}/{team_id}: {exc}')
        return []


# ============================================================
# SECTION 4: PLAYER HISTORICAL STATS FETCHER
# ============================================================

def fetch_player_gamelog(sport: str, player_id: str, req_counter: dict) -> List[dict]:
    """
    Fetches a player's recent game log from ESPN.
    Returns list of game entries.
    """
    if req_counter.get('espn_player', 0) >= RATE_LIMITS['espn_player']:
        return []
    if not player_id:
        return []

    path = ESPN_PATHS.get(sport, '')
    url  = f'{ESPN_API_BASE}/{path}/athletes/{player_id}/gamelog'

    try:
        resp = _http_get(url, timeout=20)
        req_counter['espn_player'] = req_counter.get('espn_player', 0) + 1
        time.sleep(0.5)

        if not resp:
            return []

        entries = []
        # ESPN gamelog varies by sport — extract what's available
        for season in (resp.get('seasonTypes') or []):
            for category in (season.get('categories') or []):
                for event in (category.get('events') or []):
                    event_id = str(event.get('eventId', ''))
                    stats    = event.get('stats') or []
                    if event_id:
                        entries.append({
                            'event_id': event_id,
                            'stats':    stats,
                        })

        return entries

    except Exception as exc:
        log.debug(f'[scout] Player gamelog error {sport}/{player_id}: {exc}')
        return []


def fetch_team_roster(sport: str, team_id: str, req_counter: dict) -> List[dict]:
    """
    Fetches the current roster for a team.
    Returns list of player dicts.
    """
    if not team_id:
        return []

    path = ESPN_PATHS.get(sport, '')
    url  = f'{ESPN_API_BASE}/{path}/teams/{team_id}/roster'

    try:
        resp = _http_get(url, timeout=20)
        time.sleep(0.5)

        if not resp:
            return []

        players = []
        for athlete in (resp.get('athletes') or []):
            for player_obj in (athlete.get('items') or [athlete]):
                pid  = str(player_obj.get('id', ''))
                name = (player_obj.get('displayName') or '').strip()
                pos  = ((player_obj.get('position') or {}).get('abbreviation') or '').upper()
                if pid and name:
                    players.append({'player_id': pid, 'name': name, 'position': pos,
                                    'team_id': team_id, 'sport': sport})
        return players

    except Exception as exc:
        log.debug(f'[scout] Roster error {sport}/{team_id}: {exc}')
        return []


# ============================================================
# SECTION 5: PLAYER WIN IMPACT (PWI) CALCULATOR
# ============================================================

def compute_pwi_from_history(team: str, player_name: str,
                              sport: str, team_history: dict) -> Tuple[float, float, int]:
    """
    Computes Player Win Impact from team_history.
    Compares team win rate in games where player was absent vs. present.

    Requires player absence data in game entries (field: 'missing_players').
    Returns (pwi, confidence, n_games_without).

    If insufficient data, returns (positional_fallback, 0.0, 0).
    """
    sport_history = team_history.get(sport, {})
    if not isinstance(sport_history, dict):
        return 0.02, 0.0, 0

    team_games = sport_history.get(team, [])
    if not isinstance(team_games, list) or len(team_games) < MIN_GAMES_FOR_PWI:
        return 0.02, 0.0, 0

    games_with    = [g for g in team_games if player_name not in (g.get('missing_players') or [])]
    games_without = [g for g in team_games if player_name in (g.get('missing_players') or [])]

    if len(games_without) < MIN_GAMES_FOR_PWI or len(games_with) < MIN_GAMES_FOR_PWI:
        return 0.02, 0.0, len(games_without)

    wr_with    = sum(1 for g in games_with    if g.get('win')) / len(games_with)
    wr_without = sum(1 for g in games_without if g.get('win')) / len(games_without)

    raw_pwi = wr_with - wr_without

    # Confidence via Wilson score
    n = len(games_without)
    if n >= 10:
        # Simple confidence: more games = more confident
        confidence = min(0.95, 0.50 + n * 0.02)
    else:
        confidence = 0.0

    # Cap PWI at reasonable bounds
    pwi = max(-0.30, min(0.30, raw_pwi))

    return float(pwi), float(confidence), n


def get_positional_pwi(position: str, sport: str, is_star: bool = False) -> float:
    """Returns positional fallback PWI for a player at a given position."""
    fallbacks = POSITIONAL_PWI_FALLBACK.get(sport, {})
    pos_upper = (position or '').upper()

    # Try exact match
    if pos_upper in fallbacks:
        base = fallbacks[pos_upper]
    else:
        # Try partial match (e.g., 'OLB' → 'LB')
        matched = None
        for key in fallbacks:
            if pos_upper.startswith(key) or key.startswith(pos_upper):
                matched = fallbacks[key]
                break
        base = matched or fallbacks.get('default', 0.02)

    # Star players get higher fallback
    if is_star and pos_upper in ('QB', 'PG', 'SG', 'SF', 'PF', 'C', 'G', 'SP1'):
        base = min(0.25, base * 1.5)

    return float(base)


def _is_star_player(player_data: dict) -> bool:
    """Determines if a player is a star based on available data."""
    if player_data.get('depth_rank', 99) == 1:
        return True
    # Stats-based determination if available
    stats = player_data.get('season_stats') or {}
    pts = stats.get('pts') or stats.get('points', 0)
    return float(pts or 0) >= 20.0


# ============================================================
# SECTION 6: LINEUP IMPACT MODEL
# ============================================================

def compute_lineup_adjustment(
    team: str, sport: str,
    injury_report: List[dict],
    data: dict,
) -> Tuple[float, List[str], float]:
    """
    Computes win probability adjustment for a team based on injured players.

    Parameters:
        team: team name
        sport: sport string
        injury_report: list of {name, status, position, player_id} dicts
        data: full data dict (reads scout.players and team_history)

    Returns:
        adjustment: float [-MAX_LINEUP_ADJUSTMENT, 0.0] (always ≤ 0, injuries only hurt)
        missing_players: list of player names expected to be out
        confidence: float [0, 1]
    """
    if not injury_report:
        return 0.0, [], 0.5

    total_pwi_loss = 0.0
    missing_players = []
    confidence_sum  = 0.0
    n_players       = 0

    for inj in injury_report:
        if not isinstance(inj, dict):
            continue

        player_name = (inj.get('name') or inj.get('displayName') or '').strip()
        status_raw  = (inj.get('status') or inj.get('type') or '').lower()
        position    = (inj.get('position') or inj.get('pos') or '').upper()
        player_id   = str(inj.get('player_id') or inj.get('id') or '')

        if not player_name:
            continue

        # Availability probability from designation
        availability = _get_availability_prob(status_raw, data, sport, player_name, player_id)

        # Only count players below 80% availability
        if availability >= 0.80:
            continue

        # PWI from scout data or team history
        pwi, confidence, n_without = _get_pwi(
            player_name, player_id, team, sport, position, data
        )

        # Expected contribution loss
        full_contribution     = pwi
        expected_contribution = availability * pwi
        pwi_loss              = full_contribution - expected_contribution

        if pwi_loss > 0.001:
            total_pwi_loss += pwi_loss
            confidence_sum += confidence
            n_players      += 1

            if availability < 0.50:
                missing_players.append(player_name)

    # Cap total adjustment
    adjustment = -min(MAX_LINEUP_ADJUSTMENT, total_pwi_loss)
    avg_confidence = (confidence_sum / n_players) if n_players > 0 else 0.5

    return float(adjustment), missing_players, float(avg_confidence)


def _get_availability_prob(status_raw: str, data: dict,
                             sport: str, player_name: str,
                             player_id: str) -> float:
    """
    Returns availability probability [0,1] for a player.
    Prefers edge.py's linguistic injury score if available,
    falls back to designation mapping.
    """
    # Check edge.py sentiment for more granular language score
    edge_data = data.get('edge', {})
    inj_texts = edge_data.get('cache', {}).get(f'espn_inj_{sport}', {}).get('items', [])

    if inj_texts:
        player_texts = [t for t in inj_texts
                        if player_name.split()[-1].lower() in t.lower()]
        if player_texts:
            # Use edge.py's classifier
            try:
                from edge import parse_injury_designation
                combined_text = ' '.join(player_texts[:3])
                return parse_injury_designation(combined_text)
            except ImportError:
                pass

    # Fallback: designation string mapping
    for key, prob in sorted(DESIGNATION_PROB.items(), key=lambda x: -len(x[0])):
        if key in status_raw:
            return prob

    # Unknown status — slight uncertainty
    return 0.75


def _get_pwi(player_name: str, player_id: str, team: str,
              sport: str, position: str, data: dict) -> Tuple[float, float, int]:
    """
    Gets PWI for a player. Checks scout data first, then computes from history,
    then falls back to positional estimate.
    """
    # Check scout players store
    player_store = data.get('scout', {}).get('players', {})
    stored = player_store.get(player_id) or player_store.get(player_name)

    if stored and isinstance(stored, dict):
        pwi        = float(stored.get('pwi', 0.02))
        confidence = float(stored.get('pwi_confidence', 0.5))
        n          = int(stored.get('pwi_sample_size', 0))
        if n >= MIN_GAMES_FOR_PWI:
            return pwi, confidence, n

    # Compute from team_history
    pwi, confidence, n = compute_pwi_from_history(
        team, player_name, sport, data.get('team_history', {})
    )

    if n >= MIN_GAMES_FOR_PWI:
        return pwi, confidence, n

    # Positional fallback
    is_star = stored.get('is_star', False) if stored else False
    fallback_pwi = get_positional_pwi(position, sport, is_star)
    return fallback_pwi, 0.30, n


# ============================================================
# SECTION 7: MATCHUP-SPECIFIC PLAYER PERFORMANCE
# ============================================================

def get_player_matchup_factor(player_id: str, opponent_team: str,
                               sport: str, data: dict) -> float:
    """
    Returns matchup performance factor for a player vs a specific opponent.
    Factor > 1.0 = player performs better vs this opponent than average.
    Returns 1.0 (neutral) if insufficient data.
    """
    player_store = data.get('scout', {}).get('players', {})
    player_data  = player_store.get(str(player_id))

    if not player_data or not isinstance(player_data, dict):
        return 1.0

    matchup = (player_data.get('matchup_history') or {}).get(opponent_team.lower())
    if not matchup or not isinstance(matchup, dict):
        return 1.0

    n = int(matchup.get('games', 0))
    if n < 3:
        return 1.0

    factor = float(matchup.get('performance_factor', 1.0))
    # Cap at ±15% to prevent outliers from dominating
    return max(0.85, min(1.15, factor))


def update_player_matchup(player_id: str, player_name: str,
                           opponent_team: str, sport: str,
                           performance_ratio: float, data: dict) -> None:
    """
    Updates player matchup history with a new game result.
    performance_ratio: actual_stat / season_avg_stat for the game.
    """
    ensure_scout_keys(data)
    players = data['scout']['players']

    pid = str(player_id)
    if pid not in players:
        players[pid] = {
            'name': player_name, 'sport': sport,
            'matchup_history': {},
        }

    opponent_lower = opponent_team.lower()
    mh = players[pid].setdefault('matchup_history', {})

    if opponent_lower not in mh:
        mh[opponent_lower] = {'games': 0, 'performance_sum': 0.0, 'performance_factor': 1.0}

    entry = mh[opponent_lower]
    entry['games']           += 1
    entry['performance_sum'] += performance_ratio
    entry['performance_factor'] = entry['performance_sum'] / entry['games']


# ============================================================
# SECTION 8: PLAYER AVAILABILITY ADJUSTER
# ============================================================

def get_team_injury_report(sport: str, team_id: str, data: dict) -> List[dict]:
    """
    Gets injury report for a team from brain.py's already-fetched data.
    Falls back to ESPN injuries API if not in data.
    """
    # brain.py stores injuries in predictions or we can fetch from ESPN
    # Check if edge.py fetched injury text
    edge_inj = data.get('edge', {}).get('cache', {}).get(f'espn_inj_{sport}', {})
    if edge_inj:
        # Already have text — use it via section 8's parse mechanism
        return []  # Handled in compute_lineup_adjustment via text parsing

    # Try fetching from ESPN injuries API
    path = ESPN_PATHS.get(sport, '')
    if not path or not team_id:
        return []

    try:
        url  = f'https://www.espn.com/{path.replace("/", ".")}/team/injuries/_/id/{team_id}'
        # Use the ESPN hidden API instead
        url  = f'{ESPN_API_BASE}/{path}/teams/{team_id}/injuries'
        resp = _http_get(url, timeout=20)
        time.sleep(0.5)

        if not resp:
            return []

        injuries = []
        for item in (resp.get('injuries') or []):
            athlete   = item.get('athlete') or {}
            status    = item.get('status') or {}
            position  = (athlete.get('position') or {}).get('abbreviation', '')
            injuries.append({
                'player_id': str(athlete.get('id', '')),
                'name':      (athlete.get('displayName') or '').strip(),
                'position':  position,
                'status':    (status.get('type') or {}).get('name', '').lower(),
                'comment':   item.get('longComment', ''),
            })
        return injuries

    except Exception as exc:
        log.debug(f'[scout] Injury report error {sport}/{team_id}: {exc}')
        return []


# ============================================================
# SECTION 9: REFEREE/UMPIRE ROSTER FETCHER
# ============================================================

def fetch_game_officials(sport: str, game_id: str, req_counter: dict) -> List[dict]:
    """
    Fetches the officiating crew for a specific game from ESPN.
    Returns list of {name, id, position} dicts.
    """
    if req_counter.get('espn_game', 0) >= RATE_LIMITS['espn_game']:
        return []
    if not game_id:
        return []

    path = ESPN_PATHS.get(sport, '')
    url  = f'{ESPN_API_BASE}/{path}/summary'

    try:
        resp = _http_get(url, params={'event': str(game_id)}, timeout=20)
        req_counter['espn_game'] = req_counter.get('espn_game', 0) + 1
        time.sleep(0.5)

        if not resp:
            return []

        officials = []
        for official in (resp.get('officials') or []):
            if not isinstance(official, dict):
                continue
            officials.append({
                'id':       str(official.get('id', '')),
                'name':     (official.get('displayName') or official.get('fullName') or '').strip(),
                'position': (official.get('position') or {}).get('displayName', ''),
            })

        return officials

    except Exception as exc:
        log.debug(f'[scout] Officials fetch error {sport}/{game_id}: {exc}')
        return []


def fetch_mlb_game_officials(game_date: str, req_counter: dict) -> Dict[str, List[dict]]:
    """
    Fetches MLB umpire assignments for a date using MLB Stats API.
    Returns dict: {game_id: [umpires]}
    """
    if req_counter.get('mlb_officials', 0) >= RATE_LIMITS['mlb_officials']:
        return {}

    url = f'{MLB_STATS_BASE}/schedule'
    params = {
        'sportId': '1',
        'date':    game_date[:10],
        'hydrate': 'officials',
    }

    try:
        resp = _http_get(url, params=params, timeout=20)
        req_counter['mlb_officials'] = req_counter.get('mlb_officials', 0) + 1
        time.sleep(0.5)

        if not resp:
            return {}

        result = {}
        for date_entry in (resp.get('dates') or []):
            for game in (date_entry.get('games') or []):
                game_id  = str(game.get('gamePk', ''))
                officials = []
                for official in (game.get('officials') or []):
                    otype = (official.get('officialType') or '').lower()
                    person = official.get('official') or {}
                    if 'home plate' in otype or 'plate' in otype or not otype:
                        officials.append({
                            'id':       str(person.get('id', '')),
                            'name':     (person.get('fullName') or '').strip(),
                            'position': otype or 'umpire',
                        })
                if game_id and officials:
                    result[game_id] = officials
        return result

    except Exception as exc:
        log.debug(f'[scout] MLB officials error: {exc}')
        return {}


# ============================================================
# SECTION 10: REFEREE TENDENCY PROFILER
# ============================================================

def build_referee_profile_from_history(official_id: str, official_name: str,
                                        sport: str, data: dict) -> dict:
    """
    Builds or updates a referee tendency profile.
    Uses completed games from data where this official was assigned.
    Returns profile dict.
    """
    existing = data.get('scout', {}).get('officials', {}).get(sport, {}).get(official_id, {})

    if existing and existing.get('games_officiated', 0) > 0:
        return existing  # Use existing profile — updated incrementally

    # Default profile when no history exists
    default_profile = {
        'ref_id':   official_id,
        'name':     official_name,
        'sport':    sport,
        'games_officiated': 0,
        'last_updated': None,
    }

    if sport == 'nfl':
        default_profile.update({
            'avg_penalties_per_game':       13.8,
            'home_team_penalty_rate':        0.50,
            'avg_total_yards_per_game':      6.8,
            'false_start_rate':              1.5,
            'holding_rate':                  1.9,
            'pass_interference_rate':        0.5,
            'home_win_rate_when_officiating': 0.57,
            'close_game_home_bias':          0.0,
        })
    elif sport == 'nba':
        default_profile.update({
            'avg_fouls_per_48':        42.0,
            'home_foul_advantage':      2.0,
            'star_player_foul_rate':    0.90,
            'pace_factor':              1.00,
            'technical_rate':           0.3,
            'home_win_rate_when_officiating': 0.57,
        })
    elif sport == 'mlb':
        default_profile.update({
            'zone_size_factor':       1.00,
            'called_strike_rate':     0.315,
            'walk_rate_factor':       1.00,
            'k_rate_factor':          1.00,
            'home_called_strike_rate': 0.315,
            'away_called_strike_rate': 0.315,
            'favor_righties':         False,
            'righty_strike_advantage': 0.0,
        })
    elif sport == 'nhl':
        default_profile.update({
            'avg_penalties_per_game':   6.0,
            'home_team_penalty_rate':   0.49,
            'powerplay_calls_per_game': 5.8,
            'home_win_rate_when_officiating': 0.55,
        })

    return default_profile


def update_referee_profile_from_game(official_id: str, sport: str,
                                      game_result: dict, data: dict) -> None:
    """
    Incrementally updates a referee's profile with a completed game's statistics.
    game_result: dict with penalties, home_won, etc.
    """
    ensure_scout_keys(data)
    officials = data['scout']['officials'].setdefault(sport, {})
    profile   = officials.get(official_id, {})

    if not profile:
        return

    n     = profile.get('games_officiated', 0)
    alpha = 1.0 / (n + 1)  # Exponential moving average weight

    if sport == 'nfl':
        if 'penalties' in game_result:
            old = profile.get('avg_penalties_per_game', 13.8)
            profile['avg_penalties_per_game'] = old * (1 - alpha) + game_result['penalties'] * alpha

        if 'home_penalties' in game_result and 'total_penalties' in game_result:
            total = game_result.get('total_penalties', 1)
            if total > 0:
                rate = game_result['home_penalties'] / total
                old  = profile.get('home_team_penalty_rate', 0.50)
                profile['home_team_penalty_rate'] = old * (1 - alpha) + rate * alpha

    elif sport == 'nba':
        if 'total_fouls' in game_result:
            old = profile.get('avg_fouls_per_48', 42.0)
            profile['avg_fouls_per_48'] = old * (1 - alpha) + game_result['total_fouls'] * alpha

    profile['games_officiated'] = n + 1
    profile['last_updated'] = datetime.datetime.utcnow().isoformat()
    officials[official_id] = profile


# ============================================================
# SECTION 11: UMPIRE STRIKE ZONE PROFILER (MLB)
# ============================================================

def build_umpire_profile(umpire_id: str, umpire_name: str, data: dict) -> dict:
    """
    Returns MLB umpire profile with zone tendency data.
    Uses cached profile or returns default.
    """
    officials = data.get('scout', {}).get('officials', {}).get('mlb', {})
    existing  = officials.get(umpire_id, {})

    if existing and existing.get('games_officiated', 0) >= 5:
        return existing

    return {
        'ump_id':                  umpire_id,
        'name':                    umpire_name,
        'sport':                   'mlb',
        'games_officiated':        0,
        'zone_size_factor':        1.00,
        'called_strike_rate':      0.315,
        'walk_rate_factor':        1.00,
        'k_rate_factor':           1.00,
        'home_called_strike_rate': 0.315,
        'away_called_strike_rate': 0.315,
        'favor_righties':          False,
        'righty_strike_advantage': 0.0,
        'last_updated':            None,
    }


# ============================================================
# SECTION 12: OFFICIAL IMPACT CALCULATOR
# ============================================================

def compute_official_impact(
    officials: List[dict], sport: str,
    home_team: str, away_team: str,
    data: dict
) -> dict:
    """
    Computes win probability impact of the officiating crew for a game.
    Returns dict with impact (float), impact_direction, notes.
    """
    result = {
        'officials':       [o.get('name', '') for o in officials],
        'impact':          0.0,
        'impact_direction': 'neutral',
        'notes':           [],
    }

    if not officials or not sport:
        return result

    scout_officials = data.get('scout', {}).get('officials', {}).get(sport, {})
    total_impact    = 0.0
    n_officials     = 0

    for official in officials:
        oid      = str(official.get('id', ''))
        oname    = official.get('name', '')
        profile  = scout_officials.get(oid) or build_referee_profile_from_history(
            oid, oname, sport, data
        )

        if not profile or profile.get('games_officiated', 0) < 5:
            continue  # Not enough data for this official

        impact = 0.0

        if sport == 'nfl':
            pen_rate = profile.get('home_team_penalty_rate', 0.50)
            if pen_rate < 0.46:
                impact -= 0.015  # Ref penalizes home team more than average
            elif pen_rate > 0.54:
                impact += 0.015  # Ref penalizes away team more

            if abs(impact) > 0.005:
                direction = 'home' if impact > 0 else 'away'
                result['notes'].append(
                    f'{oname}: {pen_rate:.0%} home penalties (avg 50%)'
                )

        elif sport == 'nba':
            home_foul_adv = profile.get('home_foul_advantage', 2.0)
            home_style = (data.get('scout', {}).get('team_style', {})
                          .get(home_team.lower(), {}).get('free_throw_rate', 'average'))
            away_style = (data.get('scout', {}).get('team_style', {})
                          .get(away_team.lower(), {}).get('free_throw_rate', 'average'))

            avg_fouls = profile.get('avg_fouls_per_48', 42.0)
            if avg_fouls > 45 and home_style == 'high':
                impact += 0.02
            elif avg_fouls < 40 and home_style == 'high':
                impact -= 0.02

            if abs(home_foul_adv - 2.0) > 0.5:
                impact += (home_foul_adv - 2.0) * 0.003

        elif sport == 'mlb':
            zone_factor = profile.get('zone_size_factor', 1.00)
            if abs(zone_factor - 1.0) > 0.03:
                # Tight zone hurts pitching-dependent teams more
                home_era = (data.get('scout', {}).get('team_stats_cache', {})
                            .get(home_team.lower(), {}).get('era', 4.0))
                away_era = (data.get('scout', {}).get('team_stats_cache', {})
                            .get(away_team.lower(), {}).get('era', 4.0))
                era_diff = home_era - away_era
                zone_impact = era_diff * 0.003 * (1 - zone_factor) * 5
                impact += zone_impact

        elif sport == 'nhl':
            pen_rate = profile.get('home_team_penalty_rate', 0.49)
            if pen_rate < 0.45:
                impact -= 0.012
            elif pen_rate > 0.53:
                impact += 0.012

        total_impact  += impact
        n_officials   += 1

    if n_officials == 0:
        return result

    avg_impact = total_impact / n_officials
    # Hard cap
    capped_impact = max(-MAX_OFFICIAL_IMPACT, min(MAX_OFFICIAL_IMPACT, avg_impact))

    result['impact']           = round(capped_impact, 4)
    result['impact_direction'] = 'home' if capped_impact > 0.005 else (
        'away' if capped_impact < -0.005 else 'neutral'
    )

    return result


# ============================================================
# SECTION 13: COMBINED SCOUT SIGNAL EXPORT
# ============================================================

def build_team_style(team: str, sport: str, team_stats: dict) -> dict:
    """
    Builds a team style profile from available stats.
    Used by official impact calculator.
    """
    stats = (team_stats.get(sport) or {}).get(team.lower(), {})
    style = {
        'pace':            'medium',
        'three_point_heavy': False,
        'free_throw_rate': 'average',
        'defensive_style': 'standard',
    }

    if not stats or not isinstance(stats, dict):
        return style

    # NBA-specific style detection
    if sport in ('nba', 'ncaabm', 'ncaabw'):
        pts = float(stats.get('pts') or 0)
        if pts > 115:
            style['pace'] = 'fast'
        elif pts < 105:
            style['pace'] = 'slow'

        ft_pct = float(stats.get('ft_pct') or 0)
        if ft_pct > 0.78:
            style['free_throw_rate'] = 'high'
        elif ft_pct < 0.70:
            style['free_throw_rate'] = 'low'

    return style


def export_lineup_adjustment(data: dict, game_id: str,
                              home_adj: float, away_adj: float,
                              home_missing: List[str], away_missing: List[str],
                              home_pwi_loss: float, away_pwi_loss: float,
                              confidence: float) -> None:
    """Writes lineup adjustment to data['scout']['lineup_adjustments']."""
    ensure_scout_keys(data)
    data['scout']['lineup_adjustments'][str(game_id)] = {
        'home_adjustment':  round(home_adj, 4),
        'away_adjustment':  round(away_adj, 4),
        'home_missing_players': home_missing,
        'away_missing_players': away_missing,
        'home_pwi_loss':    round(home_pwi_loss, 4),
        'away_pwi_loss':    round(away_pwi_loss, 4),
        'confidence':       round(confidence, 4),
        'updated_at':       datetime.datetime.utcnow().isoformat(),
    }


def export_official_assignment(data: dict, game_id: str, impact_dict: dict) -> None:
    """Writes official assignment and impact to data['scout']['official_assignments']."""
    ensure_scout_keys(data)
    data['scout']['official_assignments'][str(game_id)] = impact_dict


def _cleanup_old_signals(data: dict, days_back: int = 7) -> None:
    """Removes scout signals older than days_back days."""
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')

    for section in ('lineup_adjustments', 'official_assignments'):
        store = data['scout'].get(section, {})
        old_keys = [
            k for k, v in list(store.items())
            if isinstance(v, dict) and v.get('updated_at', '9999')[:10] < cutoff
        ]
        for k in old_keys:
            store.pop(k, None)


# ============================================================
# SECTION 14: TRAINING ENTRY POINT
# ============================================================

def train_scout(data: dict) -> None:
    """
    Full scout.py training run.
    1. Builds player profiles from team_history
    2. Builds referee profiles from historical ESPN data
    3. Builds team style profiles
    4. Saves everything to data['scout']

    Called by brain.py --mode train after mind.py training completes.
    """
    ensure_scout_keys(data)
    log.info('[scout] Starting training...')

    req_counter: Dict[str, int] = {}

    # Build team style profiles from team_stats
    log.info('[scout] Building team style profiles...')
    team_stats = data.get('team_stats', {})
    for sport in ALL_SPORTS:
        sport_stats = team_stats.get(sport, {})
        if not isinstance(sport_stats, dict):
            continue
        for team_name in sport_stats:
            style = build_team_style(team_name, sport, team_stats)
            data['scout']['team_style'][team_name.lower()] = style

    # Build PWI for players with sufficient history
    log.info('[scout] Computing Player Win Impact from team_history...')
    team_history = data.get('team_history', {})
    n_players_profiled = 0

    for sport in ALL_SPORTS:
        sport_history = team_history.get(sport, {})
        if not isinstance(sport_history, dict):
            continue

        for team_name, games in sport_history.items():
            if not isinstance(games, list) or len(games) < MIN_GAMES_FOR_PWI:
                continue

            # Find all players who appear in 'missing_players' field
            all_missing = set()
            for game in games:
                for missing in (game.get('missing_players') or []):
                    all_missing.add(missing)

            for player_name in all_missing:
                pwi, confidence, n = compute_pwi_from_history(
                    team_name, player_name, sport, team_history
                )
                if n >= MIN_GAMES_FOR_PWI:
                    pid = f'{sport}_{team_name.lower()}_{player_name.lower()}'
                    data['scout']['players'][pid] = {
                        'name':            player_name,
                        'team':            team_name,
                        'sport':           sport,
                        'pwi':             pwi,
                        'pwi_confidence':  confidence,
                        'pwi_sample_size': n,
                        'computed_from':   'team_history',
                    }
                    n_players_profiled += 1

    log.info(f'[scout] Profiled {n_players_profiled} players from history')

    data['scout']['last_updated'] = datetime.datetime.utcnow().isoformat()
    log.info('[scout] Training complete.')


# ============================================================
# SECTION 15: UPDATE ENTRY POINT
# ============================================================

def update_scout(data: dict) -> None:
    """
    Daily update run for scout.py.
    1. Fetches officiating assignments for upcoming games
    2. Computes lineup adjustments from current injury reports
    3. Updates player profiles incrementally
    4. Writes lineup_adjustments and official_assignments to data['scout']

    Called by brain.py --mode update after edge.py runs.
    Never raises — all errors logged.
    """
    ensure_scout_keys(data)
    log.info('[scout] Starting update...')

    req_counter: Dict[str, int] = {}
    error_log = data['scout'].get('error_log', [])

    today = datetime.datetime.utcnow().strftime('%Y-%m-%d')

    # Get upcoming games from brain.py's predictions
    upcoming_games: List[dict] = []
    for sport in ALL_SPORTS:
        preds = data.get('predictions', {}).get(sport, [])
        for p in preds:
            if isinstance(p, dict) and p.get('status') == 'pending':
                upcoming_games.append({
                    'id':      p.get('id', ''),
                    'sport':   sport,
                    'home':    p.get('home', ''),
                    'away':    p.get('away', ''),
                    'home_id': p.get('home_id', ''),
                    'away_id': p.get('away_id', ''),
                    'date':    p.get('date', ''),
                    'venue':   p.get('venue', ''),
                })

    if not upcoming_games:
        log.info('[scout] No upcoming games found — skipping update')
        return

    # Fetch MLB umpire assignments if MLB games exist
    mlb_games = [g for g in upcoming_games if g['sport'] == 'mlb']
    mlb_officials_by_game: Dict[str, List[dict]] = {}

    if mlb_games:
        try:
            mlb_officials_by_game = fetch_mlb_game_officials(today, req_counter)
        except Exception as exc:
            log.debug(f'[scout] MLB officials error: {exc}')

    # Process each upcoming game
    for game in upcoming_games:
        sport    = game['sport']
        game_id  = game.get('id') or f"{sport}_{game['home'].lower()}_{game['away'].lower()}_{game['date'][:10]}"
        home     = game.get('home', '')
        away     = game.get('away', '')
        home_id  = game.get('home_id', '')
        away_id  = game.get('away_id', '')

        try:
            # --- Fetch officials for this game ---
            officials = []

            if sport == 'mlb' and game_id in mlb_officials_by_game:
                officials = mlb_officials_by_game[game_id]
            elif sport in ('nfl', 'nba', 'nhl'):
                espn_id = _extract_espn_game_id(game_id)
                if espn_id:
                    officials = fetch_game_officials(sport, espn_id, req_counter)

            # Build/fetch official profiles for new officials
            if officials:
                for official in officials:
                    oid   = str(official.get('id', ''))
                    oname = official.get('name', '')
                    if oid and oid not in data['scout']['officials'].get(sport, {}):
                        profile = build_referee_profile_from_history(oid, oname, sport, data)
                        data['scout']['officials'].setdefault(sport, {})[oid] = profile

                # Compute official impact
                impact_dict = compute_official_impact(officials, sport, home, away, data)
                export_official_assignment(data, game_id, impact_dict)

        except Exception as exc:
            msg = f'[scout] Officials error for {home} vs {away}: {exc}'
            log.debug(msg)
            error_log.append({'time': datetime.datetime.utcnow().isoformat(), 'error': msg})

        try:
            # --- Compute lineup adjustments ---
            home_injuries = get_team_injury_report(sport, home_id, data)
            away_injuries = get_team_injury_report(sport, away_id, data)

            # Also read from brain.py's injury data if populated
            brain_injuries_home = data.get('injuries', {}).get(sport, {}).get(home, [])
            brain_injuries_away = data.get('injuries', {}).get(sport, {}).get(away, [])

            combined_home_inj = (home_injuries or brain_injuries_home or [])
            combined_away_inj = (away_injuries or brain_injuries_away or [])

            home_adj, home_missing, home_conf = compute_lineup_adjustment(
                home, sport, combined_home_inj, data
            )
            away_adj, away_missing, away_conf = compute_lineup_adjustment(
                away, sport, combined_away_inj, data
            )

            avg_conf = (home_conf + away_conf) / 2

            export_lineup_adjustment(
                data, game_id,
                home_adj, away_adj,
                home_missing, away_missing,
                abs(home_adj), abs(away_adj),
                avg_conf,
            )

        except Exception as exc:
            msg = f'[scout] Lineup adjustment error for {home} vs {away}: {exc}'
            log.debug(msg)
            error_log.append({'time': datetime.datetime.utcnow().isoformat(), 'error': msg})

    # Update team style profiles if stats have been refreshed
    team_stats = data.get('team_stats', {})
    for sport in ALL_SPORTS:
        sport_stats = team_stats.get(sport, {})
        if isinstance(sport_stats, dict):
            for team_name in sport_stats:
                style = build_team_style(team_name, sport, team_stats)
                data['scout']['team_style'][team_name.lower()] = style

    # Cleanup and finalize
    _cleanup_old_signals(data)
    data['scout']['error_log']   = error_log[-200:]
    data['scout']['last_updated'] = datetime.datetime.utcnow().isoformat()

    n_lineup   = len(data['scout']['lineup_adjustments'])
    n_officials = len(data['scout']['official_assignments'])
    log.info(f'[scout] Complete. Lineup adjustments: {n_lineup}, Official assignments: {n_officials}')


def _extract_espn_game_id(game_id: str) -> str:
    """
    Attempts to extract the ESPN numeric game ID from a composite game_id string.
    brain.py game IDs are sometimes the ESPN ID directly, or 'sport_home_away_date'.
    Returns numeric portion if found, else empty string.
    """
    # If the game_id is purely numeric, it's already an ESPN ID
    if game_id.isdigit():
        return game_id

    # Try to extract a numeric portion from a composite string
    # Format might be: 'nfl_401547417_...' or similar
    match = re.search(r'(\d{8,12})', game_id)
    if match:
        return match.group(1)

    return ''


# ============================================================
# SECTION 16: RATE LIMITING AND ERROR HANDLING
# ============================================================

BACKOFF_DELAYS = [2, 5, 15]


def _http_get(url: str, params: Optional[Dict] = None,
              headers: Optional[Dict] = None, timeout: int = 30) -> Optional[Any]:
    """GET request returning parsed JSON or None. Retry with backoff."""
    for attempt in range(3):
        try:
            resp = requests.get(
                url,
                params=params or {},
                headers=headers or {'User-Agent': 'edge-predict/1.0'},
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
