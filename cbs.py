#!/usr/bin/env python3
"""
EDGE Predict — cbs.py
CBS Confidence Pool optimizer using Integer Linear Programming.
Replaces brain.py's Monte Carlo + Hungarian approach with PuLP ILP.
Never raises — every path has a silent fallback to greedy sort.

Section 1  — Imports and constants
Section 2  — data.json keys managed by cbs.py
Section 3  — Game probability aggregator
Section 4  — Probability confidence scorer
Section 5  — Integer Linear Programming optimizer (PuLP)
Section 6  — Uncertainty penalty model
Section 7  — Game variance classifier
Section 8  — Season phase accuracy adjuster
Section 9  — Upset leverage detector (accuracy-based)
Section 10 — Historical accuracy validator
Section 11 — Output formatter and data.json writer
Section 12 — Entry point (called by brain.py --mode cbs)
"""

# ============================================================
# SECTION 1: IMPORTS AND CONSTANTS
# ============================================================
import os
import re
import json
import math
import time
import logging
import datetime
import traceback
from typing import Optional, Dict, List, Tuple, Any

import requests
import numpy as np

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

log = logging.getLogger(__name__)

ESPN_BASE    = 'https://site.api.espn.com/apis/site/v2/sports'
ESPN_NFL     = 'football/nfl'
REQUEST_TIMEOUT = 15
MAX_RETRIES     = 2

# Season phase definitions for NFL weeks 1-18
SEASON_PHASE_ACCURACY: Dict[str, dict] = {
    'early':        {'weeks': range(1, 5),   'accuracy_modifier': -0.04},
    'mid':          {'weeks': range(5, 11),  'accuracy_modifier':  0.00},
    'late':         {'weeks': range(11, 15), 'accuracy_modifier': +0.02},
    'playoff_push': {'weeks': range(15, 19), 'accuracy_modifier': -0.02},
}

# Variance confidence modifier by game total / spread profile
VARIANCE_CONFIDENCE_MODIFIER: Dict[str, float] = {
    'very_low': +0.12,  # Expected blowout — probability very reliable
    'low':      +0.05,
    'medium':    0.00,
    'high':     -0.08,  # High-scoring close game — less predictable
    'unknown':   0.00,
}


# ============================================================
# SECTION 2: DATA.JSON KEYS MANAGED BY CBS.PY
# ============================================================
CBS_PICKS_TEMPLATE = {
    'generated_at':        None,
    'week_number':         None,
    'num_games':           0,
    'max_possible_points': 0,
    'expected_points':     0,
    'season_phase':        'mid',
    'accuracy_modifier':   0.0,
    'assignment':          [],
    'historical_validation': {
        'ilp_weekly_avg':      None,
        'greedy_weekly_avg':   None,
        'ilp_improvement_pct': None,
        'weeks_analyzed':      0,
    },
}


# ============================================================
# SECTION 3: GAME PROBABILITY AGGREGATOR
# ============================================================
def _safe_fetch(url: str, params: Optional[dict] = None) -> Optional[dict]:
    """Minimal fetch helper — independent of brain.py."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            time.sleep(1.5 ** attempt)
        except Exception as exc:
            log.warning(f'cbs fetch error ({url[:60]}): {exc}')
    return None


def _fetch_nfl_scoreboard() -> List[dict]:
    """Fetch current NFL week games from ESPN scoreboard."""
    data = _safe_fetch(f'{ESPN_BASE}/{ESPN_NFL}/scoreboard')
    if not data:
        return []

    games = []
    for event in (data.get('events') or []):
        if not isinstance(event, dict):
            continue
        for comp in (event.get('competitions') or []):
            competitors = comp.get('competitors') or []
            if len(competitors) < 2:
                continue

            home = next((c for c in competitors if c.get('homeAway') == 'home'), competitors[0])
            away = next((c for c in competitors if c.get('homeAway') == 'away'), competitors[1])

            status_type = ((comp.get('status') or {}).get('type') or {})
            status_name = status_type.get('name', '').lower()
            status_desc = status_type.get('description', '').lower()
            is_locked   = any(kw in status_name + status_desc for kw in (
                'final', 'in progress', 'halftime', 'end', 'post', 'inprogress'
            ))

            odds_block = next(iter(comp.get('odds') or [{}]), {})
            details    = odds_block.get('details', '')  # e.g. "KC -7.5" or "NE +3"
            over_under = _safe_float(odds_block.get('overUnder'))
            spread     = _extract_spread(details)

            games.append({
                'id':         str(event.get('id', '')),
                'home':       (home.get('team') or {}).get('displayName', ''),
                'away':       (away.get('team') or {}).get('displayName', ''),
                'home_id':    str((home.get('team') or {}).get('id', '')),
                'away_id':    str((away.get('team') or {}).get('id', '')),
                'date':       comp.get('date', ''),
                'is_locked':  is_locked,
                'spread':     spread,
                'over_under': over_under,
            })
    return games


def _extract_spread(details: str) -> Optional[float]:
    """Parse spread value from odds detail string like 'KC -7.5'."""
    match = re.search(r'([+-]\d+\.?\d*)', details)
    if match:
        try:
            return abs(float(match.group(1)))
        except ValueError:
            pass
    return None


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def aggregate_game_probability(pred: dict, data: dict) -> dict:
    """
    Aggregates all system signals for a single game into one probability bundle.
    pred: entry from data['predictions']['nfl']
    Returns a dict ready for confidence scoring and ILP.
    """
    game_id = str(pred.get('id', ''))

    prob     = _safe_float(pred.get('win_probability')) or 0.55
    raw_prob = _safe_float(pred.get('raw_win_probability')) or prob

    # Clamp both to safe range
    prob     = max(0.05, min(0.95, prob))
    raw_prob = max(0.05, min(0.95, raw_prob))

    confidence_lower = _safe_float(pred.get('confidence_lower'))
    confidence_upper = _safe_float(pred.get('confidence_upper'))
    confidence_width = (
        (confidence_upper - confidence_lower)
        if confidence_lower is not None and confidence_upper is not None
        else None
    )

    momentum_score       = _safe_float(pred.get('momentum_score')) or 0.5
    momentum_uncertainty = _safe_float(pred.get('momentum_uncertainty')) or 0.05

    # Sharp money from edge.py
    sharp      = (data.get('edge') or {}).get('sharp_signals', {}).get(game_id, {})
    sharp_prob_delta  = _safe_float(sharp.get('prob_delta'))  or 0.0
    sharp_confidence  = _safe_float(sharp.get('sharp_confidence')) or 0.0

    # Sentiment from edge.py
    sentiment           = (data.get('edge') or {}).get('sentiment', {}).get(game_id, {})
    sentiment_adjustment = _safe_float(sentiment.get('prob_adjustment')) or 0.0

    # Lineup adjustment from scout.py
    lineup          = (data.get('scout') or {}).get('lineup_adjustments', {}).get(game_id, {})
    lineup_adjustment = _safe_float(lineup.get('home_adjustment')) or 0.0

    # Official impact from scout.py
    official        = (data.get('scout') or {}).get('official_assignments', {}).get(game_id, {})
    official_raw    = _safe_float(official.get('impact')) or 0.0
    official_dir    = official.get('impact_direction', 'home')
    official_impact = official_raw * (1 if official_dir == 'home' else -1)

    return {
        'game_id':              game_id,
        'home_team':            pred.get('home', ''),
        'away_team':            pred.get('away', ''),
        'pick':                 pred.get('pick', pred.get('home', '')),
        'prob':                 prob,
        'raw_prob':             raw_prob,
        'confidence_lower':     confidence_lower,
        'confidence_upper':     confidence_upper,
        'confidence_width':     confidence_width,
        'momentum_score':       momentum_score,
        'momentum_uncertainty': momentum_uncertainty,
        'sharp_prob_delta':     sharp_prob_delta,
        'sharp_confidence':     sharp_confidence,
        'sentiment_adjustment': sentiment_adjustment,
        'lineup_adjustment':    lineup_adjustment,
        'official_impact':      official_impact,
        'system_agreement':     abs(prob - raw_prob),
        'is_guess':             pred.get('is_guess', False),
        'signals':              pred.get('signals', []),
        'date':                 pred.get('date', ''),
        'confidence':           pred.get('confidence', 'low'),
    }


# ============================================================
# SECTION 4: PROBABILITY CONFIDENCE SCORER
# ============================================================
def confidence_score(game_data: dict) -> float:
    """
    Scores how trustworthy a game's probability is.
    Returns a multiplier used in ILP objective weighting.
    Narrow Bayesian bounds + consistent signals = higher score.
    """
    score = 1.0

    cw = game_data.get('confidence_width')
    if cw is not None:
        if cw < 0.15:
            score *= 1.15
        elif cw < 0.25:
            score *= 1.00
        elif cw < 0.35:
            score *= 0.90
        else:
            score *= 0.75

    # System agreement: mind-calibrated vs raw
    sa = game_data.get('system_agreement', 0.0)
    if sa < 0.05:
        score *= 1.05
    elif sa > 0.15:
        score *= 0.92

    # Sharp money: informed bettors close at ~54-56% accuracy
    if game_data.get('sharp_confidence', 0.0) > 0.65:
        score *= 1.08

    # Significant lineup data = major accuracy signal
    if abs(game_data.get('lineup_adjustment', 0.0)) > 0.05:
        score *= 1.10

    # High momentum uncertainty = LSTM wasn't sure
    if game_data.get('momentum_uncertainty', 0.0) > 0.08:
        score *= 0.95

    return min(1.5, max(0.5, score))


# ============================================================
# SECTION 5: INTEGER LINEAR PROGRAMMING OPTIMIZER
# ============================================================
def optimize_points_ilp(games_data: List[dict]) -> Optional[Dict[str, int]]:
    """
    Uses PuLP ILP to find the provably optimal point assignment.
    Each game gets a unique integer 1..N. Maximize sum of
    (win_prob * points * confidence_score) across all games.

    Returns dict of game_id -> points, or None if ILP fails.
    """
    if not PULP_AVAILABLE:
        log.warning('PuLP not available — falling back to greedy assignment')
        return None

    N = len(games_data)
    if N == 0:
        return {}

    try:
        prob = pulp.LpProblem('CBS_Points', pulp.LpMaximize)

        # x[i][j] = 1 if game i gets point value (j+1)
        # j runs 0..N-1, point value = j+1 (so 1..N)
        x = [
            [pulp.LpVariable(f'x_{i}_{j}', cat='Binary') for j in range(N)]
            for i in range(N)
        ]

        # Each game gets exactly one point value
        for i in range(N):
            prob += pulp.lpSum(x[i][j] for j in range(N)) == 1

        # Each point value used by exactly one game
        for j in range(N):
            prob += pulp.lpSum(x[i][j] for i in range(N)) == 1

        # Uncertain games (wide confidence bounds) cannot receive top 30% of points
        for i in range(N):
            cw = games_data[i].get('confidence_width')
            if cw is not None and cw > 0.40:
                max_j = max(0, int(N * 0.70) - 1)  # highest allowed j index
                for j in range(max_j, N):
                    prob += x[i][j] == 0

        # Games below 52% win probability capped at bottom 25% of points
        for i in range(N):
            if games_data[i]['prob'] < 0.52:
                max_j = max(0, int(N * 0.25) - 1)
                for j in range(max_j, N):
                    prob += x[i][j] == 0

        # Objective: maximize expected personal score
        prob += pulp.lpSum(
            x[i][j] * games_data[i]['prob'] * (j + 1) * games_data[i]['confidence_score']
            for i in range(N)
            for j in range(N)
        )

        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=60)
        status = prob.solve(solver)

        if pulp.LpStatus[status] not in ('Optimal', 'Feasible'):
            log.warning(f'ILP solver returned status: {pulp.LpStatus[status]}')
            return None

        assignments = {}
        for i in range(N):
            for j in range(N):
                val = pulp.value(x[i][j])
                if val is not None and val > 0.5:
                    assignments[games_data[i]['game_id']] = j + 1
                    break

        if len(assignments) < N:
            log.warning(f'ILP produced incomplete assignment ({len(assignments)}/{N} games)')
            return None

        log.info(f'ILP optimization complete: {N} games assigned')
        return assignments

    except Exception as exc:
        log.warning(f'ILP optimization failed: {exc}')
        return None


def greedy_assign_points(games_data: List[dict]) -> Dict[str, int]:
    """
    Fallback: assign points by sorting on prob * confidence_score descending.
    Highest combined score gets highest points.
    """
    N = len(games_data)
    sorted_games = sorted(
        games_data,
        key=lambda g: g['prob'] * g['confidence_score'],
        reverse=True
    )
    return {g['game_id']: (N - rank) for rank, g in enumerate(sorted_games)}


# ============================================================
# SECTION 6: UNCERTAINTY PENALTY MODEL
# ============================================================
def apply_uncertainty_penalty(prob: float, confidence_width: Optional[float],
                               sample_size: int) -> float:
    """
    Shrinks probability toward 0.5 when bounds are wide or data is thin.
    A probability we're uncertain about should be treated closer to a coin flip.
    """
    if confidence_width is None or sample_size < 30:
        shrinkage = 0.15
        return prob * (1 - shrinkage) + 0.5 * shrinkage

    if confidence_width > 0.35:
        shrinkage = 0.10
        return prob * (1 - shrinkage) + 0.5 * shrinkage

    if confidence_width > 0.25:
        shrinkage = 0.05
        return prob * (1 - shrinkage) + 0.5 * shrinkage

    return prob  # Tight bounds — use as-is


# ============================================================
# SECTION 7: GAME VARIANCE CLASSIFIER
# ============================================================
def classify_variance(game_total: Optional[float], spread: Optional[float]) -> str:
    """
    Classifies game outcome variance from betting line data.
    Higher over/under + close spread = more variance = less reliable probability.
    """
    if game_total is None:
        return 'unknown'

    abs_spread = abs(spread) if spread is not None else 4.0

    if game_total >= 52 and abs_spread <= 6:
        return 'high'
    elif game_total <= 40 and abs_spread >= 7:
        return 'low'
    elif abs_spread >= 14:
        return 'very_low'
    else:
        return 'medium'


# ============================================================
# SECTION 8: SEASON PHASE ACCURACY ADJUSTER
# ============================================================
def get_season_phase(week_number: int) -> Tuple[str, float]:
    """
    Returns (phase_name, accuracy_modifier) for the given NFL week.
    Modifies how much to trust model probabilities based on training sample size.
    """
    for phase, cfg in SEASON_PHASE_ACCURACY.items():
        if week_number in cfg['weeks']:
            return phase, cfg['accuracy_modifier']
    return 'mid', 0.0  # Default for preseason / unknown week


def _get_week_number(date_str: str) -> int:
    """Estimate NFL week from a date string. Falls back to 9 (mid-season)."""
    try:
        dt = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        # NFL regular season starts first Thursday of September
        season_start = datetime.datetime(dt.year, 9, 1)
        while season_start.weekday() != 3:  # Thursday
            season_start += datetime.timedelta(days=1)
        days_in = (dt - season_start).days
        if days_in < 0:
            return 1
        week = max(1, min(18, (days_in // 7) + 1))
        return week
    except Exception:
        return 9  # Mid-season default


# ============================================================
# SECTION 9: UPSET LEVERAGE DETECTOR (ACCURACY-BASED)
# ============================================================
def accuracy_upset_signal(game_data: dict) -> dict:
    """
    Identifies genuine accuracy-based underdog picks.
    Requires 3+ independent signals confirming the underdog is the correct pick.
    Not contrarian — only fires when data says the underdog is actually better.
    """
    signals = {
        'underdog_viable':          0.40 < game_data['prob'] < 0.50,
        'sharp_agrees':             (game_data.get('sharp_confidence', 0) > 0.60
                                     and game_data.get('sharp_prob_delta', 0) < 0),
        'lineup_favors_underdog':   game_data.get('lineup_adjustment', 0) < -0.06,
        'momentum_favors_underdog': game_data.get('momentum_score', 0.5) < 0.42,
        'system_certain':           (game_data.get('confidence_width') is not None
                                     and game_data['confidence_width'] < 0.28),
    }

    signal_count = sum(signals.values())
    return {
        'pick_underdog': signal_count >= 3,
        'confidence':    signal_count / len(signals),
        'signals':       signals,
    }


# ============================================================
# SECTION 10: HISTORICAL ACCURACY VALIDATOR
# ============================================================
def validate_historical_scoring(data: dict) -> dict:
    """
    Compares ILP assignments against greedy assignments on historical CBS picks
    stored in data.json. Measures raw points earned, not pool finish.
    Only runs during training — skipped if fewer than 10 historical weeks exist.
    """
    results = {
        'ilp_total_points':    0,
        'greedy_total_points': 0,
        'ilp_weekly_avg':      None,
        'greedy_weekly_avg':   None,
        'ilp_improvement_pct': None,
        'weeks_analyzed':      0,
    }

    historical = (data.get('historical') or {}).get('cbs_weeks', [])
    if len(historical) < 10:
        return results  # Not enough data for meaningful validation

    for week in historical:
        games_raw = week.get('games', [])
        actuals   = week.get('actual_results', {})
        if not games_raw or not actuals:
            continue

        # Build game_data list for this historical week
        games_data = []
        for g in games_raw:
            gd = {
                'game_id':         g.get('game_id', str(len(games_data))),
                'prob':            _safe_float(g.get('win_probability')) or 0.55,
                'confidence_width': _safe_float(g.get('confidence_width')),
                'confidence_score': 1.0,
            }
            gd['prob']             = max(0.05, min(0.95, gd['prob']))
            gd['confidence_score'] = confidence_score(gd)
            games_data.append(gd)

        ilp_assign    = optimize_points_ilp(games_data) or greedy_assign_points(games_data)
        greedy_assign = greedy_assign_points(games_data)

        ilp_score    = _score_week(ilp_assign, actuals)
        greedy_score = _score_week(greedy_assign, actuals)

        results['ilp_total_points']    += ilp_score
        results['greedy_total_points'] += greedy_score
        results['weeks_analyzed']      += 1

    w = results['weeks_analyzed']
    if w > 0:
        results['ilp_weekly_avg']    = round(results['ilp_total_points'] / w, 1)
        results['greedy_weekly_avg'] = round(results['greedy_total_points'] / w, 1)
        if results['greedy_total_points'] > 0:
            results['ilp_improvement_pct'] = round(
                (results['ilp_total_points'] - results['greedy_total_points'])
                / results['greedy_total_points'] * 100, 1
            )

    return results


def _score_week(assignment: Dict[str, int], actuals: Dict[str, str]) -> int:
    """Returns total points earned from an assignment given actual game results."""
    total = 0
    for game_id, points in assignment.items():
        if actuals.get(game_id) == 'correct':
            total += points
    return total


# ============================================================
# SECTION 11: OUTPUT FORMATTER AND DATA.JSON WRITER
# ============================================================
def _build_reasoning(game_data: dict, variance_class: str, upset_sig: dict,
                     points: int, n_games: int) -> str:
    """Builds a human-readable reasoning string for a CBS pick."""
    parts = []

    cw = game_data.get('confidence_width')
    if cw is not None:
        if cw < 0.15:
            parts.append(f'Tight Bayesian bounds ({cw:.2f})')
        elif cw < 0.25:
            parts.append(f'Moderate confidence bounds ({cw:.2f})')
        else:
            parts.append(f'Wide confidence bounds ({cw:.2f}) — uncertainty penalised')

    sc = game_data.get('sharp_confidence', 0.0)
    if sc > 0.65:
        parts.append('Sharp money confirms direction')
    elif sc > 0.40:
        parts.append('Mild sharp signal')

    la = game_data.get('lineup_adjustment', 0.0)
    if abs(la) > 0.05:
        direction = 'home' if la > 0 else 'away'
        parts.append(f'Lineup favours {direction} team (scout: {la:+.2f})')

    ms = game_data.get('momentum_score', 0.5)
    if ms > 0.60:
        parts.append('Strong home momentum (LSTM)')
    elif ms < 0.40:
        parts.append('Home team cold entering game (LSTM)')

    if variance_class in ('high',):
        parts.append('High-variance game — confidence penalised')
    elif variance_class in ('very_low', 'low'):
        parts.append('Low-variance profile — probability reliable')

    if points > int(n_games * 0.75):
        tier = 'Top-tier assignment'
    elif points > int(n_games * 0.50):
        tier = 'Mid-tier assignment'
    else:
        tier = 'Low-tier assignment'

    if upset_sig.get('pick_underdog'):
        parts.append('Accuracy-based underdog pick — 3+ independent signals confirm')

    prefix = tier + '. ' if parts else tier + '.'
    return prefix + '. '.join(parts) + ('.' if parts else '')


def build_assignment_output(game_data: dict, points: int, n_games: int,
                             uncertainty_adjusted_prob: float,
                             variance_class: str,
                             upset_sig: dict) -> dict:
    """Builds the per-game assignment dict for data.json output."""
    pick_side = 'home'
    if game_data['pick'] == game_data['away_team']:
        pick_side = 'away'

    sharp_signal = ''
    sc = game_data.get('sharp_confidence', 0.0)
    sd = game_data.get('sharp_prob_delta', 0.0)
    if sc > 0.65:
        sharp_signal = 'steam_move' if abs(sd) > 0.05 else 'sharp_action'
    elif sc > 0.40:
        sharp_signal = 'mild_sharp'

    lineup_signal = ''
    la = game_data.get('lineup_adjustment', 0.0)
    if abs(la) > 0.05:
        lineup_signal = 'home_starter_out' if la < 0 else 'away_starter_out'

    return {
        'game_id':                  game_data['game_id'],
        'home_team':                game_data['home_team'],
        'away_team':                game_data['away_team'],
        'pick':                     pick_side,
        'pick_team':                game_data['pick'],
        'points':                   points,
        'win_probability':          round(game_data['prob'], 4),
        'uncertainty_adjusted_prob': round(uncertainty_adjusted_prob, 4),
        'confidence_score':         round(game_data.get('confidence_score', 1.0), 3),
        'confidence_width':         game_data.get('confidence_width'),
        'variance_class':           variance_class,
        'sharp_signal':             sharp_signal,
        'lineup_signal':            lineup_signal,
        'momentum_score':           round(game_data.get('momentum_score', 0.5), 3),
        'system_agreement':         round(game_data.get('system_agreement', 0.0), 4),
        'reasoning':                _build_reasoning(game_data, variance_class, upset_sig, points, n_games),
    }


# ============================================================
# SECTION 12: ENTRY POINT
# ============================================================
def run_cbs(data: dict) -> None:
    """
    Main entry point called by brain.py --mode cbs.
    Reads game predictions from data['predictions']['nfl'],
    applies ILP optimization, writes results to data['cbs_picks'].
    Modifies data in place. Never raises.
    """
    log.info('[cbs] Starting CBS confidence pool optimization...')
    log.info(f'[cbs] ILP available: {PULP_AVAILABLE}')

    try:
        _run_cbs_inner(data)
    except Exception as exc:
        log.error(f'[cbs] Unhandled error in run_cbs: {exc}')
        log.error(traceback.format_exc())
        # Leave existing cbs_picks untouched rather than overwriting with bad data


def _run_cbs_inner(data: dict) -> None:
    # ── Fetch current NFL scoreboard for locked/unlocked status ──────────────
    espn_games = _fetch_nfl_scoreboard()
    if not espn_games:
        log.warning('[cbs] No NFL games from ESPN scoreboard')
        return

    log.info(f'[cbs] ESPN scoreboard: {len(espn_games)} games')

    # ── Read predictions from data ─────────────────────────────────────────
    predictions = data.get('predictions', {}).get('nfl', [])
    if not predictions:
        log.warning('[cbs] No NFL predictions in data — run --mode update first')
        return

    # Build lookup by team name pair for matching predictions to ESPN games
    pred_lookup: Dict[str, dict] = {}
    for p in predictions:
        key = f"{p.get('home', '')}|||{p.get('away', '')}"
        pred_lookup[key] = p

    # ── Season phase ──────────────────────────────────────────────────────
    sample_date   = espn_games[0].get('date', '') if espn_games else ''
    week_number   = _get_week_number(sample_date)
    phase, acc_mod = get_season_phase(week_number)
    log.info(f'[cbs] Week {week_number} | Phase: {phase} | Accuracy modifier: {acc_mod:+.2f}')

    # ── Build game data for all games ─────────────────────────────────────
    # Count how many predictions have calibration data for uncertainty penalty
    sample_size = sum(
        1 for p in predictions
        if p.get('confidence_lower') is not None and p.get('confidence_upper') is not None
    )

    all_games: List[dict]    = []
    locked_games: List[dict] = []

    # Recover existing locked picks from previous cbs run
    existing_locked: Dict[str, dict] = {}
    for entry in (data.get('cbs_picks') or {}).get('assignment', []):
        key = f"{entry.get('home_team', '')}|||{entry.get('away_team', '')}"
        existing_locked[key] = entry

    for espn_game in espn_games:
        key  = f"{espn_game['home']}|||{espn_game['away']}"
        pred = pred_lookup.get(key)

        if pred is None:
            # Try reverse (away @ home sometimes swapped in source data)
            rev_key = f"{espn_game['away']}|||{espn_game['home']}"
            pred    = pred_lookup.get(rev_key)

        if pred is None:
            # Build a minimal stub so we don't skip the game
            pred = {
                'id':              espn_game['id'],
                'home':            espn_game['home'],
                'away':            espn_game['away'],
                'win_probability': 0.55,
                'raw_win_probability': 0.55,
                'pick':            espn_game['home'],
                'is_guess':        True,
                'confidence':      'low',
            }
            log.debug(f"[cbs] No prediction for {espn_game['away']} @ {espn_game['home']} — using stub")

        # Ensure game_id set
        if not pred.get('id'):
            pred['id'] = espn_game['id']

        gd = aggregate_game_probability(pred, data)

        # Apply season phase modifier to probability
        phase_adj_prob = gd['prob'] * (1 + acc_mod * (gd['prob'] - 0.5))
        phase_adj_prob = max(0.05, min(0.95, phase_adj_prob))
        gd['prob']     = phase_adj_prob

        # Apply uncertainty penalty
        adj_prob = apply_uncertainty_penalty(gd['prob'], gd['confidence_width'], sample_size)
        adj_prob = max(0.05, min(0.95, adj_prob))

        # Classify variance using spread / total
        vc = classify_variance(espn_game.get('over_under'), espn_game.get('spread'))
        var_mod = VARIANCE_CONFIDENCE_MODIFIER.get(vc, 0.0)

        # Score confidence
        gd['confidence_score'] = confidence_score(gd) + var_mod
        gd['confidence_score'] = max(0.5, min(1.5, gd['confidence_score']))

        # Upset signal
        upset_sig = accuracy_upset_signal(gd)

        gd['uncertainty_adjusted_prob'] = adj_prob
        gd['variance_class']            = vc
        gd['upset_signal']              = upset_sig
        gd['over_under']                = espn_game.get('over_under')
        gd['spread']                    = espn_game.get('spread')
        gd['is_locked']                 = espn_game['is_locked']

        all_games.append(gd)

        if espn_game['is_locked']:
            prior = existing_locked.get(key)
            locked_games.append({
                'game_data':   gd,
                'locked_entry': prior,
            })

    if not all_games:
        log.warning('[cbs] No games to optimize')
        return

    num_games    = len(all_games)
    locked_ids   = {g['game_data']['game_id'] for g in locked_games}
    unlocked_gd  = [g for g in all_games if not g['is_locked']]
    locked_pts_used = set()

    # ── Recover point values for locked games ────────────────────────────
    locked_assignment_output = []
    for lk in locked_games:
        gd    = lk['game_data']
        prior = lk.get('locked_entry')
        pts   = prior.get('points', 1) if prior else 1
        locked_pts_used.add(pts)
        locked_assignment_output.append(
            build_assignment_output(gd, pts, num_games,
                                    gd['uncertainty_adjusted_prob'],
                                    gd['variance_class'],
                                    gd['upset_signal'])
        )

    # ── Assign available point slots to unlocked games ────────────────────
    available_pts = [p for p in range(1, num_games + 1) if p not in locked_pts_used]

    if len(unlocked_gd) > len(available_pts):
        unlocked_gd = unlocked_gd[:len(available_pts)]

    if not unlocked_gd:
        final_assignment = locked_assignment_output
    else:
        # Remap game IDs to sequential indices for ILP (handles stub predictions)
        id_map = {g['game_id']: str(i) for i, g in enumerate(unlocked_gd)}
        for g in unlocked_gd:
            g['_ilp_id'] = id_map[g['game_id']]

        # Build game_data list with ILP IDs
        ilp_input = [{**g, 'game_id': g['_ilp_id']} for g in unlocked_gd]

        ilp_result = optimize_points_ilp(ilp_input)
        if ilp_result is None:
            ilp_result = greedy_assign_points(ilp_input)
            log.info('[cbs] Using greedy fallback assignment')
        else:
            log.info('[cbs] ILP assignment successful')

        # Map ILP point values to available_pts slots
        # ILP assigns 1..N but we need to fill available_pts slots
        available_sorted = sorted(available_pts, reverse=True)
        ilp_ranks = sorted(ilp_result.items(), key=lambda kv: kv[1], reverse=True)

        pts_map: Dict[str, int] = {}
        for rank, (ilp_id, _) in enumerate(ilp_ranks):
            pts_map[ilp_id] = available_sorted[rank] if rank < len(available_sorted) else 1

        unlocked_output = []
        for g in unlocked_gd:
            pts = pts_map.get(g['_ilp_id'], 1)
            unlocked_output.append(
                build_assignment_output(g, pts, num_games,
                                        g['uncertainty_adjusted_prob'],
                                        g['variance_class'],
                                        g['upset_signal'])
            )

        final_assignment = locked_assignment_output + unlocked_output

    # Sort by points descending for display
    final_assignment.sort(key=lambda x: x['points'], reverse=True)

    # ── Compute expected points ───────────────────────────────────────────
    expected_pts = sum(
        entry['uncertainty_adjusted_prob'] * entry['points']
        for entry in final_assignment
    )

    # ── Historical validation ─────────────────────────────────────────────
    hist = validate_historical_scoring(data)

    # ── Write to data ─────────────────────────────────────────────────────
    max_possible = sum(range(1, num_games + 1))

    data['cbs_picks'] = {
        'generated_at':        datetime.datetime.utcnow().isoformat(),
        'week_number':         week_number,
        'num_games':           num_games,
        'max_possible_points': max_possible,
        'expected_points':     round(expected_pts, 1),
        'season_phase':        phase,
        'accuracy_modifier':   acc_mod,
        'assignment':          final_assignment,
        'historical_validation': hist,
    }

    log.info(f'[cbs] Complete: {num_games} games | {len(locked_games)} locked | '
             f'expected {expected_pts:.1f}/{max_possible} pts | phase={phase}')
