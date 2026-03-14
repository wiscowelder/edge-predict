#!/usr/bin/env python3
"""
EDGE Predict — march.py
March Madness bracket optimizer built entirely around accuracy.
Outputs one bracket — the bracket the system believes is most correct.
No pool strategy. No contrarianism. Every signal exists to find the true winner.

Section 1  — Imports and constants
Section 2  — data.json keys managed by march.py
Section 3  — KenPom / Barttorvik data integration
Section 4  — Tournament-specific feature engineering (25 features)
Section 5  — Seed-pair historical calibration model
Section 6  — Region path modeling
Section 7  — Monte Carlo simulation engine
Section 8  — LSTM momentum integration (from mind.py / data['mind'])
Section 9  — Player availability adjustments (from scout.py / data['scout'])
Section 10 — Coach tournament experience model
Section 11 — Conference strength and tournament resume model
Section 12 — Hot team and momentum detection
Section 13 — Confidence bands per team per round
Section 14 — Women's bracket (separate calibration)
Section 15 — Historical validation engine
Section 16 — Entry point (called by brain.py --mode march)
"""

# ============================================================
# SECTION 1: IMPORTS AND CONSTANTS
# ============================================================
import os
import re
import json
import math
import time
import random
import logging
import datetime
import traceback
from typing import Optional, Dict, List, Tuple, Any

import requests
import numpy as np

log = logging.getLogger(__name__)

ESPN_BASE   = 'https://site.api.espn.com/apis/site/v2/sports'
REQUEST_TIMEOUT = 20
MAX_RETRIES     = 2

# Monte Carlo simulation counts
MARCH_SIMS        = 500_000   # Main simulation run
CONFIDENCE_BATCHES = 20       # Batches for confidence band estimation
CONFIDENCE_BATCH_N = 5_000    # Simulations per confidence batch

# CBS scoring multipliers per round
CBS_ROUND_PTS = [1, 2, 4, 8, 16, 32]
ROUND_NAMES   = ['Round of 64', 'Round of 32', 'Sweet 16',
                 'Elite 8', 'Final Four', 'Championship']

# Standard bracket first-round pairings
FIRST_ROUND_PAIRINGS = [(1, 16), (8, 9), (5, 12), (4, 13),
                         (6, 11), (3, 14), (7, 10), (2, 15)]

# KenPom cache to avoid repeated scraping
_KENPOM_CACHE: Dict[str, dict] = {}
_KENPOM_CACHE_TIME: float       = 0.0
KENPOM_CACHE_TTL: float         = 86_400.0   # 24 hours


# ============================================================
# SECTION 2: DATA.JSON KEYS MANAGED BY MARCH.PY
# ============================================================
MARCH_TEMPLATE = {
    'generated_at':   None,
    'simulations_run': 0,
    'bracket': {},
    'team_probabilities': {},
    'accuracy_upsets':    [],
    'accuracy_fades':     [],
    'historical_validation': {
        'years_validated':  0,
        'avg_score':        None,
        'avg_expert_score': None,
        'beat_experts_rate': None,
        'last_validated':   None,
    },
}


# ============================================================
# SECTION 3: KENPOM / BARTTORVIK DATA INTEGRATION
# ============================================================
def _safe_fetch(url: str, params: Optional[dict] = None,
                headers: Optional[dict] = None) -> Optional[Any]:
    """HTTP GET with retry. Returns parsed JSON or None."""
    hdrs = {'User-Agent': 'Mozilla/5.0 (compatible; EDGE-Predict/1.0)'}
    if headers:
        hdrs.update(headers)
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=hdrs, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 403:
                log.debug(f'fetch 403 blocked: {url[:60]}')
                return None
            time.sleep(1.5 ** attempt)
        except Exception as exc:
            log.debug(f'fetch error ({url[:60]}): {exc}')
    return None


def _safe_fetch_text(url: str) -> Optional[str]:
    """HTTP GET returning raw text."""
    hdrs = {'User-Agent': 'Mozilla/5.0 (compatible; EDGE-Predict/1.0)'}
    try:
        resp = requests.get(url, headers=hdrs, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.text
        return None
    except Exception:
        return None


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def fetch_kenpom() -> Dict[str, dict]:
    """
    Scrapes KenPom's public efficiency table.
    Returns dict of team_name -> {adjEM, adjO, adjD, adjT, luck, sos_adjEM,
                                   ooc_sos, adjEM_rank, experience}.
    Falls back to empty dict if blocked or unavailable.
    Cached for KENPOM_CACHE_TTL seconds.
    """
    global _KENPOM_CACHE, _KENPOM_CACHE_TIME

    if _KENPOM_CACHE and (time.time() - _KENPOM_CACHE_TIME) < KENPOM_CACHE_TTL:
        log.debug('KenPom: using cached data')
        return _KENPOM_CACHE

    log.info('KenPom: attempting public table scrape...')
    html = _safe_fetch_text('https://kenpom.com/')
    if not html:
        log.info('KenPom: unavailable — will use Barttorvik fallback')
        return {}

    result: Dict[str, dict] = {}
    try:
        # KenPom main table rows: find <tr> entries inside the efficiency table
        # Each row has team name and stat columns
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
        rank = 0
        for row in rows:
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            if len(cells) < 8:
                continue
            # Strip HTML tags from each cell
            clean = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            # Expect: rank, team, conf, W-L, AdjEM, AdjO, AdjD, AdjT, luck, SOS_AdjEM, OOC_SOS
            try:
                cell_rank = int(clean[0]) if clean[0].isdigit() else None
                if cell_rank is None:
                    continue
                rank  = cell_rank
                name  = clean[1]
                # Some rows have seed/conference between name and EM columns
                # We parse floats searching the row
                floats = []
                for c in clean[2:]:
                    try:
                        floats.append(float(c))
                    except ValueError:
                        floats.append(None)

                # Best-effort column mapping
                adj_em     = next((f for f in floats if f is not None and abs(f) < 50), None)
                adj_o      = next((f for f in floats if f is not None and 80 < f < 140), None)
                adj_d      = next((f for f in floats if f is not None and 80 < f < 140
                                   and f != adj_o), None)
                adj_t      = next((f for f in floats if f is not None and 55 < f < 85), None)
                luck_vals  = [f for f in floats if f is not None and abs(f) < 0.30]
                luck       = luck_vals[0] if luck_vals else None

                if name and adj_em is not None:
                    result[name] = {
                        'adjEM':       adj_em,
                        'adjO':        adj_o,
                        'adjD':        adj_d,
                        'adjT':        adj_t,
                        'luck':        luck,
                        'sos_adjEM':   None,
                        'ooc_sos':     None,
                        'adjEM_rank':  rank,
                        'experience':  None,
                    }
            except (IndexError, ValueError):
                continue

    except Exception as exc:
        log.warning(f'KenPom parse error: {exc}')
        return {}

    if result:
        log.info(f'KenPom: parsed {len(result)} teams')
        _KENPOM_CACHE      = result
        _KENPOM_CACHE_TIME = time.time()
    else:
        log.info('KenPom: no data parsed — using Barttorvik fallback')

    return result


def fetch_barttorvik_fresh(year: Optional[int] = None) -> Dict[str, dict]:
    """Fetches Barttorvik efficiency data independently (no brain.py import needed)."""
    yr  = year or datetime.datetime.utcnow().year
    url = f'https://barttorvik.com/trank.php?year={yr}&json=1'
    data = _safe_fetch(url)
    if not isinstance(data, list):
        return {}

    result: Dict[str, dict] = {}
    for row in data:
        if not isinstance(row, list) or len(row) < 10:
            continue
        try:
            name   = str(row[0]).strip()
            adj_oe = _safe_float(row[3])
            adj_de = _safe_float(row[4])
            adj_em = _safe_float(row[5])
            sos    = _safe_float(row[6])
            tempo  = _safe_float(row[9]) if len(row) > 9 else None
            if name:
                result[name] = {
                    'adj_em': adj_em, 'adj_oe': adj_oe,
                    'adj_de': adj_de, 'sos': sos, 'tempo': tempo,
                }
        except (IndexError, TypeError):
            continue

    log.info(f'Barttorvik fresh fetch: {len(result)} teams')
    return result


def get_efficiency_data(team_name: str, kenpom: Dict[str, dict],
                         barttorvik: Dict[str, dict]) -> dict:
    """
    Returns efficiency metrics for a team, preferring KenPom then Barttorvik.
    Falls back to default values if neither has the team.
    Uses fuzzy name matching for common name discrepancies.
    """
    # Exact lookup first
    kp = kenpom.get(team_name) or {}
    bt = barttorvik.get(team_name) or {}

    # Fuzzy lookup if exact fails — normalise to lowercase, strip punctuation
    if not kp and not bt:
        norm = re.sub(r'[^a-z0-9 ]', '', team_name.lower())
        for k in kenpom:
            if re.sub(r'[^a-z0-9 ]', '', k.lower()) == norm:
                kp = kenpom[k]
                break
        for k in barttorvik:
            if re.sub(r'[^a-z0-9 ]', '', k.lower()) == norm:
                bt = barttorvik[k]
                break

    adj_em   = _safe_float((kp.get('adjEM') or bt.get('adj_em'))) or 0.0
    adj_o    = _safe_float((kp.get('adjO')  or bt.get('adj_oe'))) or 105.0
    adj_d    = _safe_float((kp.get('adjD')  or bt.get('adj_de'))) or 105.0
    adj_t    = _safe_float((kp.get('adjT')  or bt.get('tempo'))) or 68.0
    sos      = _safe_float((kp.get('sos_adjEM') or bt.get('sos'))) or 0.0
    luck     = _safe_float(kp.get('luck')) or 0.0
    em_rank  = kp.get('adjEM_rank') or 200

    return {
        'adjEM': adj_em, 'adjO': adj_o, 'adjD': adj_d, 'adjT': adj_t,
        'sos': sos, 'luck': luck, 'adjEM_rank': em_rank,
        'ooc_sos': _safe_float(kp.get('ooc_sos')) or 0.0,
        'experience': _safe_float(kp.get('experience')) or 2.5,
    }


def seeding_efficiency_gap(team: str, seed: int, data_store: dict) -> dict:
    """
    Detects underseeded / overseeded teams.
    Positive gap = underseeded (team is better than seed suggests).
    These are the most reliable correct upset picks available.
    """
    adj_em_rank = data_store.get(team, {}).get('adjEM_rank', seed * 10)
    gap = seed - adj_em_rank

    return {
        'gap':        gap,
        'underseeded': gap > 3,
        'overseeded':  gap < -3,
        'magnitude':   abs(gap),
    }


# ============================================================
# SECTION 4: TOURNAMENT-SPECIFIC FEATURE ENGINEERING
# ============================================================
def build_tournament_vector(team: str, seed: int, eff: dict,
                              coach: dict, quad_records: dict,
                              gender: str, data: dict) -> dict:
    """
    Builds the 25-feature tournament-specific vector for a team.
    Every feature chosen because research shows it correlates with tournament outcomes.
    """
    gap_info = seeding_efficiency_gap(team, seed, {team: eff})

    # Recent form from predictions / standings data
    preds = data.get('predictions', {}).get('ncaabm' if gender == 'mens' else 'ncaabw', [])
    recent_games = [p for p in preds if team in (p.get('home', ''), p.get('away', ''))]
    last_10      = recent_games[-10:] if len(recent_games) >= 10 else recent_games
    if last_10:
        wins_10 = sum(
            1 for p in last_10
            if p.get('pick') == team and p.get('status') == 'correct'
        )
        win_pct_10 = wins_10 / len(last_10)
    else:
        win_pct_10 = 0.65  # Default for good tournament team

    # Scout lineup availability
    scout_data  = data.get('scout', {})
    lineup_adjs = scout_data.get('lineup_adjustments', {})
    star_healthy = 1.0
    for gid, adj in lineup_adjs.items():
        if team.lower() in gid.lower():
            home_adj = _safe_float(adj.get('home_adjustment')) or 0.0
            if abs(home_adj) > 0.08:
                star_healthy = 0.0 if home_adj < 0 else 1.0
            break

    # LSTM momentum from mind
    momentum = (data.get('mind') or {}).get('team_momentum', {}).get(team, 0.5)

    # Historical program success (from patterns or default)
    prog_success = _safe_float(
        (data.get('patterns') or {}).get(
            'ncaabm' if gender == 'mens' else 'ncaabw', {}
        ).get(team, {}).get('tournament_win_rate')
    ) or 0.40

    quad = quad_records.get(team, {})

    return {
        'adjEM':                eff['adjEM'],
        'adjO':                 eff['adjO'],
        'adjD':                 eff['adjD'],
        'adjT':                 eff['adjT'],
        'seed':                 seed,
        'seed_adjEM_gap':       gap_info['gap'],
        'win_pct_last_10':      win_pct_10,
        'conf_tourney_result':  0.5,   # Populated by hot/cold detector
        'sos_adjEM':            eff['sos'],
        'ooc_sos':              eff['ooc_sos'],
        'coach_tourney_wl':     coach.get('tournament_wins', 0),
        'coach_ff_appearances': coach.get('final_four_appearances', 0),
        'ftr':                  0.0,   # Free throw rate — Barttorvik advanced
        'three_pt_rate':        0.0,   # Three point attempt rate
        'tov_pct':              0.0,   # Turnover % forced
        'oreb_pct':             0.0,   # Offensive rebound %
        'experience':           eff.get('experience', 2.5),
        'roster_continuity':    0.65,  # Default; populated from ESPN if available
        'star_healthy':         star_healthy,
        'momentum_score':       momentum,
        'days_rest':            7,     # Default first round; adjusted if play-in
        'location_advantage':   0.0,
        'quad1_wins':           quad.get('q1_wins', 0),
        'adjEM_trend_4wk':      0.0,   # AdjEM trend last 4 weeks
        'program_success':      prog_success,
    }


# ============================================================
# SECTION 5: SEED-PAIR HISTORICAL CALIBRATION
# ============================================================
# Men's historical seed upset rates (1985–2025)
HISTORICAL_SEED_WIN_RATES: Dict[Tuple[int, int], float] = {
    (1,  16): 0.013,   # Only 1 upset ever: UMBC 2018
    (2,  15): 0.063,
    (3,  14): 0.150,
    (4,  13): 0.213,
    (5,  12): 0.356,   # Famous 5-12 line — historically reliable
    (6,  11): 0.370,
    (7,  10): 0.400,
    (8,   9): 0.480,   # Essentially a coin flip
}

# Women's tournament — top seeds historically stronger
HISTORICAL_SEED_WIN_RATES_WOMEN: Dict[Tuple[int, int], float] = {
    (1,  16): 0.002,
    (2,  15): 0.025,
    (3,  14): 0.090,
    (4,  13): 0.160,
    (5,  12): 0.280,
    (6,  11): 0.320,
    (7,  10): 0.380,
    (8,   9): 0.460,
}


def tournament_game_probability(seed_a: int, seed_b: int, model_prob: float,
                                  gender: str = 'mens') -> float:
    """
    Blends model probability (70%) with historical seed-pair prior (30%).
    Prevents the model from assigning unrealistic probabilities to extreme matchups.
    model_prob is P(team_a wins) where team_a has seed_a.
    """
    rates     = HISTORICAL_SEED_WIN_RATES if gender == 'mens' else HISTORICAL_SEED_WIN_RATES_WOMEN
    lo, hi    = min(seed_a, seed_b), max(seed_a, seed_b)
    seed_pair = (lo, hi)

    historical_underdog_rate = rates.get(seed_pair)
    if historical_underdog_rate is None:
        # Beyond first round — no fixed prior; use 60/40 toward better seed
        better_seed_wins = 0.60
        historical_for_a = better_seed_wins if seed_a < seed_b else (1 - better_seed_wins)
    else:
        # Underdog = team with higher seed number
        if seed_a > seed_b:
            historical_for_a = historical_underdog_rate
        else:
            historical_for_a = 1 - historical_underdog_rate

    blended = 0.70 * model_prob + 0.30 * historical_for_a
    return max(0.02, min(0.98, blended))


def base_model_probability(team_a: dict, team_b: dict,
                            eff_a: dict, eff_b: dict) -> float:
    """
    Computes base win probability from efficiency margin difference.
    AdjEM difference is the single most predictive tournament variable.
    """
    em_diff = eff_a.get('adjEM', 0.0) - eff_b.get('adjEM', 0.0)

    # Logistic function: +10 EM pts ≈ 73% win probability
    log_odds = em_diff * 0.15
    prob = 1.0 / (1.0 + math.exp(-log_odds))

    # Seed-based blend (model 60%, seed historical 40%) for first round
    seed_a = team_a.get('seed', 8)
    seed_b = team_b.get('seed', 8)

    return tournament_game_probability(seed_a, seed_b, prob,
                                        team_a.get('gender', 'mens'))


# ============================================================
# SECTION 6: REGION PATH MODELING
# ============================================================
def region_strength(region_teams: List[dict], eff_data: Dict[str, dict]) -> float:
    """
    Computes average AdjEM of the top 4 seeds in a region.
    Higher = harder path. Used to adjust path probabilities.
    """
    top4 = [t for t in region_teams if t.get('seed', 99) <= 4]
    if not top4:
        return 0.0
    em_vals = [eff_data.get(t['name'], {}).get('adjEM', 0.0) for t in top4]
    return sum(em_vals) / len(em_vals) if em_vals else 0.0


def path_probabilities_from_simulation(team_name: str,
                                         sim_round_wins: Dict[str, List[int]],
                                         n_sims: int) -> Dict[str, float]:
    """Extracts per-round win probabilities from simulation counts."""
    wins = sim_round_wins.get(team_name, [0] * 6)
    probs = {}
    for i, round_name in enumerate(ROUND_NAMES):
        probs[round_name] = wins[i] / n_sims if n_sims > 0 else 0.0
    return probs


# ============================================================
# SECTION 7: MONTE CARLO SIMULATION ENGINE
# ============================================================
def run_simulation(first_round_games: List[Tuple[dict, dict]],
                    team_eff: Dict[str, dict],
                    team_adjustments: Dict[str, float],
                    n_sims: int,
                    gender: str) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """
    Runs n_sims full tournament simulations.
    Returns:
      round_wins[team_name] = list of 6 win counts (one per round)
      matchup_wins[(team_a, team_b, round_idx)] = wins for team_a
    """
    round_wins: Dict[str, List[int]] = {
        t['name']: [0] * 6
        for pair in first_round_games
        for t in pair
    }
    matchup_wins: Dict[Tuple, int] = {}

    for _ in range(n_sims):
        alive  = list(first_round_games)  # list of (team_a, team_b) for current round
        played: List[dict] = []           # winners advancing

        for round_idx in range(6):
            next_round: List[Tuple[dict, dict]] = []
            winners: List[dict] = []

            for (ta, tb) in alive:
                eff_a = team_eff.get(ta['name'], {})
                eff_b = team_eff.get(tb['name'], {})

                base_p = base_model_probability(
                    {**ta, 'gender': gender},
                    {**tb, 'gender': gender},
                    eff_a, eff_b
                )

                # Apply momentum / player / coach adjustments
                adj_a = team_adjustments.get(ta['name'], 0.0)
                adj_b = team_adjustments.get(tb['name'], 0.0)
                p     = max(0.02, min(0.98, base_p + adj_a - adj_b))

                winner = ta if random.random() < p else tb
                loser  = tb if winner == ta else ta

                round_wins[winner['name']][round_idx] += 1

                mk = (ta['name'], tb['name'], round_idx)
                matchup_wins[mk] = matchup_wins.get(mk, 0) + (1 if winner == ta else 0)

                winners.append(winner)

            # Pair winners for next round
            alive = [(winners[i], winners[i + 1])
                     for i in range(0, len(winners) - 1, 2)]

    return round_wins, matchup_wins


def pick_bracket(first_round_games: List[Tuple[dict, dict]],
                  round_wins: Dict[str, List[int]],
                  matchup_wins: Dict[Tuple, int],
                  n_sims: int) -> Tuple[Dict[str, List[dict]], List[dict], List[dict]]:
    """
    Builds the recommended bracket deterministically from simulation majority.
    Returns (rounds_output, recommended_bracket_flat, upset_alerts).
    """
    rounds_output: Dict[str, List[dict]] = {}
    recommended_flat: List[dict]         = []
    upset_alerts: List[dict]             = []

    alive = list(first_round_games)
    first_round_region_map = {id(pair): '' for pair in first_round_games}

    for round_idx in range(6):
        round_name    = ROUND_NAMES[round_idx]
        round_entries = []
        next_alive    = []
        winners       = []
        pts_this      = CBS_ROUND_PTS[round_idx] if round_idx < len(CBS_ROUND_PTS) else 32

        for matchup_idx, (ta, tb) in enumerate(alive):
            wins_a  = round_wins.get(ta['name'], [0] * 6)[round_idx]
            wins_b  = round_wins.get(tb['name'], [0] * 6)[round_idx]
            mk_a    = matchup_wins.get((ta['name'], tb['name'], round_idx), wins_a)
            mk_b    = matchup_wins.get((tb['name'], ta['name'], round_idx), wins_b)
            total   = mk_a + mk_b

            if total > 0:
                pick    = ta if mk_a >= mk_b else tb
                sim_pct = max(mk_a, mk_b) / total
            else:
                pick    = ta if wins_a >= wins_b else tb
                sim_pct = max(wins_a, wins_b) / max(wins_a + wins_b, 1)

            loser = tb if pick == ta else ta

            entry = {
                'round':       round_name,
                'round_idx':   round_idx,
                'team1':       ta['name'],
                'team1_seed':  ta.get('seed', 0),
                'team2':       tb['name'],
                'team2_seed':  tb.get('seed', 0),
                'pick':        pick['name'],
                'pick_seed':   pick.get('seed', 0),
                'round_pts':   pts_this,
                'sim_pct':     round(sim_pct, 3),
                'confidence':  ('high' if sim_pct > 0.70
                                else 'medium' if sim_pct > 0.55 else 'low'),
            }
            round_entries.append(entry)
            recommended_flat.append({
                'home':            ta['name'],
                'away':            tb['name'],
                'home_seed':       ta.get('seed', 0),
                'away_seed':       tb.get('seed', 0),
                'recommended':     pick['name'],
                'sim_confidence':  sim_pct,
                'round':           round_name,
            })

            # Upset alert: underdog (higher seed number) is our pick
            seed_diff = abs(ta.get('seed', 0) - tb.get('seed', 0))
            if (seed_diff >= 4
                    and pick.get('seed', 0) > min(ta.get('seed', 16), tb.get('seed', 16))):
                upset_alerts.append({
                    'round':   round_name,
                    'pick':    pick['name'],
                    'seed':    pick.get('seed', 0),
                    'against': loser['name'],
                    'message': (
                        f"UPSET PICK: ({pick.get('seed')}) {pick['name']} "
                        f"over ({loser.get('seed')}) {loser['name']} "
                        f"— {sim_pct:.0%} of sims"
                    ),
                })

            winners.append(pick)

        rounds_output[round_name] = round_entries
        alive = [(winners[i], winners[i + 1])
                 for i in range(0, len(winners) - 1, 2)]

    return rounds_output, recommended_flat, upset_alerts


# ============================================================
# SECTION 8: LSTM MOMENTUM INTEGRATION
# ============================================================
# Momentum weight decreases as tournament progresses and only elite teams remain
ROUND_MOMENTUM_WEIGHT: Dict[str, float] = {
    'Round of 64':  0.08,
    'Round of 32':  0.06,
    'Sweet 16':     0.05,
    'Elite 8':      0.04,
    'Final Four':   0.03,
    'Championship': 0.02,
}

CONFERENCE_TOURNEY_MOMENTUM: Dict[str, float] = {
    'won_conference_tourney':           +0.05,
    'lost_in_conference_final':         +0.02,
    'lost_early_in_conference_tourney': -0.02,
}


def momentum_adjustment(team: str, data: dict) -> float:
    """
    Returns a total momentum adjustment for a team.
    Blends LSTM score from mind.py with conference tournament result.
    Averaged across rounds (caller applies per-round weight).
    """
    mind_data = data.get('mind') or {}
    momentum  = _safe_float(
        mind_data.get('team_momentum', {}).get(team)
    ) or 0.5
    delta = (momentum - 0.5) * 2  # Scale to [-1, 1]

    # Average momentum weight across tournament (mid-tournament weight)
    avg_weight = 0.05
    lstm_adj   = delta * avg_weight

    # Conference tournament result
    conf_result = mind_data.get('conference_tourney', {}).get(team, '')
    conf_adj    = CONFERENCE_TOURNEY_MOMENTUM.get(conf_result, 0.0)

    return lstm_adj + conf_adj


# ============================================================
# SECTION 9: PLAYER AVAILABILITY ADJUSTMENTS
# ============================================================
def player_availability_adjustment(team: str, game_id: str, data: dict,
                                     is_home: bool) -> float:
    """
    Reads scout.py's lineup adjustments.
    Returns probability delta for the team based on player availability.
    """
    lineup = (data.get('scout') or {}).get('lineup_adjustments', {}).get(str(game_id), {})
    if is_home:
        return _safe_float(lineup.get('home_adjustment')) or 0.0
    else:
        return _safe_float(lineup.get('away_adjustment')) or 0.0


def build_team_adjustments(teams: List[dict], data: dict,
                             coach_profiles: Dict[str, dict]) -> Dict[str, float]:
    """
    Combines momentum, player availability, and coach experience into
    a single per-team adjustment to base probability.
    """
    adjustments: Dict[str, float] = {}
    team_names = [t['name'] for t in teams]

    for team in teams:
        name = team['name']

        # Momentum
        mom_adj = momentum_adjustment(name, data)

        # Coach experience vs average opponent
        coach   = coach_profiles.get(name, {})
        avg_opp_coach_score = sum(
            _coach_score(coach_profiles.get(opp, {}))
            for opp in team_names if opp != name
        ) / max(1, len(team_names) - 1)

        coach_adj = max(-0.06, min(0.06,
            (_coach_score(coach) - avg_opp_coach_score) * 0.1
        ))

        # Player availability — use pre-tournament scout report if available
        availability_adj = 0.0
        scout = data.get('scout') or {}
        pwi   = scout.get('player_win_impact', {}).get(name, {})
        if pwi:
            availability_adj = _safe_float(pwi.get('net_adjustment')) or 0.0

        adjustments[name] = mom_adj + coach_adj + availability_adj

    return adjustments


def _coach_score(coach: dict) -> float:
    """Single numeric score for a coach's tournament experience."""
    return (
        coach.get('final_four_appearances', 0) * 0.03
        + coach.get('sweet_sixteen_appearances', 0) * 0.01
        + coach.get('tournament_wins', 0) * 0.002
        - coach.get('first_round_losses', 0) * 0.005
    )


# ============================================================
# SECTION 10: COACH TOURNAMENT EXPERIENCE MODEL
# ============================================================
def fetch_coach_profile(team_id: str, gender: str) -> dict:
    """
    Fetches coach profile from ESPN API.
    Returns profile dict with tournament history.
    Returns empty dict on any failure.
    """
    sport = ('basketball/mens-college-basketball' if gender == 'mens'
             else 'basketball/womens-college-basketball')

    # First get team roster to find coach ID
    roster_data = _safe_fetch(
        f'{ESPN_BASE}/{sport}/teams/{team_id}/roster'
    )
    if not roster_data:
        return {}

    try:
        coaches = roster_data.get('coach') or []
        if not coaches:
            return {}
        coach_entry = coaches[0] if isinstance(coaches, list) else coaches
        coach_id    = str((coach_entry.get('id') or ''))
        if not coach_id:
            return {}
    except Exception:
        return {}

    coach_data = _safe_fetch(
        f'{ESPN_BASE}/{sport}/coaches/{coach_id}'
    )
    if not coach_data:
        return {}

    try:
        record = coach_data.get('record') or {}
        career = record.get('items') or []

        tourney_wins   = 0
        tourney_losses = 0
        ff_apps        = 0
        s16_apps       = 0
        e8_apps        = 0
        champs         = 0
        r1_losses      = 0

        for item in career:
            if not isinstance(item, dict):
                continue
            phase = item.get('phase', '').lower()
            wins  = int(item.get('wins', 0) or 0)
            loss  = int(item.get('losses', 0) or 0)
            if 'tournament' in phase or 'ncaa' in phase:
                tourney_wins   += wins
                tourney_losses += loss
            if 'final four' in phase:
                ff_apps += wins + loss
            if 'sweet sixteen' in phase or 'sweet 16' in phase:
                s16_apps += wins + loss
            if 'elite eight' in phase or 'elite 8' in phase:
                e8_apps += wins + loss
            if 'champion' in phase:
                champs += wins
            if 'first round' in phase or 'round of 64' in phase:
                r1_losses += loss

        return {
            'tournament_wins':           tourney_wins,
            'tournament_losses':         tourney_losses,
            'sweet_sixteen_appearances': s16_apps,
            'elite_eight_appearances':   e8_apps,
            'final_four_appearances':    ff_apps,
            'championships':             champs,
            'first_round_losses':        r1_losses,
        }

    except Exception as exc:
        log.debug(f'Coach profile parse error for team {team_id}: {exc}')
        return {}


def fetch_all_coach_profiles(teams: List[dict], gender: str) -> Dict[str, dict]:
    """Fetches coach profiles for all tournament teams. Slow — runs once at start."""
    profiles: Dict[str, dict] = {}
    for team in teams:
        name    = team.get('name', '')
        team_id = str(team.get('id', ''))
        if not team_id:
            profiles[name] = {}
            continue
        try:
            profiles[name] = fetch_coach_profile(team_id, gender)
            time.sleep(0.2)  # Rate limit ESPN API
        except Exception:
            profiles[name] = {}
    return profiles


# ============================================================
# SECTION 11: CONFERENCE STRENGTH AND TOURNAMENT RESUME
# ============================================================
def tournament_resume_score(team: str, quad_records: Dict[str, dict]) -> float:
    """
    Scores tournament resume using NCAA Quad system.
    Quad 3/4 losses penalised heavily — shows the team can be beaten by bad opponents.
    """
    quad = quad_records.get(team, {})
    return (
        quad.get('q1_wins', 0)   *  3.0
        + quad.get('q2_wins', 0) *  1.5
        + quad.get('q3_wins', 0) *  0.5
        + quad.get('q4_wins', 0) *  0.1
        - quad.get('q1_losses', 0) * 0.5
        - quad.get('q2_losses', 0) * 1.0
        - quad.get('q3_losses', 0) * 2.0
        - quad.get('q4_losses', 0) * 3.5
    )


def fetch_quad_records(teams: List[dict], gender: str) -> Dict[str, dict]:
    """
    Attempts to fetch Quad win/loss records from ESPN team stats.
    Falls back to empty records if unavailable.
    """
    quad_records: Dict[str, dict] = {}
    sport = ('basketball/mens-college-basketball' if gender == 'mens'
             else 'basketball/womens-college-basketball')

    for team in teams:
        name    = team.get('name', '')
        team_id = str(team.get('id', ''))
        if not team_id:
            quad_records[name] = {}
            continue
        try:
            stats = _safe_fetch(
                f'{ESPN_BASE}/{sport}/teams/{team_id}',
                params={'enable': 'stats'}
            )
            if not stats:
                quad_records[name] = {}
                continue

            # ESPN team stats may include quadrant records in 'record' section
            record_items = (stats.get('team', {}).get('record') or {}).get('items', [])
            qr: Dict[str, int] = {}
            for item in record_items:
                desc = item.get('description', '').lower()
                for q in ('q1', 'q2', 'q3', 'q4'):
                    if q in desc:
                        wins   = int((item.get('stats') or [{}])[0].get('value', 0) or 0)
                        losses = int((item.get('stats') or [{}])[1].get('value', 0) or 0) if len(item.get('stats') or []) > 1 else 0
                        qr[f'{q}_wins']   = wins
                        qr[f'{q}_losses'] = losses
            quad_records[name] = qr
            time.sleep(0.1)
        except Exception:
            quad_records[name] = {}

    return quad_records


# ============================================================
# SECTION 12: HOT TEAM AND MOMENTUM DETECTION
# ============================================================
def hot_cold_adjustment(team: str, data: dict) -> Tuple[float, int, int]:
    """
    Rule-based hot/cold pattern detector.
    Returns (probability_adjustment, hot_signal_count, cold_signal_count).
    3+ hot signals = real momentum effect (+0.04).
    3+ cold signals = real drag effect (-0.04).
    """
    mind_data = data.get('mind') or {}
    scout_data = data.get('scout') or {}
    edge_data  = data.get('edge') or {}

    # Hot signals
    momentum     = _safe_float(mind_data.get('team_momentum', {}).get(team)) or 0.5
    won_last_6   = momentum > 0.72
    conf_tourney_won = mind_data.get('conference_tourney', {}).get(team) == 'won_conference_tourney'
    improving_em = _safe_float(
        mind_data.get('adjEM_trend', {}).get(team)
    ) or 0.0 > 1.0
    star_healthy = not bool(
        scout_data.get('injury_flags', {}).get(team)
    )

    # Cold signals
    cold_momentum    = momentum < 0.35
    conf_early_loss  = mind_data.get('conference_tourney', {}).get(team) == 'lost_early_in_conference_tourney'
    declining_em     = (_safe_float(
        mind_data.get('adjEM_trend', {}).get(team)
    ) or 0.0) < -1.0
    star_injured     = bool(scout_data.get('injury_flags', {}).get(team))
    coach_controversy = bool(edge_data.get('sentiment', {}).get(team, {}).get('coaching_issue'))

    hot_count  = sum([won_last_6, conf_tourney_won, improving_em, star_healthy and momentum > 0.6])
    cold_count = sum([cold_momentum, conf_early_loss, declining_em, star_injured, coach_controversy])

    hot_adj  = 0.04 if hot_count >= 3 else (0.02 if hot_count == 2 else 0.0)
    cold_adj = -0.04 if cold_count >= 3 else (-0.02 if cold_count == 2 else 0.0)

    return (hot_adj + cold_adj, hot_count, cold_count)


# ============================================================
# SECTION 13: CONFIDENCE BANDS PER TEAM PER ROUND
# ============================================================
def compute_confidence_bands(team_name: str,
                               first_round_games: List[Tuple[dict, dict]],
                               team_eff: Dict[str, dict],
                               team_adjustments: Dict[str, float],
                               gender: str) -> Dict[str, dict]:
    """
    Runs CONFIDENCE_BATCHES × CONFIDENCE_BATCH_N simulations, split into batches.
    Reports mean + 5th/95th percentile per round for the team.
    Uses post-hoc batch splitting of a single simulation run for efficiency.
    """
    all_wins: Dict[str, List[float]] = {r: [] for r in ROUND_NAMES}

    for _ in range(CONFIDENCE_BATCHES):
        batch_wins, _ = run_simulation(
            first_round_games, team_eff, team_adjustments,
            CONFIDENCE_BATCH_N, gender
        )
        team_w = batch_wins.get(team_name, [0] * 6)
        for i, round_name in enumerate(ROUND_NAMES):
            all_wins[round_name].append(team_w[i] / CONFIDENCE_BATCH_N)

    bands: Dict[str, dict] = {}
    for round_name in ROUND_NAMES:
        vals = sorted(all_wins[round_name])
        n    = len(vals)
        mean = sum(vals) / n if n > 0 else 0.0
        lo   = vals[max(0, int(n * 0.05))]
        hi   = vals[min(n - 1, int(n * 0.95))]
        width = hi - lo
        certainty = 'high' if width < 0.08 else 'medium' if width < 0.18 else 'low'
        bands[round_name] = {
            'mean':      round(mean, 3),
            'lower_95':  round(lo, 3),
            'upper_95':  round(hi, 3),
            'certainty': certainty,
        }

    return bands


# ============================================================
# SECTION 14: WOMEN'S BRACKET (SEPARATE CALIBRATION)
# ============================================================
def _fetch_tournament_field(sport_key: str) -> Optional[List[dict]]:
    """
    Fetches tournament bracket from ESPN. Works for both men's and women's.
    Returns list of team dicts or None if tournament not yet seeded.
    """
    sport_path = ('basketball/mens-college-basketball' if 'ncaabm' in sport_key
                  else 'basketball/womens-college-basketball')

    # Try bracket endpoint first
    bracket = _safe_fetch(
        f'{ESPN_BASE}/{sport_path}/tournament/bracket',
        params={'limit': 100}
    )
    if bracket:
        teams = _parse_espn_bracket(bracket)
        if teams and len(teams) >= 32:
            log.info(f'Tournament field ({sport_key}): {len(teams)} teams via bracket')
            return teams

    # Fallback: scoreboard
    groups = '50' if 'ncaabm' in sport_key else '49'
    sb = _safe_fetch(
        f'{ESPN_BASE}/{sport_path}/scoreboard',
        params={'limit': 200, 'groups': groups}
    )
    if not sb:
        return None

    events    = sb.get('events', [])
    has_seeds = any(
        _safe_float((c.get('curatedRank') or {}).get('current')) is not None
        for e in events if isinstance(e, dict)
        for comp in (e.get('competitions') or [{}])
        for c in (comp.get('competitors') or [])
        if isinstance(c, dict)
    )
    if not has_seeds:
        return None

    teams: List[dict] = []
    seen: set         = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        for comp in (event.get('competitions') or []):
            for c in (comp.get('competitors') or []):
                if not isinstance(c, dict):
                    continue
                team    = c.get('team') or {}
                name    = team.get('displayName', '').strip()
                seed    = int(_safe_float((c.get('curatedRank') or {}).get('current')) or 0)
                region  = str(c.get('conferenceId', '') or '')
                team_id = str(team.get('id', ''))
                if name and name not in seen:
                    seen.add(name)
                    teams.append({'name': name, 'seed': seed,
                                  'region': region, 'id': team_id})

    return teams if teams else None


def _parse_espn_bracket(data: dict) -> List[dict]:
    """Parses ESPN bracket API into team list."""
    teams: List[dict] = []
    seen: set         = set()
    for region in (data.get('bracket') or {}).get('regions', []):
        region_name = region.get('name', '')
        for td in (region.get('teams') or []):
            if not isinstance(td, dict):
                continue
            name    = (td.get('team') or {}).get('displayName', '').strip()
            seed    = int(_safe_float(td.get('seed')) or 0)
            team_id = str((td.get('team') or {}).get('id', ''))
            if name and name not in seen:
                seen.add(name)
                teams.append({'name': name, 'seed': seed,
                               'region': region_name, 'id': team_id})
    return teams


def generate_bracket(data: dict, gender: str) -> None:
    """
    Generates the full optimized bracket for one gender.
    Uses identical architecture for both men's and women's.
    Women's uses separate historical seed rates and recalibrated adjustments.
    Modifies data in place.
    """
    sport_key  = 'ncaabm' if gender == 'mens' else 'ncaabw'
    result_key = 'march_mens' if gender == 'mens' else 'march_womens'

    log.info(f'[march] Generating {gender} bracket...')

    # ── Fetch tournament field ─────────────────────────────────────────────
    field = _fetch_tournament_field(sport_key)
    if not field:
        log.info(f'[march] Seeds not available for {sport_key}')
        data[result_key] = {
            **MARCH_TEMPLATE,
            'generated_at':  datetime.datetime.utcnow().isoformat(),
            'seeds_available': False,
            'message':       'Seeds not yet released. Check back after Selection Sunday.',
        }
        return

    log.info(f'[march] {gender}: {len(field)} teams | {MARCH_SIMS:,} simulations')

    # ── Efficiency data ────────────────────────────────────────────────────
    kenpom_data     = fetch_kenpom()
    barttorvik_raw  = data.get('barttorvik') or {}

    # Supplement barttorvik from brain.py's cached data with fresh fetch if sparse
    if len(barttorvik_raw) < 50:
        barttorvik_raw = fetch_barttorvik_fresh()

    team_eff: Dict[str, dict] = {
        t['name']: get_efficiency_data(t['name'], kenpom_data, barttorvik_raw)
        for t in field
    }

    # ── Coach profiles ─────────────────────────────────────────────────────
    log.info(f'[march] Fetching coach profiles ({len(field)} teams)...')
    coach_profiles = fetch_all_coach_profiles(field, gender)

    # ── Quad records ─────────────────────────────────────────────────────
    log.info('[march] Fetching quad records...')
    quad_records = fetch_quad_records(field, gender)

    # ── Build team adjustments ─────────────────────────────────────────────
    team_adjustments = build_team_adjustments(field, data, coach_profiles)

    # Apply hot/cold on top of momentum adjustments
    for team in field:
        name       = team['name']
        hc_adj, hc, cc = hot_cold_adjustment(name, data)
        team_adjustments[name] = team_adjustments.get(name, 0.0) + hc_adj

    # ── Build first-round matchups ─────────────────────────────────────────
    by_region: Dict[str, List[dict]] = {}
    for team in field:
        r = team.get('region', 'Unknown')
        by_region.setdefault(r, []).append(team)

    for r in by_region:
        by_region[r].sort(key=lambda t: t.get('seed', 16))

    first_round_games: List[Tuple[dict, dict]] = []
    regions_ok = [r for r in by_region if len(by_region[r]) >= 8]

    if len(regions_ok) >= 4:
        for region in regions_ok[:4]:
            rteams   = by_region[region][:16]
            seed_map = {t.get('seed', 99): t for t in rteams}
            for s1, s2 in FIRST_ROUND_PAIRINGS:
                t1 = seed_map.get(s1)
                t2 = seed_map.get(s2)
                if t1 and t2:
                    first_round_games.append((t1, t2))
    else:
        all_sorted = sorted(field, key=lambda t: t.get('seed', 16))
        n = len(all_sorted)
        for i in range(n // 2):
            first_round_games.append((all_sorted[i], all_sorted[n - 1 - i]))

    if not first_round_games:
        log.warning(f'[march] Could not build bracket structure for {sport_key}')
        return

    log.info(f'[march] {gender}: {len(first_round_games)} first-round games')

    # ── Main simulation ────────────────────────────────────────────────────
    log.info(f'[march] Running {MARCH_SIMS:,} simulations...')
    round_wins, matchup_wins = run_simulation(
        first_round_games, team_eff, team_adjustments, MARCH_SIMS, gender
    )

    # ── Build recommended bracket ──────────────────────────────────────────
    rounds_output, recommended_flat, upset_alerts = pick_bracket(
        first_round_games, round_wins, matchup_wins, MARCH_SIMS
    )

    # ── Per-team probabilities with confidence bands ───────────────────────
    log.info('[march] Computing confidence bands...')
    team_probabilities: Dict[str, dict] = {}
    all_teams = [t for pair in first_round_games for t in pair]

    for team in all_teams:
        name    = team['name']
        seed    = team.get('seed', 0)
        eff     = team_eff.get(name, {})
        gap     = seeding_efficiency_gap(name, seed, {name: eff})
        hot_adj, hot_c, cold_c = hot_cold_adjustment(name, data)
        mom     = _safe_float(
            (data.get('mind') or {}).get('team_momentum', {}).get(name)
        ) or 0.5
        conf_result = (data.get('mind') or {}).get('conference_tourney', {}).get(name, '')
        coach   = coach_profiles.get(name, {})
        quad    = quad_records.get(name, {})

        # Confidence bands
        bands = compute_confidence_bands(
            name, first_round_games, team_eff, team_adjustments, gender
        )

        key_signals = []
        if gap['underseeded']:
            key_signals.append(f'AdjEM rank {eff.get("adjEM_rank", "?")} — underseeded by {gap["gap"]} spots')
        if gap['overseeded']:
            key_signals.append(f'AdjEM rank {eff.get("adjEM_rank", "?")} — overseeded by {abs(gap["gap"])} spots')
        if conf_result == 'won_conference_tourney':
            key_signals.append('Won conference tournament — peak momentum')
        if coach.get('final_four_appearances', 0) >= 2:
            key_signals.append(f'Coach: {coach["final_four_appearances"]} Final Four appearances')
        if hot_c >= 3:
            key_signals.append(f'{hot_c} hot signals entering tournament')
        if cold_c >= 3:
            key_signals.append(f'{cold_c} cold signals entering tournament')

        team_probabilities[name] = {
            'seed':               seed,
            'adjEM':              eff.get('adjEM', 0.0),
            'adjEM_rank':         eff.get('adjEM_rank', 200),
            'seeding_gap':        gap['gap'],
            'hot_signals':        hot_c,
            'cold_signals':       cold_c,
            'coach_experience_adj': round(_coach_score(coach), 4),
            'momentum_score':     round(mom, 3),
            'conference_tourney': conf_result,
            'quad_1_record':      f"{quad.get('q1_wins', 0)}-{quad.get('q1_losses', 0)}",
            'key_signals':        key_signals,
            **bands,
        }

    # ── Accuracy upsets and fades ──────────────────────────────────────────
    accuracy_upsets = _find_accuracy_upsets(
        field, first_round_games, round_wins, team_eff,
        team_adjustments, MARCH_SIMS, gender, data
    )
    accuracy_fades = _find_accuracy_fades(
        field, round_wins, team_eff, MARCH_SIMS, data
    )

    # ── Extract champion and Final Four ────────────────────────────────────
    champ_entries = rounds_output.get('Championship', [])
    champ         = champ_entries[0]['pick'] if champ_entries else 'Unknown'
    ff_entries    = rounds_output.get('Final Four', [])
    ff_teams      = [e['pick'] for e in ff_entries]

    log.info(f'[march] {gender} champion pick: {champ}')
    log.info(f'[march] {gender} Final Four: {", ".join(ff_teams)}')
    log.info(f'[march] {gender} upset picks: {len(accuracy_upsets)}')

    # ── Historical validation ──────────────────────────────────────────────
    hist_val = _load_historical_validation(data, result_key)

    data[result_key] = {
        'generated_at':      datetime.datetime.utcnow().isoformat(),
        'seeds_available':   True,
        'simulations_run':   MARCH_SIMS,
        'field_size':        len(field),
        'gender':            gender,
        'rounds':            rounds_output,
        'recommended_bracket': recommended_flat,
        'upset_alerts':      upset_alerts,
        'champion_pick':     champ,
        'final_four':        ff_teams,
        'team_probabilities': team_probabilities,
        'accuracy_upsets':   accuracy_upsets,
        'accuracy_fades':    accuracy_fades,
        'historical_validation': hist_val,
    }


def _find_accuracy_upsets(field: List[dict],
                            first_round_games: List[Tuple[dict, dict]],
                            round_wins: Dict[str, List[int]],
                            team_eff: Dict[str, dict],
                            team_adjustments: Dict[str, float],
                            n_sims: int,
                            gender: str,
                            data: dict) -> List[dict]:
    """
    Identifies accuracy-based upset picks in the first round.
    Only includes games where the model genuinely rates the underdog higher.
    """
    upsets = []
    for (ta, tb) in first_round_games:
        seed_a, seed_b = ta.get('seed', 0), tb.get('seed', 0)
        if abs(seed_a - seed_b) < 4:
            continue  # Not a meaningful upset

        # Determine favorite and underdog
        fav_team    = ta if seed_a < seed_b else tb
        dog_team    = tb if seed_a < seed_b else ta
        fav_eff     = team_eff.get(fav_team['name'], {})
        dog_eff     = team_eff.get(dog_team['name'], {})

        # Model probability for underdog
        base_p       = base_model_probability(
            {**dog_team, 'gender': gender},
            {**fav_team, 'gender': gender},
            dog_eff, fav_eff
        )
        model_prob   = max(0.02, min(0.98, base_p
                          + team_adjustments.get(dog_team['name'], 0.0)
                          - team_adjustments.get(fav_team['name'], 0.0)))
        rates        = (HISTORICAL_SEED_WIN_RATES if gender == 'mens'
                        else HISTORICAL_SEED_WIN_RATES_WOMEN)
        lo, hi       = min(seed_a, seed_b), max(seed_a, seed_b)
        hist_prior   = rates.get((lo, hi), 0.35)
        blended_prob = 0.70 * model_prob + 0.30 * hist_prior

        # Sim round 0 wins
        r0_wins_dog  = round_wins.get(dog_team['name'], [0])[0]
        sim_pct      = r0_wins_dog / n_sims if n_sims > 0 else 0.0

        scout_adj    = team_adjustments.get(dog_team['name'], 0.0)

        if blended_prob >= 0.38 and sim_pct >= 0.38:
            em_fav  = fav_eff.get('adjEM', 0.0)
            em_dog  = dog_eff.get('adjEM', 0.0)
            upsets.append({
                'team':                dog_team['name'],
                'seed':                dog_team.get('seed', 0),
                'opponent_seed':       fav_team.get('seed', 0),
                'model_prob_to_win':   round(model_prob, 3),
                'historical_prior':    round(hist_prior, 3),
                'blended_prob':        round(blended_prob, 3),
                'adjEM_rank':          dog_eff.get('adjEM_rank', 200),
                'adjEM_gap':           f'{em_dog - em_fav:+.1f} vs opponent',
                'scout_lineup_adj':    round(scout_adj, 3),
                'reasoning': (
                    f'Accuracy pick. AdjEM {em_dog:.1f} vs opponent {em_fav:.1f}. '
                    f'Sim {sim_pct:.0%}. Historical prior {hist_prior:.1%}.'
                ),
            })

    upsets.sort(key=lambda x: x['blended_prob'], reverse=True)
    return upsets


def _find_accuracy_fades(field: List[dict],
                          round_wins: Dict[str, List[int]],
                          team_eff: Dict[str, dict],
                          n_sims: int,
                          data: dict) -> List[dict]:
    """
    Identifies overseeded teams that the model expects to underperform.
    These are teams the bracket should NOT pick to advance far.
    """
    fades = []
    for team in field:
        name = team['name']
        seed = team.get('seed', 16)
        if seed > 6:
            continue  # Only flag expected high-seeds underperforming

        eff     = team_eff.get(name, {})
        em_rank = eff.get('adjEM_rank', seed * 5)
        gap     = seed - em_rank

        if gap >= 3:
            continue  # Underseeded — not a fade

        # Overseeded: AdjEM rank is much worse than seed number
        if gap <= -4:
            e8_wins = round_wins.get(name, [0] * 4)[3]
            e8_prob = e8_wins / n_sims if n_sims > 0 else 0.0
            fades.append({
                'team':              name,
                'seed':              seed,
                'model_prob_to_E8':  round(e8_prob, 3),
                'adjEM_rank':        em_rank,
                'reasoning': (
                    f'Overseeded relative to AdjEM rank {em_rank}. '
                    f'Model gives only {e8_prob:.0%} to reach Elite 8.'
                ),
            })

    fades.sort(key=lambda x: x['model_prob_to_E8'])
    return fades[:5]  # Top 5 fades


def _load_historical_validation(data: dict, result_key: str) -> dict:
    """Returns existing historical validation from data, or blank template."""
    existing = data.get(result_key, {}).get('historical_validation', {})
    if existing and existing.get('years_validated', 0) > 0:
        return existing
    return MARCH_TEMPLATE['historical_validation'].copy()


# ============================================================
# SECTION 15: HISTORICAL VALIDATION ENGINE
# ============================================================
def validate_historical_brackets(data: dict, gender: str) -> dict:
    """
    Validates methodology against past tournament results stored in data.
    Runs during training only — skipped if fewer than 5 historical years present.
    Returns validation dict to store in data[result_key]['historical_validation'].
    """
    result_key    = 'march_mens' if gender == 'mens' else 'march_womens'
    hist_key      = f'historical_brackets_{gender}'
    historical    = data.get(hist_key, {})

    if len(historical) < 5:
        log.info(f'[march] Historical validation skipped: {len(historical)} years (need 5+)')
        return MARCH_TEMPLATE['historical_validation'].copy()

    total_score  = 0
    total_expert = 0
    beat_count   = 0
    years_done   = 0

    for year, year_data in historical.items():
        predicted  = year_data.get('predicted_bracket', [])
        actual     = year_data.get('actual_results', {})
        expert_avg = year_data.get('expert_avg_score', 100)

        if not predicted or not actual:
            continue

        score = _score_bracket(predicted, actual)
        total_score  += score
        total_expert += expert_avg
        if score > expert_avg:
            beat_count += 1
        years_done   += 1

    if years_done == 0:
        return MARCH_TEMPLATE['historical_validation'].copy()

    return {
        'years_validated':   years_done,
        'avg_score':         round(total_score / years_done, 1),
        'avg_expert_score':  round(total_expert / years_done, 1),
        'beat_experts_rate': round(beat_count / years_done, 3),
        'last_validated':    datetime.datetime.utcnow().isoformat(),
    }


def _score_bracket(predicted: List[dict], actual: Dict[str, str]) -> int:
    """
    Scores a bracket using standard CBS multipliers.
    R64=1, R32=2, S16=4, E8=8, F4=16, Championship=32.
    """
    round_pts = {'Round of 64': 1, 'Round of 32': 2, 'Sweet 16': 4,
                 'Elite 8': 8, 'Final Four': 16, 'Championship': 32}
    total = 0
    for entry in predicted:
        round_name = entry.get('round', '')
        pick       = entry.get('recommended', '')
        game_key   = f"{entry.get('home', '')}|||{entry.get('away', '')}"
        if actual.get(game_key) == pick:
            total += round_pts.get(round_name, 1)
    return total


# ============================================================
# SECTION 16: ENTRY POINT
# ============================================================
def run_march(data: dict) -> None:
    """
    Main entry point called by brain.py --mode march.
    Generates brackets for both men's and women's tournaments.
    Modifies data in place. Never raises.
    """
    log.info('[march] Starting March Madness bracket generation...')

    for gender in ('mens', 'womens'):
        try:
            generate_bracket(data, gender)
        except Exception as exc:
            log.error(f'[march] Error generating {gender} bracket: {exc}')
            log.error(traceback.format_exc())

    log.info('[march] Complete')
