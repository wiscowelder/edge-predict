#!/usr/bin/env python3
"""
EDGE Predict — mind.py
Adaptive intelligence layer: Bayesian calibration + LSTM momentum modeling.
Sits above brain.py. Consumes brain.py's raw probabilities and refines them.
Never crashes brain.py — every path has a silent fallback.

Section 1  — Imports and constants
Section 2  — data.json keys managed by mind.py
Section 3  — Bayesian calibration engine
Section 4  — Isotonic regression calibrator
Section 5  — LSTM sequence model definition
Section 6  — LSTM feature engineering
Section 7  — LSTM training loop
Section 8  — LSTM inference
Section 9  — Uncertainty quantification (Monte Carlo dropout)
Section 10 — Calibration validation and reporting
Section 11 — Master refinement function (called by brain.py)
Section 12 — Training entry point
Section 13 — Update entry point
Section 14 — Persistence
"""

# ============================================================
# SECTION 1: IMPORTS AND CONSTANTS
# ============================================================
import os
import sys
import json
import math
import time
import copy
import pickle
import logging
import datetime
import traceback
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import requests

try:
    from scipy.stats import beta as beta_dist
    from scipy.special import expit as sigmoid
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

log = logging.getLogger(__name__)

MIND_MODEL_FILE = 'mind_models.pkl'
ESPN_BASE       = 'https://site.api.espn.com/apis/site/v2/sports'
ESPN_PATHS      = {
    'nfl':    'football/nfl',
    'nba':    'basketball/nba',
    'mlb':    'baseball/mlb',
    'nhl':    'hockey/nhl',
    'ncaaf':  'football/college-football',
    'ncaabm': 'basketball/mens-college-basketball',
    'ncaabw': 'basketball/womens-college-basketball',
}
ALL_SPORTS = list(ESPN_PATHS.keys())

# Sequence length per sport (number of prior games as input)
LSTM_SEQ_LEN = {
    'nfl': 10, 'nba': 15, 'mlb': 15, 'nhl': 15,
    'ncaaf': 8, 'ncaabm': 12, 'ncaabw': 12,
}
LSTM_FEATURE_DIM = 12  # Features per game in the sequence

# Momentum contribution weight per sport (max probability shift)
MOMENTUM_WEIGHT = {
    'nfl': 0.06, 'nba': 0.09, 'mlb': 0.05,
    'nhl': 0.07, 'ncaaf': 0.05, 'ncaabm': 0.08, 'ncaabw': 0.08,
}

# Margin normalization denominators per sport
MARGIN_NORM = {
    'nfl': 28, 'nba': 30, 'mlb': 5, 'nhl': 3,
    'ncaaf': 28, 'ncaabm': 25, 'ncaabw': 25,
}

# Minimum resolved predictions before Bayesian calibration activates
MIN_CALIBRATION_SAMPLES = 30

# Calibration bins (20 bins of width 0.05 from 0.0 to 1.0)
NUM_BINS     = 20
BIN_WIDTH    = 1.0 / NUM_BINS

# Context calibration minimum samples
MIN_CONTEXT_SAMPLES = 20

# Season recency weights for calibration
SEASON_WEIGHTS = {0: 1.0, 1: 0.7, 2: 0.4}
SEASON_WEIGHT_DEFAULT = 0.2

# ============================================================
# SECTION 2: DATA.JSON KEYS MANAGED BY MIND.PY
# ============================================================

def get_mind_default() -> dict:
    """Returns default structure for data['mind']."""
    return {
        'version': '1.0',
        'last_trained': None,
        'lstm_trained': False,
        'calibration_bins': {s: _empty_bins() for s in ALL_SPORTS},
        'calibration_context': {
            s: {
                'underdog':       {'alpha': 1, 'beta': 1, 'n': 0},
                'heavy_favorite': {'alpha': 1, 'beta': 1, 'n': 0},
                'home':           {'alpha': 1, 'beta': 1, 'n': 0},
                'away':           {'alpha': 1, 'beta': 1, 'n': 0},
                'division':       {'alpha': 1, 'beta': 1, 'n': 0},
            }
            for s in ALL_SPORTS
        },
        'lstm_normalization': {
            s: {'mean': [0.0] * LSTM_FEATURE_DIM, 'std': [1.0] * LSTM_FEATURE_DIM}
            for s in ALL_SPORTS
        },
        'momentum_weights': copy.deepcopy(MOMENTUM_WEIGHT),
        'performance': {
            s: {'raw_brier': None, 'calibrated_brier': None, 'improvement_pct': None}
            for s in ALL_SPORTS
        },
        'error_log': [],
        'training_sequences': {s: [] for s in ALL_SPORTS},
    }


def _empty_bins() -> dict:
    """Returns empty Bayesian calibration bins for one sport."""
    bins = []
    for i in range(NUM_BINS):
        low  = i * BIN_WIDTH
        high = low + BIN_WIDTH
        bins.append({
            'low':          low,
            'high':         high,
            'alpha':        1,   # Laplace prior
            'beta':         1,
            'n':            0,
            'season_data':  []   # list of {correct: bool, season_ago: int}
        })
    return {'bins': bins, 'n_total': 0, 'last_updated': None}


def ensure_mind_keys(data: dict) -> None:
    """Ensures data['mind'] has all required keys without overwriting existing data."""
    if 'mind' not in data or not isinstance(data.get('mind'), dict):
        data['mind'] = get_mind_default()
        return

    default = get_mind_default()
    m = data['mind']

    for k, v in default.items():
        if k not in m:
            m[k] = v

    # Ensure all sports exist in nested dicts
    for sport in ALL_SPORTS:
        if sport not in m.get('calibration_bins', {}):
            m.setdefault('calibration_bins', {})[sport] = _empty_bins()
        if sport not in m.get('calibration_context', {}):
            m.setdefault('calibration_context', {})[sport] = default['calibration_context'][sport]
        if sport not in m.get('lstm_normalization', {}):
            m.setdefault('lstm_normalization', {})[sport] = default['lstm_normalization'][sport]


# ============================================================
# SECTION 3: BAYESIAN CALIBRATION ENGINE
# ============================================================

def _get_bin_index(prob: float) -> int:
    """Returns the bin index (0–19) for a probability value."""
    idx = int(prob / BIN_WIDTH)
    return max(0, min(NUM_BINS - 1, idx))


def _compute_season_ago(pred_date: str) -> int:
    """
    Returns how many seasons ago a prediction was made.
    Approximates seasons as calendar years.
    """
    try:
        year = int(pred_date[:4])
        current_year = datetime.datetime.utcnow().year
        return max(0, current_year - year)
    except (ValueError, TypeError, IndexError):
        return 0


def _recompute_bin_from_history(bin_entry: dict) -> None:
    """
    Recomputes alpha/beta for a calibration bin using recency-weighted history.
    Applies exponential decay: older seasons matter less.
    """
    alpha = 1  # Laplace smoothing
    beta  = 1

    for record in bin_entry.get('season_data', []):
        s_ago  = record.get('season_ago', 0)
        weight = SEASON_WEIGHTS.get(s_ago, SEASON_WEIGHT_DEFAULT)
        # Weight is treated as fractional count
        if record.get('correct', False):
            alpha += weight
        else:
            beta += weight

    bin_entry['alpha'] = alpha
    bin_entry['beta']  = beta


def update_calibration_bin(data: dict, sport: str, raw_prob: float,
                            was_correct: bool, pred_date: str = '') -> None:
    """
    Adds a resolved prediction to the Bayesian calibration bins.
    Called by update_mind() for every newly verified prediction.
    """
    ensure_mind_keys(data)
    bins_data = data['mind']['calibration_bins'].get(sport)
    if not bins_data or 'bins' not in bins_data:
        return

    idx = _get_bin_index(raw_prob)
    b   = bins_data['bins'][idx]

    season_ago = _compute_season_ago(pred_date)
    b['season_data'].append({'correct': was_correct, 'season_ago': season_ago})
    # Keep last 500 records per bin to prevent unbounded growth
    if len(b['season_data']) > 500:
        b['season_data'] = b['season_data'][-500:]

    _recompute_bin_from_history(b)
    b['n'] = sum(1 for r in b['season_data'] if r.get('season_ago', 0) <= 2)

    bins_data['n_total'] = sum(bb['n'] for bb in bins_data['bins'])
    bins_data['last_updated'] = datetime.datetime.utcnow().isoformat()


def update_context_calibration(data: dict, sport: str, raw_prob: float,
                                was_correct: bool, context: str) -> None:
    """Updates context-specific calibration (underdog, heavy_favorite, home, etc.)."""
    ensure_mind_keys(data)
    ctx = data['mind']['calibration_context'].get(sport, {}).get(context)
    if not ctx:
        return

    if was_correct:
        ctx['alpha'] += 1
    else:
        ctx['beta'] += 1
    ctx['n'] = ctx.get('n', 0) + 1


def calibrate_probability(data: dict, sport: str, raw_prob: float,
                           context_flags: Optional[dict] = None) -> Tuple[float, float, float, bool]:
    """
    Applies Bayesian calibration to a raw probability.

    Returns:
        calibrated_prob (float): calibrated probability
        lower_bound (float): 5th percentile uncertainty
        upper_bound (float): 95th percentile uncertainty
        insufficient_data (bool): True if calibration had insufficient data
    """
    ensure_mind_keys(data)
    bins_data = data['mind']['calibration_bins'].get(sport)

    if not bins_data or bins_data.get('n_total', 0) < MIN_CALIBRATION_SAMPLES:
        # Not enough data — return raw with wide uncertainty
        uncertainty = 0.20
        lower = max(0.02, raw_prob - uncertainty)
        upper = min(0.98, raw_prob + uncertainty)
        return raw_prob, lower, upper, True

    idx = _get_bin_index(raw_prob)
    b   = bins_data['bins'][idx]
    n   = b.get('n', 0)

    if n < MIN_CALIBRATION_SAMPLES:
        # Bin-level insufficient — return raw with moderate uncertainty
        uncertainty = max(0.10, 0.20 - n * 0.003)
        lower = max(0.02, raw_prob - uncertainty)
        upper = min(0.98, raw_prob + uncertainty)
        return raw_prob, lower, upper, True

    alpha = b.get('alpha', 1)
    beta_val = b.get('beta', 1)

    # Bayesian posterior mean
    bin_calibrated = alpha / (alpha + beta_val)

    # Context calibration blend (30% weight once 20+ samples)
    context_calibrated = bin_calibrated
    if context_flags and SCIPY_AVAILABLE:
        ctx_store = data['mind']['calibration_context'].get(sport, {})
        blend_total = 0.0
        blend_weight = 0.0

        for ctx_name, is_active in context_flags.items():
            if not is_active:
                continue
            ctx = ctx_store.get(ctx_name, {})
            if ctx.get('n', 0) >= MIN_CONTEXT_SAMPLES:
                ca = ctx.get('alpha', 1)
                cb = ctx.get('beta', 1)
                ctx_prob = ca / (ca + cb)
                blend_total  += ctx_prob * 0.3
                blend_weight += 0.3

        if blend_weight > 0:
            bin_weight = 1.0 - blend_weight
            context_calibrated = bin_calibrated * bin_weight + blend_total

    calibrated = max(0.02, min(0.98, context_calibrated))

    # Uncertainty from Beta distribution
    if SCIPY_AVAILABLE:
        lower = float(beta_dist.ppf(0.05, alpha, beta_val))
        upper = float(beta_dist.ppf(0.95, alpha, beta_val))
    else:
        # Fallback: Wilson-style approximation
        p = alpha / (alpha + beta_val)
        n_eff = alpha + beta_val - 2
        z = 1.645  # 90% CI
        uncertainty = z * math.sqrt(p * (1 - p) / max(n_eff, 1))
        lower = max(0.02, p - uncertainty)
        upper = min(0.98, p + uncertainty)

    return calibrated, float(lower), float(upper), False


# ============================================================
# SECTION 4: ISOTONIC REGRESSION CALIBRATOR (fallback)
# ============================================================

def isotonic_calibrate(raw_probs: List[float], actuals: List[int]) -> Optional[Any]:
    """
    Trains isotonic regression as an alternative to Bayesian bins.
    Used when sklearn is available and Bayesian bins are sparse.
    Returns fitted IsotonicRegression or None.
    """
    if not SKLEARN_AVAILABLE or len(raw_probs) < 50:
        return None
    try:
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(raw_probs, actuals)
        return ir
    except Exception:
        return None


# ============================================================
# SECTION 5: LSTM SEQUENCE MODEL DEFINITION
# ============================================================

if TORCH_AVAILABLE:
    class MomentumLSTM(nn.Module):
        """
        LSTM model that reads a sequence of recent games and outputs
        a momentum score [0, 1] for the team.
        0.5 = neutral, >0.5 = positive momentum, <0.5 = negative.
        Architecture: LSTM(12→64, 2 layers) → Linear(64→32) → Linear(32→1) → Sigmoid
        """
        def __init__(self, input_size: int = 12, hidden_size: int = 64,
                     num_layers: int = 2, dropout: float = 0.3):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers  = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=False,
            )
            self.drop1 = nn.Dropout(dropout)
            self.fc1   = nn.Linear(hidden_size, 32)
            self.relu  = nn.ReLU()
            self.drop2 = nn.Dropout(0.2)
            self.fc2   = nn.Linear(32, 1)
            self.sig   = nn.Sigmoid()

        def forward(self, x):
            # x: [batch, seq_len, input_size]
            out, _ = self.lstm(x)
            # Take last timestep
            last = out[:, -1, :]
            last = self.drop1(last)
            last = self.relu(self.fc1(last))
            last = self.drop2(last)
            last = self.sig(self.fc2(last))
            return last.squeeze(-1)


# ============================================================
# SECTION 6: LSTM FEATURE ENGINEERING
# ============================================================

def _build_game_feature_vector(game_entry: dict, sport: str,
                                elo_ratings: Optional[dict] = None,
                                season_length: int = 30) -> List[float]:
    """
    Converts a single game history entry into a 12-dimensional feature vector.
    game_entry has: date, win, margin, is_home, score_for, score_against
    Returns list of 12 floats (un-normalized).
    """
    margin_norm = MARGIN_NORM.get(sport, 14)

    f0  = 1.0 if game_entry.get('win', False) else 0.0
    f1  = max(-1.0, min(1.0, game_entry.get('margin', 0) / margin_norm))
    f2  = 1.0 if game_entry.get('is_home', False) else 0.0

    # Opponent Elo: if not available, use 0.5 (neutral)
    f3  = 0.5  # placeholder — would need opponent name to look up

    # Days rest: normalize 0=back-to-back, 1=7+ days
    rest = game_entry.get('rest_days', 4)
    f4   = min(1.0, rest / 7.0) if isinstance(rest, (int, float)) else 0.5

    f5  = 1.0 if game_entry.get('is_playoff', False) else 0.0

    # Offensive efficiency: points scored / typical range
    typical_pts = {'nfl': 24, 'nba': 110, 'mlb': 4.5, 'nhl': 3.0,
                   'ncaaf': 30, 'ncaabm': 75, 'ncaabw': 65}
    typ = typical_pts.get(sport, 50)
    score_for = game_entry.get('score_for', 0) or 0
    f6 = min(2.0, score_for / typ) if typ > 0 else 0.5

    score_against = game_entry.get('score_against', 0) or 0
    f7 = min(2.0, score_against / typ) if typ > 0 else 0.5

    # Turnover proxy: close-game indicator
    close_thresholds = {'nfl': 8, 'nba': 6, 'mlb': 2, 'nhl': 1,
                        'ncaaf': 8, 'ncaabm': 6, 'ncaabw': 6}
    close_thresh = close_thresholds.get(sport, 6)
    margin_abs = abs(game_entry.get('margin', 0))
    f8 = 1.0 if margin_abs <= close_thresh else 0.0

    f9  = 1.0 if game_entry.get('is_division', False) else 0.0

    # Season progress (estimated from date)
    f10 = game_entry.get('season_progress', 0.5)

    # Injury impact (if available, else 0)
    f11 = max(-1.0, min(1.0, game_entry.get('injury_impact', 0.0)))

    return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]


def build_sequences_from_history(team_history: dict, sport: str) -> Tuple[List, List]:
    """
    Converts team_history[sport] into LSTM training sequences.
    Each sample: X = sequence of N prior games, y = did team win next game?

    Returns:
        X_list: list of sequences (each sequence is list of feature vectors)
        y_list: list of labels (1=win, 0=loss)
    """
    seq_len = LSTM_SEQ_LEN.get(sport, 10)
    sport_history = team_history.get(sport, {})

    X_list = []
    y_list = []

    if not isinstance(sport_history, dict):
        return X_list, y_list

    for team, games in sport_history.items():
        if not isinstance(games, list) or len(games) <= seq_len:
            continue

        # Sort by date
        sorted_games = sorted(
            [g for g in games if isinstance(g, dict) and g.get('date')],
            key=lambda g: g.get('date', '')
        )

        if len(sorted_games) <= seq_len:
            continue

        # Slide window: for each position after seq_len, build one sample
        for i in range(seq_len, len(sorted_games)):
            seq_games = sorted_games[i - seq_len: i]
            target_game = sorted_games[i]

            # Build feature vectors for sequence
            seq_features = []
            for j, g in enumerate(seq_games):
                fv = _build_game_feature_vector(g, sport)
                seq_features.append(fv)

            X_list.append(seq_features)
            y_list.append(1.0 if target_game.get('win', False) else 0.0)

    return X_list, y_list


def compute_normalization_params(X_list: List) -> Tuple[List[float], List[float]]:
    """
    Computes per-feature mean and std across all sequences.
    Returns (means, stds), each of length LSTM_FEATURE_DIM.
    """
    if not X_list:
        return [0.0] * LSTM_FEATURE_DIM, [1.0] * LSTM_FEATURE_DIM

    # Flatten all timesteps
    all_vecs = []
    for seq in X_list:
        for vec in seq:
            if len(vec) == LSTM_FEATURE_DIM:
                all_vecs.append(vec)

    if not all_vecs:
        return [0.0] * LSTM_FEATURE_DIM, [1.0] * LSTM_FEATURE_DIM

    arr  = np.array(all_vecs, dtype=np.float32)
    mean = arr.mean(axis=0).tolist()
    std  = arr.std(axis=0)
    # Avoid division by zero
    std  = [max(s, 1e-8) for s in std.tolist()]

    return mean, std


def normalize_sequences(X_list: List, mean: List[float], std: List[float]) -> List:
    """Applies z-score normalization to all feature vectors in X_list."""
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr  = np.array(std,  dtype=np.float32)

    normalized = []
    for seq in X_list:
        norm_seq = []
        for vec in seq:
            v = np.array(vec, dtype=np.float32)
            norm_seq.append(((v - mean_arr) / std_arr).tolist())
        normalized.append(norm_seq)

    return normalized


def build_team_sequence(team: str, sport: str, team_history: dict,
                         mean: List[float], std: List[float]) -> Optional[List]:
    """
    Builds a normalized feature sequence for a team from team_history.
    Used during inference. Returns None if insufficient history.
    """
    seq_len = LSTM_SEQ_LEN.get(sport, 10)
    sport_history = team_history.get(sport, {})
    if not isinstance(sport_history, dict):
        return None

    games = sport_history.get(team)
    if not games or not isinstance(games, list):
        return None

    sorted_games = sorted(
        [g for g in games if isinstance(g, dict) and g.get('date')],
        key=lambda g: g.get('date', '')
    )

    if len(sorted_games) < 3:
        return None

    # Use last seq_len games (or all if fewer)
    recent = sorted_games[-seq_len:]

    # Pad if too short
    while len(recent) < seq_len:
        recent.insert(0, {'win': False, 'margin': 0, 'is_home': False,
                          'score_for': 0, 'score_against': 0})

    mean_arr = np.array(mean, dtype=np.float32)
    std_arr  = np.array(std,  dtype=np.float32)

    seq = []
    for g in recent:
        fv = np.array(_build_game_feature_vector(g, sport), dtype=np.float32)
        seq.append(((fv - mean_arr) / std_arr).tolist())

    return seq


# ============================================================
# SECTION 7: LSTM TRAINING LOOP
# ============================================================

def _fetch_espn_season_simple(sport: str, year: int) -> List[dict]:
    """
    Minimal ESPN season fetcher for training data.
    Returns list of game dicts with: home, away, date, home_won, home_score, away_score.
    """
    path = ESPN_PATHS.get(sport, '')
    if not path:
        return []

    games = []
    try:
        url = f'{ESPN_BASE}/{path}/scoreboard'
        params = {'dates': str(year), 'limit': '900'}
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            return []
        data = resp.json()
        for event in data.get('events', []):
            comps = event.get('competitions', [])
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get('competitors', [])
            if len(competitors) < 2:
                continue
            home_c = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            away_c = next((c for c in competitors if c.get('homeAway') == 'away'), None)
            if not home_c or not away_c:
                continue
            hs = int(home_c.get('score', 0) or 0)
            as_ = int(away_c.get('score', 0) or 0)
            status = (event.get('status') or {}).get('type', {}).get('completed', False)
            if not status:
                continue
            home_name = (home_c.get('team') or {}).get('displayName', '')
            away_name = (away_c.get('team') or {}).get('displayName', '')
            date_str  = (event.get('date') or '')[:10]
            if not home_name or not away_name or not date_str:
                continue
            games.append({
                'home': home_name, 'away': away_name,
                'date': date_str, 'home_score': hs, 'away_score': as_,
                'home_won': hs > as_,
            })
    except Exception:
        pass

    return games


def _build_history_from_games(games: List[dict], sport: str) -> dict:
    """
    Builds a full team_history dict from a list of games.
    No pruning — stores all games for LSTM training.
    """
    history: Dict[str, list] = {}

    sorted_games = sorted(
        [g for g in games if isinstance(g, dict) and g.get('home') and g.get('date')],
        key=lambda g: g.get('date', '')
    )

    for i, game in enumerate(sorted_games):
        home = game['home']
        away = game['away']
        date = game['date']
        hs   = game.get('home_score', 0) or 0
        as_  = game.get('away_score', 0) or 0
        home_won = bool(game.get('home_won', hs > as_))

        season_total = len([g for g in sorted_games if g['date'][:4] == date[:4]])
        season_pos   = sum(1 for g in sorted_games if g['home'] == home
                           and g['date'][:4] == date[:4] and g['date'] <= date)
        season_prog  = season_pos / max(season_total, 1)

        home_entry = {
            'date': date, 'win': home_won, 'margin': hs - as_,
            'is_home': True, 'score_for': hs, 'score_against': as_,
            'season_progress': season_prog,
        }
        away_entry = {
            'date': date, 'win': not home_won, 'margin': as_ - hs,
            'is_home': False, 'score_for': as_, 'score_against': hs,
            'season_progress': season_prog,
        }

        history.setdefault(home, []).append(home_entry)
        history.setdefault(away, []).append(away_entry)

    return history


def train_lstm_for_sport(X_list: List, y_list: List, sport: str) -> Optional[Any]:
    """
    Trains an LSTM model for one sport.
    Returns trained model or None if training fails.
    """
    if not TORCH_AVAILABLE:
        return None

    if len(X_list) < 50:
        log.info(f'[mind] {sport}: insufficient training samples ({len(X_list)}) — skipping LSTM')
        return None

    seq_len = LSTM_SEQ_LEN.get(sport, 10)

    try:
        # Convert to tensors
        X_np = np.array(X_list, dtype=np.float32)   # [N, seq_len, 12]
        y_np = np.array(y_list, dtype=np.float32)   # [N]

        # Train/val split (80/20, chronological)
        n_val   = max(10, int(len(X_list) * 0.2))
        n_train = len(X_list) - n_val

        X_train = torch.tensor(X_np[:n_train])
        y_train = torch.tensor(y_np[:n_train])
        X_val   = torch.tensor(X_np[n_train:])
        y_val   = torch.tensor(y_np[n_train:])

        dataset = TensorDataset(X_train, y_train)
        loader  = DataLoader(dataset, batch_size=64, shuffle=True)

        model = MomentumLSTM(
            input_size=LSTM_FEATURE_DIM,
            hidden_size=64,
            num_layers=2,
            dropout=0.3
        )
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.BCELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, mode='min'
        )

        best_val_loss = float('inf')
        patience_count = 0
        MAX_EPOCHS = 100
        EARLY_STOP = 10

        for epoch in range(MAX_EPOCHS):
            model.train()
            for Xb, yb in loader:
                optimizer.zero_grad()
                pred = model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss - 0.001:
                best_val_loss = val_loss
                patience_count = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_count += 1

            if patience_count >= EARLY_STOP:
                log.info(f'[mind] {sport} LSTM: early stop at epoch {epoch+1}')
                break

        model.load_state_dict(best_state)
        log.info(f'[mind] {sport} LSTM trained: {len(X_list)} samples, val_loss={best_val_loss:.4f}')
        return model

    except Exception as exc:
        log.error(f'[mind] LSTM training failed for {sport}: {exc}')
        return None


# ============================================================
# SECTION 8: LSTM INFERENCE
# ============================================================

_LSTM_MODELS: Dict[str, Any] = {}


def _load_lstm_models() -> None:
    """Loads pre-trained LSTM models from mind_models.pkl into memory."""
    global _LSTM_MODELS
    if not os.path.exists(MIND_MODEL_FILE):
        return
    try:
        with open(MIND_MODEL_FILE, 'rb') as f:
            pkg = pickle.load(f)

        if not TORCH_AVAILABLE:
            return

        for sport, state_dict in pkg.get('lstm_models', {}).items():
            try:
                arch = pkg.get('lstm_architectures', {}).get(sport, {})
                model = MomentumLSTM(
                    input_size=arch.get('input_size', LSTM_FEATURE_DIM),
                    hidden_size=arch.get('hidden_size', 64),
                    num_layers=arch.get('num_layers', 2),
                    dropout=arch.get('dropout', 0.3),
                )
                model.load_state_dict(state_dict)
                model.eval()
                _LSTM_MODELS[sport] = model
            except Exception:
                pass

        log.info(f'[mind] LSTM models loaded: {list(_LSTM_MODELS.keys())}')
    except Exception as exc:
        log.warning(f'[mind] Could not load mind_models.pkl: {exc}')


def infer_momentum(team: str, sport: str, data: dict) -> Tuple[float, float]:
    """
    Runs LSTM inference for a team.
    Returns (momentum_score, uncertainty).
    momentum_score: 0.5 = neutral, >0.5 = positive momentum.
    uncertainty: std dev from Monte Carlo dropout.
    """
    if not TORCH_AVAILABLE or sport not in _LSTM_MODELS:
        return 0.5, 0.0

    model = _LSTM_MODELS[sport]
    if model is None:
        return 0.5, 0.0

    mean = data['mind']['lstm_normalization'].get(sport, {}).get('mean', [0.0] * LSTM_FEATURE_DIM)
    std  = data['mind']['lstm_normalization'].get(sport, {}).get('std',  [1.0] * LSTM_FEATURE_DIM)

    seq = build_team_sequence(team, sport, data.get('team_history', {}), mean, std)
    if seq is None:
        return 0.5, 0.0

    try:
        x = torch.tensor([seq], dtype=torch.float32)  # [1, seq_len, 12]
        # Monte Carlo Dropout: 50 passes
        scores = run_mc_dropout(model, x, n_passes=50)
        mean_score = float(np.mean(scores))
        std_score  = float(np.std(scores))
        return float(np.clip(mean_score, 0.0, 1.0)), std_score
    except Exception as exc:
        log.debug(f'[mind] LSTM inference failed for {team}/{sport}: {exc}')
        return 0.5, 0.0


# ============================================================
# SECTION 9: UNCERTAINTY QUANTIFICATION (MONTE CARLO DROPOUT)
# ============================================================

def run_mc_dropout(model: Any, x: Any, n_passes: int = 50) -> List[float]:
    """
    Runs n_passes forward passes with dropout active.
    Returns list of output scalars.
    """
    scores = []
    model.train()  # Activates dropout
    with torch.no_grad():
        for _ in range(n_passes):
            out = model(x)
            scores.append(float(out.item()))
    model.eval()
    return scores


# ============================================================
# SECTION 10: CALIBRATION VALIDATION
# ============================================================

def compute_brier_score(predictions: List[dict]) -> float:
    """Computes Brier score from a list of {prob, outcome} dicts."""
    if not predictions:
        return 0.5
    total = 0.0
    for p in predictions:
        prob    = float(p.get('prob', 0.5))
        outcome = float(p.get('outcome', 0))
        total  += (prob - outcome) ** 2
    return total / len(predictions)


def validate_calibration(data: dict, sport: str) -> dict:
    """
    Computes Brier score improvement from raw → calibrated probabilities.
    Returns dict with raw_brier, calibrated_brier, improvement_pct.
    """
    preds = data.get('predictions', {}).get(sport, [])
    if not isinstance(preds, list):
        return {}

    resolved = [
        p for p in preds
        if isinstance(p, dict)
        and p.get('status') in ('correct', 'wrong')
        and p.get('raw_win_probability') is not None
    ]

    if len(resolved) < 20:
        return {}

    raw_samples = []
    cal_samples = []

    for p in resolved:
        outcome = 1.0 if p.get('actual_result') == 'home_won' else 0.0
        raw_p   = float(p.get('raw_win_probability', 0.5))
        ref_p   = float(p.get('win_probability', raw_p))

        raw_samples.append({'prob': raw_p, 'outcome': outcome})
        cal_samples.append({'prob': ref_p, 'outcome': outcome})

    raw_brier = compute_brier_score(raw_samples)
    cal_brier = compute_brier_score(cal_samples)
    imp_pct   = (raw_brier - cal_brier) / raw_brier * 100 if raw_brier > 0 else 0.0

    return {
        'raw_brier':        raw_brier,
        'calibrated_brier': cal_brier,
        'improvement_pct':  imp_pct,
    }


# ============================================================
# SECTION 11: MASTER REFINEMENT FUNCTION
# ============================================================

def refine_prediction(game_dict: dict, raw_prob: float,
                       data: dict, sport: str) -> dict:
    """
    Master refinement function called by brain.py for every game prediction.

    Parameters:
        game_dict: the game object being predicted
        raw_prob:  brain.py's ensemble win probability for home team [0,1]
        data:      full data.json dict (read/write)
        sport:     string, one of the 7 sports

    Returns dict with:
        refined_prob, confidence_lower, confidence_upper,
        momentum_score, momentum_uncertainty, calibration_correction, flags
    """
    flags = {}
    momentum_score       = 0.5
    momentum_uncertainty = 0.0
    calibration_correction = 0.0

    ensure_mind_keys(data)

    # --- Step 1: Load LSTM models if not yet in memory ---
    if not _LSTM_MODELS:
        _load_lstm_models()

    # --- Step 2: LSTM momentum adjustment ---
    prob_after_momentum = raw_prob

    if TORCH_AVAILABLE and sport in _LSTM_MODELS:
        home = game_dict.get('home', '')
        away = game_dict.get('away', '')

        home_momentum, home_unc = infer_momentum(home, sport, data)
        away_momentum, away_unc = infer_momentum(away, sport, data)

        # Net momentum: home advantage over away
        net_momentum = home_momentum - away_momentum  # range ~[-0.5, 0.5]
        weight       = data['mind'].get('momentum_weights', MOMENTUM_WEIGHT).get(sport, 0.06)

        # High uncertainty = reduce weight
        avg_unc = (home_unc + away_unc) / 2
        if avg_unc > 0.15:
            weight *= 0.5

        momentum_delta      = net_momentum * weight
        prob_after_momentum = max(0.02, min(0.98, raw_prob + momentum_delta))
        momentum_score      = float(home_momentum)
        momentum_uncertainty = float(home_unc)
    else:
        flags['lstm_unavailable'] = True

    # --- Step 3: Bayesian calibration ---
    home_prob   = game_dict.get('home', '')
    is_underdog = raw_prob < 0.45
    is_favorite = raw_prob > 0.80
    is_home     = True  # We always predict from home team perspective

    context_flags = {
        'underdog':       is_underdog,
        'heavy_favorite': is_favorite,
        'home':           is_home,
        'away':           False,
        'division':       game_dict.get('is_division', False),
    }

    calibrated, lower, upper, insufficient = calibrate_probability(
        data, sport, prob_after_momentum, context_flags
    )

    if insufficient:
        flags['calibration_insufficient_data'] = True

    calibration_correction = float(calibrated - raw_prob)
    refined_prob = float(max(0.02, min(0.98, calibrated)))

    return {
        'refined_prob':            refined_prob,
        'confidence_lower':        float(lower),
        'confidence_upper':        float(upper),
        'momentum_score':          momentum_score,
        'momentum_uncertainty':    momentum_uncertainty,
        'calibration_correction':  calibration_correction,
        'flags':                   flags,
    }


# ============================================================
# SECTION 12: TRAINING ENTRY POINT
# ============================================================

def train_mind(data: dict) -> None:
    """
    Full mind.py training run.
    1. Fetches 2 historical seasons per sport from ESPN
    2. Builds LSTM training sequences
    3. Trains 7 LSTM models (one per sport)
    4. Computes normalization parameters
    5. Saves models to mind_models.pkl
    6. Initializes Bayesian calibration bins (empty — fills during updates)

    Called by brain.py --mode train after its own training completes.
    """
    ensure_mind_keys(data)
    log.info('[mind] Starting training...')

    lstm_models       = {}
    lstm_architectures = {}
    val_accuracy      = {}

    current_year = datetime.datetime.utcnow().year

    for sport in ALL_SPORTS:
        log.info(f'[mind] Training LSTM for {sport}...')

        # Fetch last 2 seasons from ESPN for training data
        all_games = []
        for y_offset in [2, 1]:
            year = current_year - y_offset
            log.info(f'[mind]   Fetching {sport} {year}...')
            games = _fetch_espn_season_simple(sport, year)
            all_games.extend(games)
            time.sleep(0.5)

        if not all_games:
            log.info(f'[mind]   {sport}: no historical games fetched — LSTM skipped')
            continue

        # Build full history (unpruned) from fetched games
        full_history = _build_history_from_games(all_games, sport)
        combined_history = {sport: full_history}

        # Build sequences
        X_list, y_list = build_sequences_from_history(combined_history, sport)

        if len(X_list) < 20:
            log.info(f'[mind]   {sport}: only {len(X_list)} sequences — LSTM skipped')
            continue

        # Compute and store normalization params
        mean, std = compute_normalization_params(X_list)
        data['mind']['lstm_normalization'][sport] = {'mean': mean, 'std': std}

        # Normalize
        X_norm = normalize_sequences(X_list, mean, std)

        # Train
        if TORCH_AVAILABLE:
            model = train_lstm_for_sport(X_norm, y_list, sport)
            if model is not None:
                lstm_models[sport] = model.state_dict()
                lstm_architectures[sport] = {
                    'input_size': LSTM_FEATURE_DIM,
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.3,
                }
                val_accuracy[sport] = 0.0  # Will be computed on first update
                _LSTM_MODELS[sport] = model
        else:
            log.info(f'[mind]   PyTorch not available — Bayesian-only for {sport}')

    # Save models
    if lstm_models:
        save_mind_models(lstm_models, lstm_architectures, val_accuracy)
        data['mind']['lstm_trained'] = True
        log.info(f'[mind] LSTM models saved: {list(lstm_models.keys())}')
    else:
        log.info('[mind] No LSTM models trained (Bayesian-only mode active)')

    data['mind']['last_trained'] = datetime.datetime.utcnow().isoformat()
    log.info('[mind] Training complete.')


# ============================================================
# SECTION 13: UPDATE ENTRY POINT
# ============================================================

def update_mind(data: dict) -> None:
    """
    Daily update run.
    1. Processes newly resolved predictions and updates calibration bins
    2. Updates context calibration
    3. Fine-tunes LSTM on new games (if enough new samples)
    4. Recomputes performance metrics

    Called by brain.py --mode update after predictions are generated.
    """
    ensure_mind_keys(data)
    log.info('[mind] Updating calibration...')

    new_samples_by_sport: Dict[str, int] = {s: 0 for s in ALL_SPORTS}

    # Load LSTM models if not in memory
    if not _LSTM_MODELS:
        _load_lstm_models()

    for sport in ALL_SPORTS:
        preds = data.get('predictions', {}).get(sport, [])
        if not isinstance(preds, list):
            continue

        for pred in preds:
            if not isinstance(pred, dict):
                continue

            # Only process newly resolved, not yet mind-processed
            if pred.get('status') not in ('correct', 'wrong'):
                continue
            if pred.get('mind_calibration_updated'):
                continue

            raw_prob = pred.get('raw_win_probability') or pred.get('home_prob', 0.5)
            raw_prob = float(raw_prob)
            was_correct = pred.get('status') == 'correct'

            # Infer actual outcome direction
            pick_is_home = pred.get('pick') == pred.get('home', '')
            actual_home_won = pred.get('actual_result') == 'home_won'
            home_was_correct = pick_is_home == actual_home_won

            pred_date = pred.get('date', '')

            # Update Bayesian bins
            update_calibration_bin(data, sport, raw_prob, was_correct, pred_date)

            # Update context calibration
            if raw_prob < 0.45:
                update_context_calibration(data, sport, raw_prob, was_correct, 'underdog')
            elif raw_prob > 0.80:
                update_context_calibration(data, sport, raw_prob, was_correct, 'heavy_favorite')

            update_context_calibration(data, sport, raw_prob, was_correct, 'home')

            # Mark as processed
            pred['mind_calibration_updated'] = True
            new_samples_by_sport[sport] += 1

        # Update performance metrics
        perf = validate_calibration(data, sport)
        if perf:
            data['mind']['performance'][sport] = perf
            log.info(f'[mind] {sport} Brier improvement: {perf.get("improvement_pct", 0):.1f}%')

    total_new = sum(new_samples_by_sport.values())
    log.info(f'[mind] Calibration updated: {total_new} new resolved predictions')

    # Fine-tune LSTM if we have meaningful new data
    if TORCH_AVAILABLE and total_new >= 10:
        _finetune_lstm(data, new_samples_by_sport)


def _finetune_lstm(data: dict, new_samples_by_sport: Dict[str, int]) -> None:
    """Fine-tunes LSTM models on new resolved games."""
    updated_models = {}
    need_save = False

    for sport, n_new in new_samples_by_sport.items():
        if n_new < 5:
            continue
        if sport not in _LSTM_MODELS:
            continue

        X_list, y_list = build_sequences_from_history(data.get('team_history', {}), sport)
        if len(X_list) < 20:
            continue

        mean = data['mind']['lstm_normalization'].get(sport, {}).get('mean', [0.0] * LSTM_FEATURE_DIM)
        std  = data['mind']['lstm_normalization'].get(sport, {}).get('std',  [1.0] * LSTM_FEATURE_DIM)
        X_norm = normalize_sequences(X_list, mean, std)

        try:
            model = _LSTM_MODELS[sport]
            model.train()

            X_t = torch.tensor(np.array(X_norm, dtype=np.float32))
            y_t = torch.tensor(np.array(y_list, dtype=np.float32))

            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            criterion = nn.BCELoss()

            for _ in range(5):  # 5 fine-tune epochs
                optimizer.zero_grad()
                pred = model(X_t)
                loss = criterion(pred, y_t)
                loss.backward()
                optimizer.step()

            model.eval()
            updated_models[sport] = model.state_dict()
            need_save = True
            log.info(f'[mind] {sport} LSTM fine-tuned on {len(X_list)} sequences')
        except Exception as exc:
            log.debug(f'[mind] Fine-tune failed for {sport}: {exc}')

    if need_save and updated_models:
        try:
            if os.path.exists(MIND_MODEL_FILE):
                with open(MIND_MODEL_FILE, 'rb') as f:
                    pkg = pickle.load(f)
            else:
                pkg = {'lstm_models': {}, 'lstm_architectures': {}, 'validation_accuracy': {}}

            pkg['lstm_models'].update(updated_models)
            save_mind_models(
                pkg['lstm_models'],
                pkg.get('lstm_architectures', {}),
                pkg.get('validation_accuracy', {}),
            )
        except Exception as exc:
            log.warning(f'[mind] Could not save fine-tuned models: {exc}')


# ============================================================
# SECTION 14: PERSISTENCE
# ============================================================

def save_mind_models(lstm_models: dict, architectures: dict, val_accuracy: dict) -> None:
    """Saves LSTM models to mind_models.pkl."""
    pkg = {
        'lstm_models':        lstm_models,
        'lstm_architectures': architectures,
        'trained_date':       datetime.datetime.utcnow().isoformat(),
        'validation_accuracy': val_accuracy,
    }
    try:
        with open(MIND_MODEL_FILE, 'wb') as f:
            pickle.dump(pkg, f, protocol=4)
        log.info(f'[mind] Models saved to {MIND_MODEL_FILE}')
    except Exception as exc:
        log.error(f'[mind] Failed to save models: {exc}')
