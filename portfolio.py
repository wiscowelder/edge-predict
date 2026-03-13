"""
portfolio.py — EDGE Predict Portfolio Kelly Optimizer
Sizes simultaneous bets as a correlated portfolio using multi-asset Kelly Criterion.
ONLY relevant for actual sports betting — irrelevant for CBS pool and March bracket.
Called by: brain.py --mode update (if enabled)
"""

import numpy as np
from datetime import datetime

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# Section 1 — Constants & Hard Rules
# ─────────────────────────────────────────────────────────────

MAX_SINGLE_BET       = 0.05    # Never > 5% of bankroll on one game
MAX_TOTAL_EXPOSURE   = 0.25    # Never > 25% at risk simultaneously
MIN_SYSTEM_ACCURACY  = 0.54    # Don't bet if condition accuracy is below this
MAX_CONFIDENCE_WIDTH = 0.35    # Don't bet if Bayesian bounds are too wide
QUARTER_KELLY        = 0.25    # Use quarter-Kelly for safety
MIN_EDGE             = 0.005   # Minimum edge to place a bet (0.5%)


# ─────────────────────────────────────────────────────────────
# Section 2 — Kelly Criterion (Single Bet)
# ─────────────────────────────────────────────────────────────

def kelly_fraction(win_prob, odds_decimal):
    """
    Quarter-Kelly fraction of bankroll to bet.
    win_prob: system's estimated win probability
    odds_decimal: decimal odds (e.g., 1.91 for -110 American)
    Returns: fraction of bankroll (0 to MAX_SINGLE_BET)
    """
    if odds_decimal <= 1.0:
        return 0.0
    b = odds_decimal - 1.0   # Net profit per unit
    q = 1.0 - win_prob

    full_kelly = (b * win_prob - q) / b
    quarter_k  = full_kelly * QUARTER_KELLY

    return max(0.0, min(MAX_SINGLE_BET, quarter_k))


def american_to_decimal(american_odds):
    """Converts American odds to decimal odds."""
    american_odds = float(american_odds)
    if american_odds > 0:
        return 1.0 + (american_odds / 100.0)
    else:
        return 1.0 + (100.0 / abs(american_odds))


def implied_probability(american_odds):
    """Converts American moneyline to implied probability (vig-inclusive)."""
    american_odds = float(american_odds)
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)


def edge(win_prob, american_odds):
    """Expected value edge: system prob - implied market prob."""
    return float(win_prob) - implied_probability(american_odds)


# ─────────────────────────────────────────────────────────────
# Section 3 — Bet Correlation Matrix
# ─────────────────────────────────────────────────────────────

def bet_correlation(bet_a, bet_b):
    """
    Estimates correlation between two bets.
    High correlation = both likely to win/lose together.
    """
    correlations = []

    # Same sport same day
    if bet_a.get("sport") == bet_b.get("sport") and bet_a.get("date") == bet_b.get("date"):
        correlations.append(0.15)

    # Both home favorites
    if bet_a.get("pick_type") == "home_favorite" and bet_b.get("pick_type") == "home_favorite":
        correlations.append(0.10)

    # Both driven by the same primary signal
    if (bet_a.get("primary_signal") and
            bet_a.get("primary_signal") == bet_b.get("primary_signal")):
        correlations.append(0.12)

    # Same weather zone (outdoor sports)
    if (bet_a.get("weather_zone") and
            bet_a.get("weather_zone") == bet_b.get("weather_zone")):
        correlations.append(0.08)

    # Both on heavy favorites (similar error patterns)
    if (float(bet_a.get("win_probability", 0.5)) > 0.72 and
            float(bet_b.get("win_probability", 0.5)) > 0.72):
        correlations.append(0.06)

    return min(0.5, sum(correlations))


def build_correlation_matrix(bets):
    """Builds n×n correlation matrix for a list of active bets."""
    n = len(bets)
    C = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = bet_correlation(bets[i], bets[j])
            C[i, j] = c
            C[j, i] = c
    return C


# ─────────────────────────────────────────────────────────────
# Section 4 — Portfolio Kelly Optimizer
# ─────────────────────────────────────────────────────────────

def portfolio_kelly(bets):
    """
    Multi-asset Kelly: finds optimal bet sizes accounting for correlations.
    bets: list of bet dicts, each with 'edge', 'win_probability', 'odds_decimal'
    Returns: array of optimal bet fractions
    """
    if not SCIPY_AVAILABLE:
        # Fallback: independent quarter-Kelly per bet
        return np.array([
            kelly_fraction(float(b["win_probability"]), float(b.get("odds_decimal", 1.91)))
            for b in bets
        ])

    n = len(bets)
    if n == 0:
        return np.array([])
    if n == 1:
        frac = kelly_fraction(
            float(bets[0]["win_probability"]),
            float(bets[0].get("odds_decimal", 1.91))
        )
        return np.array([frac])

    C = build_correlation_matrix(bets)
    edges = np.array([float(b.get("edge", 0.0)) for b in bets])

    def neg_expected_log_growth(f):
        individual_ev = np.dot(f, edges)
        correlation_penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                correlation_penalty += f[i] * f[j] * C[i, j]
        return -(individual_ev - correlation_penalty)

    def neg_gradient(f):
        grad = -edges.copy()
        for i in range(n):
            for j in range(n):
                if i != j:
                    grad[i] += f[j] * C[i, j]
        return grad

    constraints = [
        {"type": "ineq", "fun": lambda f: MAX_TOTAL_EXPOSURE - np.sum(f)},
        {"type": "ineq", "fun": lambda f: np.sum(f)},  # sum >= 0
    ]
    bounds = [(0.0, MAX_SINGLE_BET)] * n
    x0     = np.array([max(0, min(MAX_SINGLE_BET, float(b.get("edge", 0)) * 2)) for b in bets])

    try:
        result = minimize(
            neg_expected_log_growth,
            x0,
            jac=neg_gradient,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": 1000, "ftol": 1e-8},
        )
        return np.clip(result.x, 0.0, MAX_SINGLE_BET)
    except Exception:
        # Fallback on optimizer failure
        return np.array([
            kelly_fraction(float(b["win_probability"]), float(b.get("odds_decimal", 1.91)))
            for b in bets
        ])


# ─────────────────────────────────────────────────────────────
# Section 5 — Bet Filter (Hard Rules)
# ─────────────────────────────────────────────────────────────

def bet_passes_hard_rules(bet, data, sport):
    """
    Returns (passes: bool, reason: str).
    All hard rules must pass before a bet is sized.
    """
    win_prob     = float(bet.get("win_probability", 0.5))
    conf_width   = float(bet.get("confidence_width", 0.5))
    american_ml  = bet.get("moneyline")
    bet_edge     = float(bet.get("edge", 0.0))

    # Must have positive edge
    if bet_edge < MIN_EDGE:
        return False, f"Edge too small ({bet_edge:.1%} < {MIN_EDGE:.1%})"

    # Must have tight confidence bounds
    if conf_width > MAX_CONFIDENCE_WIDTH:
        return False, f"Confidence width too wide ({conf_width:.2f} > {MAX_CONFIDENCE_WIDTH:.2f})"

    # Check condition-specific accuracy from audit data
    audit_data = data.get("audit", {})
    biases     = audit_data.get("significant_biases", [])
    for bias in biases:
        if (bias.get("sport") == sport and
                bias.get("direction") == "overconfident" and
                bet.get("game", {}).get(bias.get("segment", ""), False)):
            corrected_prob = win_prob + float(bias.get("bias", 0))
            corrected_edge = corrected_prob - implied_probability(american_ml or -110)
            if corrected_edge < MIN_EDGE:
                return False, f"Audit correction removes edge in segment '{bias['segment']}'"

    return True, "OK"


# ─────────────────────────────────────────────────────────────
# Section 6 — Main Entry Point
# ─────────────────────────────────────────────────────────────

def run_portfolio(data):
    """
    Called by brain.py --mode update if PORTFOLIO_AVAILABLE.
    Reads upcoming games with predictions, filters candidates, sizes bets.
    Writes to data['portfolio']['recommended_bets'].
    """
    print("[portfolio] Running portfolio Kelly optimization...")

    bankroll_units = float(data.get("portfolio", {}).get("bankroll_units", 100))
    candidates     = []

    for sport in ("nfl", "nba", "mlb", "nhl"):
        predictions = data.get("predictions", {}).get(sport, [])
        if not isinstance(predictions, list):
            continue
        for pred in predictions:
            if not isinstance(pred, dict):
                continue
            if pred.get("status") != "pending":
                continue
            win_prob  = float(pred.get("pick_prob", 0.5))
            home_ml   = pred.get("home_ml")
            if home_ml is None:
                continue
            decimal   = american_to_decimal(home_ml)
            bet_edge  = edge(win_prob, home_ml)

            if bet_edge < MIN_EDGE:
                continue

            game_id  = pred.get("id", "")
            conf_low  = float(pred.get("confidence_lower", 0.3))
            conf_high = float(pred.get("confidence_upper", 0.7))
            signals   = pred.get("signals") or []
            bet_obj = {
                "game_id":          game_id,
                "sport":            sport,
                "home_team":        pred.get("home"),
                "away_team":        pred.get("away"),
                "date":             pred.get("date"),
                "win_probability":  win_prob,
                "moneyline":        home_ml,
                "odds_decimal":     decimal,
                "edge":             round(bet_edge, 4),
                "confidence_width": conf_high - conf_low,
                "pick_type":        "home_favorite" if win_prob >= 0.5 else "away_favorite",
                "primary_signal":   signals[0] if signals else "",
                "weather_zone":     pred.get("weather_zone", ""),
            }

            passes, reason = bet_passes_hard_rules(bet_obj, data, sport)
            if passes:
                candidates.append(bet_obj)

    if not candidates:
        print("[portfolio] No qualifying bets found today.")
        data.setdefault("portfolio", {}).update({
            "last_updated":       datetime.utcnow().isoformat(),
            "recommended_bets":   [],
            "total_at_risk":      0.0,
            "bankroll_units":     bankroll_units,
        })
        return

    # Enforce max total exposure
    if sum(kelly_fraction(b["win_probability"], b["odds_decimal"]) for b in candidates) > MAX_TOTAL_EXPOSURE:
        # Sort by edge and take top bets that fit under total exposure
        candidates.sort(key=lambda b: b["edge"], reverse=True)
        selected, total = [], 0.0
        for bet in candidates:
            frac = kelly_fraction(bet["win_probability"], bet["odds_decimal"])
            if total + frac <= MAX_TOTAL_EXPOSURE:
                selected.append(bet)
                total += frac
        candidates = selected

    # Optimize portfolio Kelly
    fractions = portfolio_kelly(candidates)
    total_at_risk = float(np.sum(fractions))

    recommended = []
    for bet, frac in zip(candidates, fractions):
        if frac < 0.001:
            continue
        units = round(frac * bankroll_units, 2)
        recommended.append({
            **bet,
            "kelly_fraction":    round(float(frac), 4),
            "units_to_bet":      units,
            "expected_profit":   round(units * (bet["edge"] / (1 - implied_probability(bet["moneyline"]))), 2),
        })

    recommended.sort(key=lambda b: b["kelly_fraction"], reverse=True)

    print(f"[portfolio] {len(recommended)} bet(s) recommended. "
          f"Total at risk: {total_at_risk:.1%} of bankroll.")

    data.setdefault("portfolio", {}).update({
        "last_updated":     datetime.utcnow().isoformat(),
        "recommended_bets": recommended,
        "total_at_risk":    round(total_at_risk, 4),
        "bankroll_units":   bankroll_units,
        "n_candidates_before_filter": len(candidates),
    })
