"""
simulate.py — EDGE Predict Game Simulator
Drive-level (NFL) and possession-level (NBA) simulation.
Produces full score distributions instead of a single win probability.
Used by cbs.py to size confidence points on games where close_game_prob is high.
Called by: brain.py --mode train_models (training)
           brain.py --mode update (inference for upcoming games)
"""

import os
import pickle
import random
import numpy as np
from statistics import mode as stat_mode
from datetime import datetime

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# Section 1 — Constants
# ─────────────────────────────────────────────────────────────

MODELS_FILE     = "simulate_models.pkl"
NFL_SIMS        = 10000
NBA_SIMS        = 10000
NFL_DRIVES_PER_GAME = 12   # Expected drives per team per game
NBA_POSSESSIONS_PER_GAME = 100  # ~100 possessions per team

NFL_DRIVE_OUTCOMES  = ["touchdown", "field_goal", "punt", "turnover",
                       "turnover_on_downs", "safety", "end_of_half"]
NFL_DRIVE_FEATURES  = [
    "starting_yard_line",
    "score_differential",
    "quarter",
    "time_remaining_half",
    "offense_adj_efficiency",
    "defense_adj_efficiency",
    "offense_rush_tendency",
    "red_zone_efficiency",
    "turnover_rate",
]

NBA_POSS_OUTCOMES  = ["made_2", "made_3", "missed_fg", "turnover", "free_throw_pair"]
NBA_POSS_FEATURES  = [
    "score_differential",
    "quarter",
    "offense_off_rating",
    "defense_def_rating",
    "offense_3pt_rate",
    "defense_pace",
    "time_remaining",
]


# ─────────────────────────────────────────────────────────────
# Section 2 — NFL Drive Outcome Model
# ─────────────────────────────────────────────────────────────

def _build_nfl_drive_training_data(data):
    """
    Builds (X, y) from nflverse-style play-by-play data stored in data.json.
    Each row is one drive. y = outcome index.
    """
    drives = data.get("nfl_drives", [])
    if len(drives) < 500:
        return None, None

    X, y = [], []
    for drive in drives:
        feat = [
            float(drive.get("starting_yard_line", 25)) / 100.0,
            float(drive.get("score_differential", 0)) / 28.0,
            float(drive.get("quarter", 2)) / 4.0,
            float(drive.get("time_remaining_half", 900)) / 1800.0,
            float(drive.get("offense_adj_efficiency", 0.5)),
            float(drive.get("defense_adj_efficiency", 0.5)),
            float(drive.get("offense_rush_tendency", 0.45)),
            float(drive.get("red_zone_efficiency", 0.55)),
            float(drive.get("turnover_rate", 0.12)),
        ]
        outcome = drive.get("outcome", "punt")
        if outcome not in NFL_DRIVE_OUTCOMES:
            outcome = "punt"
        y_idx = NFL_DRIVE_OUTCOMES.index(outcome)
        X.append(feat)
        y.append(y_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def _build_nba_possession_training_data(data):
    """Builds (X, y) from NBA possession-level data."""
    possessions = data.get("nba_possessions", [])
    if len(possessions) < 500:
        return None, None

    X, y = [], []
    for poss in possessions:
        feat = [
            float(poss.get("score_differential", 0)) / 30.0,
            float(poss.get("quarter", 2)) / 4.0,
            float(poss.get("offense_off_rating", 110)) / 130.0,
            float(poss.get("defense_def_rating", 110)) / 130.0,
            float(poss.get("offense_3pt_rate", 0.35)),
            float(poss.get("defense_pace", 100)) / 120.0,
            float(poss.get("time_remaining", 600)) / 2880.0,
        ]
        outcome = poss.get("outcome", "missed_fg")
        if outcome not in NBA_POSS_OUTCOMES:
            outcome = "missed_fg"
        y_idx = NBA_POSS_OUTCOMES.index(outcome)
        X.append(feat)
        y.append(y_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ─────────────────────────────────────────────────────────────
# Section 3 — Training Entry Point
# ─────────────────────────────────────────────────────────────

def train_simulate(data):
    """
    Called by brain.py --mode train_models.
    Trains multinomial logistic regression models for NFL drives and NBA possessions.
    Saves to simulate_models.pkl.
    """
    if not SKLEARN_AVAILABLE:
        print("[simulate] sklearn not available — skipping simulation model training.")
        return

    print("[simulate] Training NFL drive outcome model...")
    models = {}

    # NFL drive model
    X_nfl, y_nfl = _build_nfl_drive_training_data(data)
    if X_nfl is not None:
        scaler_nfl = StandardScaler()
        X_nfl_scaled = scaler_nfl.fit_transform(X_nfl)
        split = int(len(X_nfl) * 0.80)
        nfl_model = LogisticRegression(
            multi_class="multinomial", solver="lbfgs",
            max_iter=1000, C=1.0
        )
        nfl_model.fit(X_nfl_scaled[:split], y_nfl[:split])
        val_acc = float(np.mean(
            nfl_model.predict(X_nfl_scaled[split:]) == y_nfl[split:]
        ))
        print(f"  [simulate] NFL drive model val accuracy: {val_acc:.4f} ({len(X_nfl)} drives)")
        models["nfl_drive"] = {"model": nfl_model, "scaler": scaler_nfl}
    else:
        print("  [simulate] Insufficient NFL drive data — using rule-based fallback")

    # NBA possession model
    print("[simulate] Training NBA possession outcome model...")
    X_nba, y_nba = _build_nba_possession_training_data(data)
    if X_nba is not None:
        scaler_nba = StandardScaler()
        X_nba_scaled = scaler_nba.fit_transform(X_nba)
        split = int(len(X_nba) * 0.80)
        nba_model = LogisticRegression(
            multi_class="multinomial", solver="lbfgs",
            max_iter=1000, C=1.0
        )
        nba_model.fit(X_nba_scaled[:split], y_nba[:split])
        val_acc = float(np.mean(
            nba_model.predict(X_nba_scaled[split:]) == y_nba[split:]
        ))
        print(f"  [simulate] NBA possession model val accuracy: {val_acc:.4f} ({len(X_nba)} possessions)")
        models["nba_possession"] = {"model": nba_model, "scaler": scaler_nba}
    else:
        print("  [simulate] Insufficient NBA possession data — using rule-based fallback")

    with open(MODELS_FILE, "wb") as f:
        pickle.dump(models, f)
    print(f"[simulate] Models saved to {MODELS_FILE}")

    data.setdefault("simulate_meta", {}).update({
        "trained": True,
        "last_trained": datetime.utcnow().isoformat(),
        "nfl_drive_available": X_nfl is not None,
        "nba_possession_available": X_nba is not None,
    })
    print("[simulate] Training complete.")


# ─────────────────────────────────────────────────────────────
# Section 4 — NFL Drive Simulator
# ─────────────────────────────────────────────────────────────

# Rule-based drive outcome probabilities (fallback when model unavailable)
RULE_BASED_NFL = {
    "starting_25": [0.25, 0.12, 0.45, 0.09, 0.06, 0.01, 0.02],  # typical drive
    "red_zone":    [0.55, 0.25, 0.05, 0.08, 0.05, 0.01, 0.01],
    "backed_up":   [0.12, 0.06, 0.62, 0.11, 0.06, 0.01, 0.02],
}

_sim_models = {}

def _load_models():
    global _sim_models
    if _sim_models:
        return
    if os.path.exists(MODELS_FILE):
        with open(MODELS_FILE, "rb") as f:
            _sim_models = pickle.load(f)


def _predict_drive_outcome_probs(yard_line, score_diff, quarter, time_remaining_half,
                                  off_eff, def_eff, rush_tendency, rz_eff, to_rate):
    _load_models()
    model_data = _sim_models.get("nfl_drive")
    if model_data is not None:
        model  = model_data["model"]
        scaler = model_data["scaler"]
        feat   = scaler.transform([[
            yard_line / 100.0, score_diff / 28.0, quarter / 4.0,
            time_remaining_half / 1800.0, off_eff, def_eff,
            rush_tendency, rz_eff, to_rate
        ]])
        return model.predict_proba(feat).flatten().tolist()
    else:
        # Rule-based fallback
        if yard_line >= 65:
            return RULE_BASED_NFL["red_zone"]
        elif yard_line <= 20:
            return RULE_BASED_NFL["backed_up"]
        return RULE_BASED_NFL["starting_25"]


def _simulate_one_nfl_drive(possession, yard_line, score_diff, quarter,
                             home_stats, away_stats):
    off_stats = home_stats if possession == "home" else away_stats
    def_stats = away_stats if possession == "home" else home_stats

    probs = _predict_drive_outcome_probs(
        yard_line       = yard_line,
        score_diff      = score_diff if possession == "home" else -score_diff,
        quarter         = quarter,
        time_remaining_half = 900.0,  # simplified
        off_eff         = float(off_stats.get("off_adj_efficiency", 0.5)),
        def_eff         = float(def_stats.get("def_adj_efficiency", 0.5)),
        rush_tendency   = float(off_stats.get("rush_tendency", 0.45)),
        rz_eff          = float(off_stats.get("red_zone_efficiency", 0.55)),
        to_rate         = float(off_stats.get("turnover_rate", 0.12)),
    )

    outcome_idx = random.choices(range(len(NFL_DRIVE_OUTCOMES)), weights=probs, k=1)[0]
    return NFL_DRIVE_OUTCOMES[outcome_idx]


def _simulate_overtime_nfl(home_score, away_score, home_stats, away_stats):
    """Simplified OT: coin flip for possession, then random score."""
    possession = "home" if random.random() > 0.5 else "away"
    for _ in range(6):  # Max 6 drives in OT
        outcome = _simulate_one_nfl_drive(possession, 25, 0, 5, home_stats, away_stats)
        if outcome == "touchdown":
            if possession == "home":
                home_score += 6
            else:
                away_score += 6
            return home_score, away_score
        elif outcome == "field_goal":
            if possession == "home":
                home_score += 3
            else:
                away_score += 3
            if possession == "away":  # Away gets chance after home FG in OT
                return home_score, away_score
        possession = "away" if possession == "home" else "home"
    return home_score + 3, away_score  # safety valve


def simulate_nfl_game(home_team, away_team, data, n_simulations=NFL_SIMS):
    """
    Simulates one NFL game n_simulations times.
    Returns score distribution and derived statistics.
    """
    home_stats = data.get("team_stats", {}).get("nfl", {}).get(home_team, {})
    away_stats = data.get("team_stats", {}).get("nfl", {}).get(away_team, {})
    results    = []

    for _ in range(n_simulations):
        home_score = 0
        away_score = 0
        possession = "away"   # Away kicks off
        yard_line  = 25
        quarter    = 1
        drives_per_q = 3

        for q in range(1, 5):
            for _ in range(drives_per_q):
                outcome = _simulate_one_nfl_drive(
                    possession, yard_line,
                    home_score - away_score, q,
                    home_stats, away_stats
                )
                if outcome == "touchdown":
                    pts = 7
                    if possession == "home": home_score += pts
                    else:                    away_score += pts
                    possession = "away" if possession == "home" else "home"
                    yard_line  = 25
                elif outcome == "field_goal":
                    pts = 3
                    if possession == "home": home_score += pts
                    else:                    away_score += pts
                    possession = "away" if possession == "home" else "home"
                    yard_line  = 25
                elif outcome in ("punt", "turnover_on_downs", "end_of_half"):
                    possession = "away" if possession == "home" else "home"
                    yard_line  = max(5, 100 - yard_line)
                elif outcome == "turnover":
                    possession = "away" if possession == "home" else "home"
                    yard_line  = max(5, min(95, yard_line + 10))
                elif outcome == "safety":
                    if possession == "home": away_score += 2
                    else:                    home_score += 2

        if home_score == away_score:
            home_score, away_score = _simulate_overtime_nfl(
                home_score, away_score, home_stats, away_stats
            )

        results.append({"home": home_score, "away": away_score})

    return aggregate_score_distribution(results)


# ─────────────────────────────────────────────────────────────
# Section 5 — NBA Possession Simulator
# ─────────────────────────────────────────────────────────────

def _simulate_one_nba_possession(possession, score_diff, quarter, home_stats, away_stats):
    _load_models()
    off_stats = home_stats if possession == "home" else away_stats
    def_stats = away_stats if possession == "home" else home_stats

    model_data = _sim_models.get("nba_possession")
    if model_data:
        scaler = model_data["scaler"]
        model  = model_data["model"]
        feat   = scaler.transform([[
            (score_diff if possession == "home" else -score_diff) / 30.0,
            quarter / 4.0,
            float(off_stats.get("off_rating", 110)) / 130.0,
            float(def_stats.get("def_rating", 110)) / 130.0,
            float(off_stats.get("three_pt_rate", 0.35)),
            float(def_stats.get("pace", 100)) / 120.0,
            0.5,  # time_remaining placeholder
        ]])
        probs = model.predict_proba(feat).flatten().tolist()
    else:
        # Rule-based fallback: ~1.05 pts per possession average
        probs = [0.30, 0.12, 0.38, 0.10, 0.10]

    outcome_idx = random.choices(range(len(NBA_POSS_OUTCOMES)), weights=probs, k=1)[0]
    outcome     = NBA_POSS_OUTCOMES[outcome_idx]

    if outcome == "made_2":
        return 2
    elif outcome == "made_3":
        return 3
    elif outcome == "free_throw_pair":
        ft_pct = float(off_stats.get("ft_pct", 0.75))
        return int(random.random() < ft_pct) + int(random.random() < ft_pct)
    return 0


def simulate_nba_game(home_team, away_team, data, n_simulations=NBA_SIMS):
    """Simulates one NBA game n_simulations times."""
    home_stats = data.get("team_stats", {}).get("nba", {}).get(home_team, {})
    away_stats = data.get("team_stats", {}).get("nba", {}).get(away_team, {})
    results    = []

    for _ in range(n_simulations):
        home_score = 0
        away_score = 0

        for q in range(1, 5):
            for _ in range(NBA_POSSESSIONS_PER_GAME // 4):
                home_score += _simulate_one_nba_possession(
                    "home", home_score - away_score, q, home_stats, away_stats
                )
                away_score += _simulate_one_nba_possession(
                    "away", home_score - away_score, q, home_stats, away_stats
                )

        if home_score == away_score:
            # OT: 5 possessions each
            for _ in range(5):
                home_score += _simulate_one_nba_possession("home", 0, 5, home_stats, away_stats)
                away_score += _simulate_one_nba_possession("away", 0, 5, home_stats, away_stats)
            if home_score == away_score:
                home_score += 1  # Force a winner

        results.append({"home": home_score, "away": away_score})

    return aggregate_score_distribution(results)


# ─────────────────────────────────────────────────────────────
# Section 6 — Score Distribution Aggregator
# ─────────────────────────────────────────────────────────────

def aggregate_score_distribution(results):
    """Aggregates simulation results into statistics."""
    n       = len(results)
    margins = [abs(r["home"] - r["away"]) for r in results]
    home_scores = [r["home"] for r in results]
    away_scores = [r["away"] for r in results]
    home_wins   = sum(1 for r in results if r["home"] > r["away"])

    sorted_margins = sorted(margins)
    ot_count = sum(1 for r in results if r["home"] == r["away"])

    try:
        likely_home = stat_mode(home_scores)
        likely_away = stat_mode(away_scores)
    except Exception:
        likely_home = int(round(float(np.mean(home_scores))))
        likely_away = int(round(float(np.mean(away_scores))))

    return {
        "home_win_prob":    round(home_wins / n, 4),
        "most_likely_score": f"{likely_home}-{likely_away}",
        "avg_margin":       round(float(np.mean(margins)), 1),
        "close_game_prob":  round(sum(1 for m in margins if m <= 7) / n, 4),
        "blowout_prob":     round(sum(1 for m in margins if m >= 17) / n, 4),
        "overtime_prob":    round(ot_count / n, 4),
        "simulations_run":  n,
        "margin_percentiles": {
            "25th": sorted_margins[n // 4],
            "50th": sorted_margins[n // 2],
            "75th": sorted_margins[3 * n // 4],
        },
    }


# ─────────────────────────────────────────────────────────────
# Section 7 — CBS Modifier
# ─────────────────────────────────────────────────────────────

def cbs_confidence_modifier(score_distribution):
    """
    Returns confidence modifier for cbs.py based on score distribution.
    Blowout expected → +0.10, Close game expected → -0.08.
    """
    if score_distribution.get("blowout_prob", 0) > 0.45:
        return +0.10
    if score_distribution.get("close_game_prob", 0) > 0.50:
        return -0.08
    return 0.0


# ─────────────────────────────────────────────────────────────
# Section 8 — Inference Entry Point
# ─────────────────────────────────────────────────────────────

def simulate_game(home_team, away_team, sport, data, n_simulations=10000):
    """
    Called during update runs for upcoming NFL and NBA games.
    Writes to data['simulate'][sport][game_id].
    Returns score_distribution dict or None for unsupported sports.
    """
    if sport == "nfl":
        dist = simulate_nfl_game(home_team, away_team, data, n_simulations)
    elif sport == "nba":
        dist = simulate_nba_game(home_team, away_team, data, n_simulations)
    else:
        return None  # Only NFL and NBA supported

    dist["cbs_modifier"] = cbs_confidence_modifier(dist)
    dist["generated_at"] = datetime.utcnow().isoformat()
    return dist
