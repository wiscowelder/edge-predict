"""
baseline.py — EDGE Predict Baseline Validator
Continuously measures system accuracy against four simple baselines.
Raises alarms if the system degrades below them.
Called by: brain.py (after update runs, if BASELINE_AVAILABLE)
"""

import random
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Section 1 — Constants
# ─────────────────────────────────────────────────────────────

SPORTS          = ["nfl", "nba", "mlb", "nhl", "ncaaf", "ncaabm", "ncaabw"]
MIN_SAMPLE      = 30
TREND_WINDOW    = 4   # Weeks for degrading trend detection

# Empirical home win rates by sport (long-run averages)
HOME_WIN_RATES = {
    "nfl":    0.572,
    "nba":    0.598,
    "mlb":    0.540,
    "nhl":    0.553,
    "ncaaf":  0.602,
    "ncaabm": 0.615,
    "ncaabw": 0.610,
}

ALARM_THRESHOLDS = {
    "below_always_home":     True,
    "below_always_favorite": True,
    "near_random":           True,   # System accuracy < 53%
    "degrading_trend":       True,
}


# ─────────────────────────────────────────────────────────────
# Section 2 — Baseline Strategies
# ─────────────────────────────────────────────────────────────

def _always_home(game):
    """Predict home team wins every game."""
    return True  # Returns True = home team wins


def _always_favorite(game):
    """Predict whichever team Vegas has as the favorite."""
    spread = float(game.get("spread", 0))
    # Negative spread = home team is favored
    return spread <= 0


def _implied_prob_from_moneyline(moneyline):
    """Convert American moneyline to implied probability."""
    if moneyline is None:
        return 0.5
    ml = float(moneyline)
    if ml > 0:
        return 100.0 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)


def _vegas_line(game):
    """Use Vegas moneyline implied probability as prediction."""
    home_ml = game.get("moneyline_home")
    if home_ml is None:
        return 0.5
    return _implied_prob_from_moneyline(home_ml)


def _random_pick(_game):
    """Random 50/50 baseline."""
    return random.random() > 0.5


# ─────────────────────────────────────────────────────────────
# Section 3 — Baseline Evaluator
# ─────────────────────────────────────────────────────────────

def evaluate_baseline(records, baseline_fn, baseline_name):
    """
    Evaluates a baseline strategy against resolved game records.
    Returns accuracy float.
    """
    if not records:
        return None
    correct = 0
    total   = 0
    for r in records:
        game   = r.get("game", {})
        actual = bool(r.get("outcome"))  # 1 = home win
        try:
            pred = baseline_fn(game)
            if isinstance(pred, float):
                pred = pred >= 0.5
            if bool(pred) == actual:
                correct += 1
        except Exception:
            pass
        total += 1
    return correct / total if total > 0 else None


def evaluate_system(records):
    """Evaluates system's own predictions against resolved outcomes."""
    if not records:
        return None
    correct = sum(
        1 for r in records
        if (float(r.get("win_probability", 0.5)) >= 0.5) == bool(r.get("outcome"))
    )
    return correct / len(records)


# ─────────────────────────────────────────────────────────────
# Section 4 — Trend Detector
# ─────────────────────────────────────────────────────────────

def detect_degrading_trend(records, window=TREND_WINDOW):
    """
    Detects if system accuracy has been declining for the last `window` weeks.
    Returns True if degrading.
    """
    if len(records) < window * 5:
        return False

    # Sort by date and split into recent windows
    sorted_records = sorted(records, key=lambda r: r.get("game", {}).get("date", ""))
    chunk_size     = max(5, len(sorted_records) // window)

    week_accuracies = []
    for i in range(window):
        start = len(sorted_records) - (window - i) * chunk_size
        end   = start + chunk_size
        chunk = sorted_records[max(0, start):end]
        if chunk:
            acc = evaluate_system(chunk)
            if acc is not None:
                week_accuracies.append(acc)

    if len(week_accuracies) < 3:
        return False

    # Trend: each week worse than the previous
    is_declining = all(
        week_accuracies[i] > week_accuracies[i + 1]
        for i in range(len(week_accuracies) - 1)
    )
    return is_declining


# ─────────────────────────────────────────────────────────────
# Section 5 — Main Entry Point
# ─────────────────────────────────────────────────────────────

def run_baseline(data):
    """
    Called after each update run.
    Measures system accuracy vs baselines for all sports.
    Writes to data['baseline']. Raises alarms if system degrades.
    """
    print("[baseline] Evaluating system vs baselines...")

    system_accuracy    = {}
    baseline_accuracy  = {
        "always_home":     {},
        "always_favorite": {},
        "vegas_line":      {},
        "random":          {},
    }
    system_vs_baselines = {}
    alarms              = []

    for sport in SPORTS:
        history  = data.get("prediction_history", {}).get(sport, [])
        resolved = [r for r in history if r.get("outcome") is not None]

        if len(resolved) < MIN_SAMPLE:
            continue

        sys_acc  = evaluate_system(resolved)
        home_acc = evaluate_baseline(resolved, _always_home,     "always_home")
        fav_acc  = evaluate_baseline(resolved, _always_favorite,  "always_favorite")
        veg_acc  = evaluate_baseline(resolved, _vegas_line,       "vegas_line")
        rnd_acc  = evaluate_baseline(resolved, _random_pick,      "random")
        degrading = detect_degrading_trend(resolved)

        system_accuracy[sport]                   = round(sys_acc, 4)  if sys_acc  else None
        baseline_accuracy["always_home"][sport]  = round(home_acc, 4) if home_acc else None
        baseline_accuracy["always_favorite"][sport] = round(fav_acc, 4) if fav_acc else None
        baseline_accuracy["vegas_line"][sport]   = round(veg_acc, 4)  if veg_acc  else None
        baseline_accuracy["random"][sport]       = 0.500

        sport_alarms = []

        beats_home = (sys_acc or 0) > (home_acc or 0)
        beats_fav  = (sys_acc or 0) > (fav_acc or 0)
        beats_veg  = (sys_acc or 0) > (veg_acc or 0)
        near_rnd   = (sys_acc or 0.5) < 0.53

        if not beats_home:
            alarm_msg = f"[ALARM] {sport}: system ({sys_acc:.1%}) below always-home baseline ({home_acc:.1%})"
            print(f"  {alarm_msg}")
            sport_alarms.append(alarm_msg)

        if not beats_fav:
            alarm_msg = f"[ALARM] {sport}: system ({sys_acc:.1%}) below always-favorite baseline ({fav_acc:.1%})"
            print(f"  {alarm_msg}")
            sport_alarms.append(alarm_msg)

        if near_rnd:
            alarm_msg = f"[ALARM] {sport}: system accuracy ({sys_acc:.1%}) near random — possible data issue"
            print(f"  {alarm_msg}")
            sport_alarms.append(alarm_msg)

        if degrading:
            alarm_msg = f"[ALARM] {sport}: accuracy declining for {TREND_WINDOW} consecutive weeks"
            print(f"  {alarm_msg}")
            sport_alarms.append(alarm_msg)

        alarms.extend(sport_alarms)

        system_vs_baselines[sport] = {
            "system_accuracy": round(sys_acc, 4) if sys_acc else None,
            "beats_home":     beats_home,
            "beats_favorite": beats_fav,
            "beats_vegas":    beats_veg,
            "near_random":    near_rnd,
            "degrading_trend": degrading,
            "alarm":          len(sport_alarms) > 0,
            "n_predictions":  len(resolved),
        }

        print(f"  [baseline] {sport}: sys={sys_acc:.1%} home={home_acc:.1%} "
              f"fav={fav_acc:.1%} vegas={veg_acc:.1%} "
              f"beats_home={'✓' if beats_home else '✗'} "
              f"beats_fav={'✓' if beats_fav else '✗'} "
              f"beats_vegas={'✓' if beats_veg else '✗'}")

    data["baseline"] = {
        "last_evaluated":      datetime.utcnow().isoformat(),
        "system_accuracy":     system_accuracy,
        "baseline_accuracy":   baseline_accuracy,
        "system_vs_baselines": system_vs_baselines,
        "alarms":              alarms,
        "alarm_count":         len(alarms),
    }

    if alarms:
        print(f"[baseline] ⚠ {len(alarms)} alarm(s) raised — check data['baseline']['alarms']")
    else:
        print("[baseline] All systems above baselines. ✓")
