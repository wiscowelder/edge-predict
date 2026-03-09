"""
audit.py — EDGE Predict Systematic Bias Detector
Analyzes resolved predictions for statistically significant error patterns.
Runs after every update. Reports findings — does NOT auto-correct.
Called by: brain.py (after update runs, if AUDIT_AVAILABLE)
"""

import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Section 1 — Constants
# ─────────────────────────────────────────────────────────────

SPORTS         = ["nfl", "nba", "mlb", "nhl", "ncaaf", "ncaabm", "ncaabw"]
MIN_SAMPLE     = 30          # Minimum games per segment to test
Z_THRESHOLD    = 1.96        # 95% confidence
STRONG_Z       = 2.58        # 99% confidence

# ─────────────────────────────────────────────────────────────
# Section 2 — Condition Segmenters
# ─────────────────────────────────────────────────────────────

AUDIT_SEGMENTS = {
    # Schedule / time
    "primetime":            lambda g: int(g.get("kickoff_hour", 13)) >= 20,
    "early_window":         lambda g: int(g.get("kickoff_hour", 13)) <= 13,
    "thursday_game":        lambda g: int(g.get("day_of_week", 6)) == 3,
    "monday_game":          lambda g: int(g.get("day_of_week", 6)) == 0,
    "saturday_game":        lambda g: int(g.get("day_of_week", 6)) == 5,

    # Game type
    "divisional":           lambda g: bool(g.get("is_division_game", False)),
    "playoff":              lambda g: bool(g.get("is_playoff", False)),
    "rivalry":              lambda g: bool(g.get("is_rivalry", False)),
    "neutral_site":         lambda g: bool(g.get("neutral_site", False)),

    # Prediction confidence tiers
    "high_confidence":      lambda g: float(g.get("win_probability", 0.5)) > 0.75,
    "medium_confidence":    lambda g: 0.55 < float(g.get("win_probability", 0.5)) <= 0.75,
    "low_confidence":       lambda g: float(g.get("win_probability", 0.5)) <= 0.55,
    "near_coinflip":        lambda g: abs(float(g.get("win_probability", 0.5)) - 0.5) < 0.05,

    # Spread
    "large_favorite":       lambda g: abs(float(g.get("spread", 0))) >= 10,
    "small_favorite":       lambda g: 2 < abs(float(g.get("spread", 0))) < 6,
    "pick_em":              lambda g: abs(float(g.get("spread", 0))) <= 2,

    # Season timing
    "early_season":         lambda g: int(g.get("week", 8)) <= 4,
    "mid_season":           lambda g: 5 <= int(g.get("week", 8)) <= 10,
    "late_season":          lambda g: int(g.get("week", 8)) >= 14,
    "bowl_season":          lambda g: bool(g.get("is_bowl", False)),

    # Weather (outdoor sports)
    "cold_weather":         lambda g: float(g.get("temperature", 70)) < 40,
    "high_wind":            lambda g: float(g.get("wind_speed", 0)) > 15,
    "precipitation":        lambda g: bool(g.get("precipitation", False)),
    "dome_game":            lambda g: bool(g.get("is_dome", False)),

    # Rest / fatigue
    "short_rest":           lambda g: float(g.get("rest_days", 7)) <= 3,
    "back_to_back":         lambda g: bool(g.get("back_to_back", False)),
    "long_rest":            lambda g: float(g.get("rest_days", 7)) >= 10,
    "rest_advantage_home":  lambda g: (float(g.get("home_rest_days", 7)) -
                                       float(g.get("away_rest_days", 7))) >= 3,
    "rest_advantage_away":  lambda g: (float(g.get("away_rest_days", 7)) -
                                       float(g.get("home_rest_days", 7))) >= 3,

    # Line signal
    "sharp_steam":          lambda g: g.get("sharp_signal_type") == "steam_move",
    "reverse_line_movement":lambda g: bool(g.get("reverse_line_movement_flag", False)),
    "big_line_move":        lambda g: float(g.get("line_movement_magnitude", 0)) > 2.0,
}


# ─────────────────────────────────────────────────────────────
# Section 3 — Statistical Significance Tester
# ─────────────────────────────────────────────────────────────

def test_segment_bias(segment_name, records, sport):
    """
    Wilson-score test for systematic over/under-confidence in a segment.
    Returns bias report dict or None if insufficient data.
    """
    if len(records) < MIN_SAMPLE:
        return None

    # Expected accuracy = mean predicted probability
    expected_acc = float(np.mean([r["win_probability"] for r in records]))
    actual_acc   = float(np.mean([int(bool(r["correct"])) for r in records]))
    n            = len(records)

    bias       = actual_acc - expected_acc
    std_error  = (expected_acc * (1 - expected_acc) / n) ** 0.5
    z_score    = bias / std_error if std_error > 1e-9 else 0.0

    significant = abs(z_score) > Z_THRESHOLD
    strong      = abs(z_score) > STRONG_Z

    return {
        "segment":           segment_name,
        "sport":             sport,
        "n":                 n,
        "expected_accuracy": round(expected_acc, 4),
        "actual_accuracy":   round(actual_acc, 4),
        "bias":              round(bias, 4),
        "z_score":           round(z_score, 3),
        "significant":       significant,
        "strong":            strong,
        "direction":         "overconfident" if bias < 0 else "underconfident",
        "recommendation":    (
            f"Apply {bias:+.1%} probability correction on {segment_name} {sport} games"
            if significant else None
        ),
        "correction_applied": False,
    }


# ─────────────────────────────────────────────────────────────
# Section 4 — Main Audit Runner
# ─────────────────────────────────────────────────────────────

def run_audit(data):
    """
    Called after each update run.
    Analyzes all resolved predictions for systematic bias patterns.
    Writes findings to data['audit']. Does NOT modify predictions.
    """
    print("[audit] Analyzing resolved predictions for systematic bias...")

    significant_biases = []
    clean_segments     = []
    total_analyzed     = 0
    last_finding       = None

    for sport in SPORTS:
        history  = data.get("prediction_history", {}).get(sport, [])
        resolved = [r for r in history if r.get("outcome") is not None]
        if not resolved:
            continue

        # Enrich records with correctness flag
        for r in resolved:
            pred_home_wins = float(r.get("win_probability", 0.5)) >= 0.5
            actual_outcome = bool(r.get("outcome"))  # 1 = home win
            r["correct"] = (pred_home_wins == actual_outcome)
            r.setdefault("win_probability", 0.5)

        total_analyzed += len(resolved)

        for seg_name, seg_fn in AUDIT_SEGMENTS.items():
            # Apply condition filter
            try:
                segment_records = [r for r in resolved
                                   if seg_fn(r.get("game", {}))]
            except Exception:
                continue

            result = test_segment_bias(seg_name, segment_records, sport)
            if result is None:
                continue

            if result["significant"]:
                significant_biases.append(result)
                if last_finding is None or result["z_score"] > (last_finding or {}).get("z_score", 0):
                    last_finding = result
                if result["strong"]:
                    print(f"  [audit] STRONG bias: {sport} {seg_name} "
                          f"bias={result['bias']:+.1%} z={result['z_score']:.2f} n={result['n']}")
            else:
                clean_segments.append({
                    "segment": seg_name,
                    "sport":   sport,
                    "n":       result["n"],
                    "z_score": result["z_score"],
                })

    # Sort by z-score magnitude (most significant first)
    significant_biases.sort(key=lambda x: abs(x["z_score"]), reverse=True)

    print(f"[audit] Found {len(significant_biases)} significant bias pattern(s). "
          f"{total_analyzed} predictions analyzed.")

    data["audit"] = {
        "last_run":                  datetime.utcnow().isoformat(),
        "significant_biases":        significant_biases,
        "clean_segments":            clean_segments[:20],  # Top 20 clean to save space
        "total_predictions_analyzed": total_analyzed,
        "last_significant_finding":  last_finding,
    }
