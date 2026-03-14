"""
explain.py — EDGE Predict Explainability Layer
For every prediction: outputs which signals drove it and by how much.
Tracks global feature importance across full history.
Identifies noise signals for removal.
Called by: brain.py --mode train_models (initialize)
           brain.py --mode update (per-prediction explanations)
"""

import os
import pickle
import numpy as np
from datetime import datetime

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[explain] WARNING: shap not installed. Run: pip install shap>=0.44.0")

# ─────────────────────────────────────────────────────────────
# Section 1 — Constants
# ─────────────────────────────────────────────────────────────

SPORTS              = ["nfl", "nba", "mlb", "nhl", "ncaaf", "ncaabm", "ncaabw"]
EXPLAINERS_FILE     = "explain_models.pkl"
NOISE_THRESHOLD     = 0.005   # SHAP below 0.5% = noise candidate
MIN_SAMPLE_NOISE    = 100     # Need 100+ predictions before calling something noise

FEATURE_LABELS = [
    "calibrated_prob", "elo_component", "pythagorean_component",
    "h2h_component", "form_component", "xgb_component",
    "sharp_prob_delta", "sharp_confidence", "sentiment_adjustment",
    "lineup_home_adj", "lineup_away_adj", "official_impact",
    "momentum_score", "momentum_uncertainty", "dow_pattern",
    "weather_wind", "weather_temp", "rest_delta",
    "home_field_advantage", "back_to_back_flag",
]


# ─────────────────────────────────────────────────────────────
# Section 2 — XGBoost SHAP Explainer
# ─────────────────────────────────────────────────────────────

def build_xgb_explainer(xgb_model):
    """
    Creates a SHAP TreeExplainer for an XGBoost model.
    Fast and exact for tree-based models.
    """
    if not SHAP_AVAILABLE:
        return None
    return shap.TreeExplainer(xgb_model)


def explain_xgb_prediction(game_features, explainer, feature_names=None):
    """
    Returns per-feature SHAP values for a single prediction.
    game_features: 1D array of feature values
    """
    if explainer is None or not SHAP_AVAILABLE:
        return {}
    feats = np.array(game_features).reshape(1, -1)
    shap_values = explainer.shap_values(feats)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Binary classification: class 1
    shap_values = shap_values.flatten()
    names = feature_names or [f"feature_{i}" for i in range(len(shap_values))]
    return {names[i]: float(shap_values[i]) for i in range(len(names))}


# ─────────────────────────────────────────────────────────────
# Section 3 — Ensemble Component Attribution
# ─────────────────────────────────────────────────────────────

def explain_ensemble_components(game, raw_prob):
    """
    Decomposes the raw ensemble probability into per-component contributions.
    Uses the additive structure of brain.py's ensemble.
    """
    baseline = 0.5
    components = {}

    elo_comp  = float(game.get("elo_component", 0.5))
    pyth_comp = float(game.get("pythagorean_component", 0.5))
    h2h_comp  = float(game.get("h2h_component", 0.5))
    form_comp = float(game.get("form_component", 0.5))
    xgb_comp  = float(game.get("xgb_component", 0.5))

    total_shift = float(raw_prob) - baseline
    total_weight = sum([
        abs(elo_comp - 0.5),
        abs(pyth_comp - 0.5),
        abs(h2h_comp - 0.5),
        abs(form_comp - 0.5),
        abs(xgb_comp - 0.5),
    ]) or 1e-9

    components["elo"]          = round((elo_comp  - 0.5) / total_weight * total_shift, 4)
    components["pythagorean"]  = round((pyth_comp - 0.5) / total_weight * total_shift, 4)
    components["h2h"]          = round((h2h_comp  - 0.5) / total_weight * total_shift, 4)
    components["form"]         = round((form_comp - 0.5) / total_weight * total_shift, 4)
    components["xgboost"]      = round((xgb_comp  - 0.5) / total_weight * total_shift, 4)

    return components


# ─────────────────────────────────────────────────────────────
# Section 4 — LSTM Attention / Gradient Attribution
# ─────────────────────────────────────────────────────────────

def explain_lstm_sequence(sequence_tensor, lstm_model):
    """
    Gradient-based attribution: which games in the sequence drove momentum most.
    sequence_tensor: [seq_len, feature_dim] PyTorch tensor
    Returns: importance array of shape [seq_len] (higher = more influential)
    """
    try:
        import torch
        seq = sequence_tensor.clone().detach().float().unsqueeze(0)  # [1, seq_len, feat]
        seq.requires_grad_(True)

        lstm_model.eval()
        output = lstm_model(seq)
        if output.dim() > 1:
            output = output.squeeze()
        output.backward()

        importance = seq.grad.abs().mean(dim=-1).squeeze(0).detach().numpy()
        # Normalize
        total = importance.sum()
        if total > 1e-9:
            importance = importance / total
        return importance
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Section 5 — Meta-Learner SHAP (Stack)
# ─────────────────────────────────────────────────────────────

def explain_stack_prediction(meta_features, stack_model, feature_names=None):
    """
    Approximates SHAP values for stack.py's MLP using KernelSHAP.
    Slower than TreeExplainer but works on any model.
    """
    if not SHAP_AVAILABLE:
        return {}
    try:
        background = np.zeros((1, len(meta_features)))
        explainer  = shap.KernelExplainer(
            lambda x: stack_model(
                __import__("torch").from_numpy(x.astype(np.float32))
            ).detach().numpy(),
            background
        )
        shap_vals = explainer.shap_values(
            np.array(meta_features).reshape(1, -1), nsamples=100
        ).flatten()
        names = feature_names or [f"meta_{i}" for i in range(len(shap_vals))]
        return {names[i]: float(shap_vals[i]) for i in range(len(names))}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────
# Section 6 — Global Feature Importance Tracker
# ─────────────────────────────────────────────────────────────

def update_global_importance(data, sport, shap_dict):
    """
    Accumulates SHAP values across all resolved predictions.
    Stores rolling mean importance per feature per sport.
    """
    explain_data = data.setdefault("explain", {})
    global_imp   = explain_data.setdefault("global_importance", {})
    sport_imp    = global_imp.setdefault(sport, {})
    counts       = explain_data.setdefault("global_counts", {}).setdefault(sport, {})

    for feature, shap_val in shap_dict.items():
        prev_mean  = sport_imp.get(feature, 0.0)
        prev_count = counts.get(feature, 0)
        new_count  = prev_count + 1
        new_mean   = (prev_mean * prev_count + abs(shap_val)) / new_count
        sport_imp[feature] = round(new_mean, 6)
        counts[feature]    = new_count


# ─────────────────────────────────────────────────────────────
# Section 7 — Signal Noise Detector
# ─────────────────────────────────────────────────────────────

def detect_noise_signals(data):
    """
    After accumulating global importance, identifies signals below threshold.
    Writes recommendations to data['explain']['noise_signals'].
    Does NOT automatically remove anything — recommendations only.
    """
    explain_data = data.get("explain", {})
    global_imp   = explain_data.get("global_importance", {})
    counts       = explain_data.get("global_counts", {})
    noise_signals = []

    for sport, features in global_imp.items():
        sport_counts = counts.get(sport, {})
        total_shap = sum(features.values()) or 1e-9

        for feature, mean_shap in features.items():
            sample_size = sport_counts.get(feature, 0)
            if sample_size < MIN_SAMPLE_NOISE:
                continue
            relative_importance = mean_shap / total_shap
            if relative_importance < NOISE_THRESHOLD:
                noise_signals.append({
                    "signal":         feature,
                    "sport":          sport,
                    "avg_shap":       round(mean_shap, 6),
                    "relative_importance": round(relative_importance, 6),
                    "sample_size":    sample_size,
                    "recommendation": (
                        f"Consider removing — contributing less than "
                        f"{NOISE_THRESHOLD*100:.1f}% of predictive power"
                    ),
                    "correction_applied": False,
                })

    data.setdefault("explain", {})["noise_signals"] = noise_signals
    if noise_signals:
        print(f"[explain] Found {len(noise_signals)} low-importance signal(s) across all sports.")


# ─────────────────────────────────────────────────────────────
# Section 8 — Per-Prediction Explanation Generator
# ─────────────────────────────────────────────────────────────

def generate_explanation(game, final_prob, raw_prob, scout_data, edge_data, mind_data,
                         shap_breakdown=None):
    """
    Produces a structured, frontend-readable explanation dict for one prediction.
    """
    ensemble_comps = explain_ensemble_components(game, raw_prob)
    calibration_shift = round(float(final_prob) - float(raw_prob), 4)

    # Sharp money
    sharp_delta   = float(edge_data.get("sharp_prob_delta", 0.0))
    sharp_conf    = float(edge_data.get("sharp_confidence", 0.0))
    sharp_signal  = edge_data.get("sharp_signal_type", "")

    # Scout adjustments
    home_lineup   = float(scout_data.get("home_lineup_adjustment", 0.0))
    away_lineup   = float(scout_data.get("away_lineup_adjustment", 0.0))
    home_missing  = scout_data.get("home_missing_players", [])
    away_missing  = scout_data.get("away_missing_players", [])

    # Momentum
    momentum      = float(mind_data.get("momentum_score", 0.5))
    momentum_unc  = float(mind_data.get("momentum_uncertainty", 0.1))

    # Rank drivers by magnitude
    driver_magnitudes = {}
    if shap_breakdown:
        driver_magnitudes = {k: abs(v) for k, v in shap_breakdown.items()}
    else:
        driver_magnitudes = {
            "lineup":      abs(home_lineup - away_lineup),
            "sharp_money": abs(sharp_delta) * sharp_conf,
            "momentum":    abs(momentum - 0.5),
            "calibration": abs(calibration_shift),
            **{k: abs(v) for k, v in ensemble_comps.items()},
        }

    sorted_drivers = sorted(driver_magnitudes.items(), key=lambda x: x[1], reverse=True)
    primary   = sorted_drivers[0][0]  if sorted_drivers     else "ensemble"
    secondary = sorted_drivers[1][0]  if len(sorted_drivers) > 1 else ""

    # Build readable supporting list
    supporting = []
    if abs(home_lineup) > 0.02:
        players = ", ".join(home_missing[:2]) if home_missing else "key player"
        supporting.append(f"Home lineup impact: {players} ({home_lineup:+.1%})")
    if abs(away_lineup) > 0.02:
        players = ", ".join(away_missing[:2]) if away_missing else "key player"
        supporting.append(f"Away lineup impact: {players} ({away_lineup:+.1%})")
    if abs(sharp_delta) > 0.02 and sharp_conf > 0.4:
        supporting.append(f"Sharp money: {sharp_signal or 'movement'} ({sharp_delta:+.1%})")
    if abs(momentum - 0.5) > 0.05:
        direction = "hot" if momentum > 0.5 else "cold"
        supporting.append(f"Momentum: team is {direction} ({momentum:.2f})")

    # Concerns
    concerns = []
    conf_width = float(mind_data.get("confidence_width", 0))
    if conf_width > 0.30:
        concerns.append("Wide confidence interval — model less certain than usual")
    if momentum_unc > 0.08:
        concerns.append("High momentum uncertainty")
    if abs(float(final_prob) - 0.5) < 0.08:
        concerns.append("Near coin-flip — low predictive edge")

    # Confidence rating
    prob_dist = abs(float(final_prob) - 0.5)
    if prob_dist > 0.20 and conf_width < 0.20:
        confidence_rating = "HIGH"
    elif prob_dist > 0.10:
        confidence_rating = "MEDIUM"
    else:
        confidence_rating = "LOW"

    # One-line summary
    home_team = game.get("home_team", "Home")
    away_team = game.get("away_team", "Away")
    favored   = home_team if float(final_prob) >= 0.5 else away_team
    one_line  = f"{favored} favored (primary driver: {primary.replace('_', ' ')})"

    return {
        "primary_driver":   primary,
        "secondary_driver": secondary,
        "supporting":       supporting[:4],
        "concerns":         concerns[:3],
        "confidence_rating": confidence_rating,
        "ensemble_breakdown": ensemble_comps,
        "shap_breakdown":   shap_breakdown or {},
        "one_line":         one_line,
    }


# ─────────────────────────────────────────────────────────────
# Section 9 — Initialization (called during train_models)
# ─────────────────────────────────────────────────────────────

def initialize_explain(data):
    """
    Called by brain.py --mode train_models.
    Loads the XGBoost model from models.pkl and runs a first global importance pass
    over all historical resolved predictions.
    """
    print("[explain] Initializing SHAP explainers...")

    if not SHAP_AVAILABLE:
        print("[explain] shap library not available — install with: pip install shap>=0.44.0")
        data.setdefault("explain", {})["initialized"] = False
        return

    # Load XGBoost models
    # models.pkl is a bundle: {'models': {sport: XGBClassifier}, 'feature_names': [...]}
    # We must extract the inner 'models' dict to get actual XGBClassifier objects.
    # Iterating the outer bundle dict gives dicts and lists, not classifiers,
    # which causes shap.TreeExplainer to fail with "Model type not supported".
    explainers = {}
    if os.path.exists("models.pkl"):
        with open("models.pkl", "rb") as f:
            bundle = pickle.load(f)
        # Safely extract the inner models dict from the bundle
        if isinstance(bundle, dict):
            xgb_models = bundle.get("models", {})
        else:
            # Unexpected format — try using it directly
            xgb_models = bundle if isinstance(bundle, dict) else {}
        for sport, model_obj in xgb_models.items():
            try:
                explainers[sport] = build_xgb_explainer(model_obj)
                print(f"  [explain] Built TreeExplainer for {sport}")
            except Exception as e:
                print(f"  [explain] Could not build explainer for {sport}: {e}")

    # Save explainers
    with open(EXPLAINERS_FILE, "wb") as f:
        pickle.dump(explainers, f)

    # Run global importance pass on historical resolved predictions
    print("[explain] Running initial global feature importance pass...")
    total_explained = 0
    for sport in SPORTS:
        history = data.get("prediction_history", {}).get(sport, [])
        explainer = explainers.get(sport)
        for record in history:
            if record.get("outcome") is None:
                continue
            game      = record.get("game", {})
            raw_prob  = record.get("raw_win_probability", 0.5)
            ens_comps = explain_ensemble_components(game, raw_prob)
            update_global_importance(data, sport, ens_comps)
            total_explained += 1

    detect_noise_signals(data)

    data.setdefault("explain", {}).update({
        "initialized": True,
        "last_initialized": datetime.utcnow().isoformat(),
        "total_predictions_explained": total_explained,
        "shap_available": SHAP_AVAILABLE,
    })
    print(f"[explain] Initialization complete. {total_explained} historical predictions analyzed.")


# ─────────────────────────────────────────────────────────────
# Section 10 — Update Entry Point (called per-prediction)
# ─────────────────────────────────────────────────────────────

_explainers = {}

def _load_explainers():
    global _explainers
    if _explainers:
        return
    if os.path.exists(EXPLAINERS_FILE):
        with open(EXPLAINERS_FILE, "rb") as f:
            _explainers = pickle.load(f)


def explain_prediction(game, final_prob, raw_prob, scout_data, edge_data, mind_data,
                       sport, data, game_features=None):
    """
    Called during update runs for each game prediction.
    Returns structured explanation dict and updates global importance tracking.

    Parameters
    ----------
    data : dict
        The live data dict (same one brain.py passes everywhere). Required so
        that update_global_importance accumulates SHAP values persistently
        across predictions rather than discarding them into a local throwaway.
    """
    _load_explainers()
    explainer      = _explainers.get(sport)
    shap_breakdown = {}

    if explainer and game_features is not None and SHAP_AVAILABLE:
        shap_breakdown = explain_xgb_prediction(game_features, explainer)

    explanation = generate_explanation(
        game, final_prob, raw_prob, scout_data, edge_data, mind_data, shap_breakdown
    )

    # Accumulate global importance into the real data dict so values persist
    # across the full update run and are eventually written to data.json.
    if shap_breakdown:
        update_global_importance(data, sport, shap_breakdown)

    return explanation


SPORTS = ["nfl", "nba", "mlb", "nhl", "ncaaf", "ncaabm", "ncaabw"]
