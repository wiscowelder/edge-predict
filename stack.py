"""
stack.py — EDGE Predict Meta-Learner
Trains one MLP per sport on all model outputs as meta-features.
Learns optimal combination weights from historical outcomes.
Called by: brain.py --mode train_models
"""

import os
import pickle
import json
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────
# Section 1 — Constants
# ─────────────────────────────────────────────────────────────

SPORTS = ["nfl", "nba", "mlb", "nhl", "ncaaf", "ncaabm", "ncaabw"]
MODELS_FILE = "stack_models.pkl"
META_FEATURE_DIM = 30
HIDDEN_DIMS = [64, 32, 16]
DROPOUT_RATES = [0.3, 0.2, 0.0]
EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 32
LR = 0.001
WEIGHT_DECAY = 1e-4
TRAIN_CUTOFF_SEASON = 2019
VAL_CUTOFF_SEASON = 2022


# ─────────────────────────────────────────────────────────────
# Section 2 — Meta-Feature Extraction
# ─────────────────────────────────────────────────────────────

def build_meta_features(game, raw_prob, scout_data, edge_data, mind_data, sport):
    """
    Assembles 30-dimensional meta-feature vector from all model outputs.
    Missing values fall back to neutral defaults.
    """
    game_id = str(game.get("id") or game.get("game_id", ""))

    # brain.py component outputs
    brain_elo       = float(game.get("elo_component", 0.5))
    brain_pyth      = float(game.get("pythagorean_component", 0.5))
    brain_h2h       = float(game.get("h2h_component", 0.5))
    brain_form      = float(game.get("form_component", 0.5))
    brain_xgb       = float(game.get("xgb_component", 0.5))
    brain_injury    = float(game.get("injury_component", 0.0))
    raw_prob_f      = float(raw_prob or game.get("raw_win_probability", 0.5))

    # mind.py outputs
    cal_prob        = float(mind_data.get("calibrated_prob", raw_prob_f))
    momentum        = float(mind_data.get("momentum_score", 0.5))
    momentum_unc    = float(mind_data.get("momentum_uncertainty", 0.1))
    cal_correction  = float(mind_data.get("calibration_correction", 0.0))
    bayes_n         = min(1.0, float(mind_data.get("bayesian_sample_size", 0)) / 500.0)

    # edge.py outputs
    sharp_delta     = float(edge_data.get("sharp_prob_delta", 0.0))
    sharp_conf      = float(edge_data.get("sharp_confidence", 0.0))
    sentiment_delta = float(edge_data.get("sentiment_delta", 0.0))
    line_move_mag   = float(edge_data.get("line_movement_magnitude", 0.0))
    rlm_flag        = float(edge_data.get("reverse_line_movement_flag", 0))

    # scout.py outputs
    home_lineup_adj = float(scout_data.get("home_lineup_adjustment", 0.0))
    away_lineup_adj = float(scout_data.get("away_lineup_adjustment", 0.0))
    official_impact = float(scout_data.get("official_impact", 0.0))
    home_star_avail = float(scout_data.get("home_star_player_available", 1))
    away_star_avail = float(scout_data.get("away_star_player_available", 1))

    # Context features
    season_len = {"nfl": 18, "nba": 82, "mlb": 162, "nhl": 82,
                  "ncaaf": 14, "ncaabm": 35, "ncaabw": 35}.get(sport, 30)
    week_norm   = min(1.0, float(game.get("week", 1)) / season_len)
    is_playoff  = float(game.get("is_playoff", 0))
    is_division = float(game.get("is_division_game", 0))
    home_rest   = min(1.0, float(game.get("home_rest_days", 7)) / 14.0)
    away_rest   = min(1.0, float(game.get("away_rest_days", 7)) / 14.0)
    game_total  = min(1.0, float(game.get("game_total", 45)) / 80.0)
    spread_mag  = min(1.0, abs(float(game.get("spread", 0))) / 20.0)

    return np.array([
        raw_prob_f, brain_elo, brain_pyth, brain_h2h, brain_form, brain_xgb, brain_injury,
        cal_prob, momentum, momentum_unc, cal_correction, bayes_n,
        sharp_delta, sharp_conf, sentiment_delta, line_move_mag, rlm_flag,
        home_lineup_adj, away_lineup_adj, official_impact, home_star_avail, away_star_avail,
        week_norm, is_playoff, is_division, home_rest, away_rest, game_total, spread_mag,
        float(sport in ["nba", "mlb"]),  # high-sequence sport flag for transformer weight
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Section 3 — Meta-Learner Architecture
# ─────────────────────────────────────────────────────────────

class MetaLearner(nn.Module):
    def __init__(self, input_dim=META_FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# Section 4 — Training Data Builder
# ─────────────────────────────────────────────────────────────

def _build_training_data(data, sport):
    """
    Iterates resolved historical predictions and builds (features, label) pairs.
    Returns X (np array), y (np array) or (None, None) if insufficient data.
    """
    history = data.get("prediction_history", {}).get(sport, [])
    if len(history) < 50:
        print(f"  [stack] {sport}: insufficient history ({len(history)} games) — skipping")
        return None, None

    X, y = [], []
    for record in history:
        if record.get("outcome") is None:
            continue
        game     = record.get("game", {})
        raw_prob = record.get("raw_win_probability", 0.5)
        scout_d  = record.get("scout_data", {})
        edge_d   = record.get("edge_data", {})
        mind_d   = record.get("mind_data", {})
        outcome  = float(record["outcome"])  # 1 = home win, 0 = away win

        feats = build_meta_features(game, raw_prob, scout_d, edge_d, mind_d, sport)
        X.append(feats)
        y.append(outcome)

    if len(X) < 50:
        return None, None
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Section 5 — Walk-Forward Training
# ─────────────────────────────────────────────────────────────

def _train_single_sport(X, y, sport):
    """
    Walk-forward split: train on earlier seasons, validate on next, test on recent.
    Returns trained MetaLearner or None.
    """
    n = len(X)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]

    if len(X_train) < 30:
        print(f"  [stack] {sport}: too little training data — skipping")
        return None

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = MetaLearner()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate on held-out test set
    X_test = torch.from_numpy(X[val_end:])
    y_test = y[val_end:]
    if len(X_test) > 0:
        model.eval()
        with torch.no_grad():
            preds = (model(X_test).numpy() > 0.5).astype(float)
        test_acc = float(np.mean(preds == y_test))
        print(f"  [stack] {sport}: test accuracy = {test_acc:.4f} ({len(X_test)} games)")

    return model


# ─────────────────────────────────────────────────────────────
# Section 6 — Train Entry Point
# ─────────────────────────────────────────────────────────────

def train_stack(data):
    """
    Entry point called by brain.py --mode train_models.
    Trains one MetaLearner per sport and saves to stack_models.pkl.
    Updates data['stack'] with training metadata.
    """
    print("[stack] Starting meta-learner training...")
    models = {}
    feature_importance = {}
    validation_accuracy = {}
    improvement = {}

    for sport in SPORTS:
        print(f"  [stack] Building training data for {sport}...")
        X, y = _build_training_data(data, sport)
        if X is None:
            continue

        model = _train_single_sport(X, y, sport)
        if model is None:
            continue

        models[sport] = model

        # Compute rough feature importance via input perturbation
        model.eval()
        X_t = torch.from_numpy(X)
        with torch.no_grad():
            base_preds = model(X_t).numpy()
        base_acc = float(np.mean((base_preds > 0.5).astype(float) == y))

        fi = {}
        feature_names = [
            "raw_prob", "elo", "pythagorean", "h2h", "form", "xgb", "injury",
            "cal_prob", "momentum", "momentum_unc", "cal_correction", "bayes_n",
            "sharp_delta", "sharp_conf", "sentiment_delta", "line_move_mag", "rlm_flag",
            "home_lineup_adj", "away_lineup_adj", "official_impact",
            "home_star_avail", "away_star_avail",
            "week_norm", "is_playoff", "is_division", "home_rest", "away_rest",
            "game_total", "spread_mag", "high_seq_sport",
        ]
        for i, fname in enumerate(feature_names):
            X_perturbed = X.copy()
            X_perturbed[:, i] = np.random.permutation(X_perturbed[:, i])
            with torch.no_grad():
                pert_preds = model(torch.from_numpy(X_perturbed)).numpy()
            pert_acc = float(np.mean((pert_preds > 0.5).astype(float) == y))
            fi[fname] = round(max(0.0, base_acc - pert_acc), 4)

        feature_importance[sport] = fi
        validation_accuracy[sport] = round(base_acc, 4)
        improvement[sport] = {
            "before": round(float(np.mean(y == (X[:, 0] > 0.5).astype(float))), 4),
            "after": round(base_acc, 4),
        }

    # Persist models
    with open(MODELS_FILE, "wb") as f:
        pickle.dump(models, f)
    print(f"[stack] Models saved to {MODELS_FILE}")

    # Update data.json section
    data["stack"] = {
        "version": "1.0",
        "trained": True,
        "last_trained": datetime.utcnow().isoformat(),
        "feature_importance": feature_importance,
        "validation_accuracy": validation_accuracy,
        "improvement_over_brain": improvement,
    }
    print("[stack] Training complete.")


# ─────────────────────────────────────────────────────────────
# Section 7 — Inference
# ─────────────────────────────────────────────────────────────

_loaded_models = {}

def _load_models():
    global _loaded_models
    if _loaded_models:
        return
    if os.path.exists(MODELS_FILE):
        with open(MODELS_FILE, "rb") as f:
            _loaded_models = pickle.load(f)


def stack_predict(game, raw_prob, scout_data, edge_data, mind_data, sport):
    """
    Returns meta-learner probability for a single game.
    Falls back to additive pipeline if models unavailable.
    """
    _load_models()
    model = _loaded_models.get(sport)
    if model is None:
        # Fallback: additive combination
        cal_prob    = float(mind_data.get("calibrated_prob", raw_prob))
        scout_delta = float(scout_data.get("home_lineup_adjustment", 0.0))
        edge_delta  = float(edge_data.get("sharp_prob_delta", 0.0)) * 0.5
        return max(0.02, min(0.98, cal_prob + scout_delta + edge_delta))

    feats = build_meta_features(game, raw_prob, scout_data, edge_data, mind_data, sport)
    model.eval()
    with torch.no_grad():
        prob = float(model(torch.from_numpy(feats).unsqueeze(0)).item())
    return max(0.02, min(0.98, prob))
