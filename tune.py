"""
tune.py — EDGE Predict Hyperparameter Optimization
Uses Optuna to find globally optimal configurations for XGBoost, ensemble weights, and LSTM.
Called by: brain.py --mode train --tune
NOT called during daily update runs.
"""

import numpy as np
from datetime import datetime

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[tune] WARNING: optuna not installed. Run: pip install optuna>=3.0.0")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from sklearn.metrics import accuracy_score, log_loss

# ─────────────────────────────────────────────────────────────
# Section 1 — Constants
# ─────────────────────────────────────────────────────────────

SPORTS              = ["nfl", "nba", "mlb", "nhl", "ncaaf", "ncaabm", "ncaabw"]
XGB_TRIALS          = 200
ENSEMBLE_TRIALS     = 500
LSTM_TRIALS         = 100
XGB_TIMEOUT_SECS    = 1800   # 30 min per sport
ENSEMBLE_TIMEOUT    = 1500   # 25 min per sport
LSTM_TIMEOUT        = 2400   # 40 min total for LSTM


# ─────────────────────────────────────────────────────────────
# Section 2 — XGBoost Search Space
# ─────────────────────────────────────────────────────────────

def _xgb_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 1.0),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_preds = model.predict(X_val)
    return accuracy_score(y_val, val_preds)


def tune_xgboost(data, sport):
    """
    Runs Optuna TPE search over XGBoost hyperparameters for one sport.
    Returns best_params dict or None.
    """
    if not OPTUNA_AVAILABLE or not XGB_AVAILABLE:
        return None

    print(f"  [tune] XGBoost tuning for {sport} ({XGB_TRIALS} trials)...")

    # Build feature matrix from brain.py's training data
    training_data = data.get("xgb_training", {}).get(sport, {})
    X = np.array(training_data.get("X", []), dtype=np.float32)
    y = np.array(training_data.get("y", []), dtype=np.float32)

    if len(X) < 200:
        print(f"  [tune] {sport}: insufficient XGB training data — skipping")
        return None

    split = int(len(X) * 0.80)
    X_tr, X_vl = X[:split], X[split:]
    y_tr, y_vl = y[:split], y[split:]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: _xgb_objective(trial, X_tr, y_tr, X_vl, y_vl),
        n_trials=XGB_TRIALS,
        timeout=XGB_TIMEOUT_SECS,
        show_progress_bar=False,
    )

    best = study.best_params
    best_val = study.best_value
    print(f"  [tune] {sport} XGB best val accuracy: {best_val:.4f}")
    return best


# ─────────────────────────────────────────────────────────────
# Section 3 — Ensemble Weight Optimization
# ─────────────────────────────────────────────────────────────

def _ensemble_objective(trial, games, actual_outcomes):
    weights = {
        "elo":         trial.suggest_float("elo",         0.0, 0.30),
        "pythagorean": trial.suggest_float("pythagorean", 0.0, 0.30),
        "h2h":         trial.suggest_float("h2h",         0.0, 0.20),
        "form":        trial.suggest_float("form",        0.0, 0.20),
        "home_field":  trial.suggest_float("home_field",  0.0, 0.15),
        "xgboost":     trial.suggest_float("xgboost",     0.0, 0.40),
        "rest":        trial.suggest_float("rest",        0.0, 0.10),
        "weather":     trial.suggest_float("weather",     0.0, 0.10),
    }
    total = sum(weights.values())
    if total < 1e-9:
        return 0.0
    weights = {k: v / total for k, v in weights.items()}

    correct = 0
    for game, outcome in zip(games, actual_outcomes):
        prob = sum(
            weights.get(k, 0.0) * float(game.get(f"{k}_component", 0.5))
            for k in weights
        )
        predicted = 1 if prob > 0.5 else 0
        if predicted == int(outcome):
            correct += 1

    return correct / len(games) if games else 0.0


def tune_ensemble_weights(data, sport):
    """
    Optuna search over per-component ensemble weights for one sport.
    Returns normalized weight dict or None.
    """
    if not OPTUNA_AVAILABLE:
        return None

    print(f"  [tune] Ensemble weight tuning for {sport} ({ENSEMBLE_TRIALS} trials)...")

    history = data.get("prediction_history", {}).get(sport, [])
    resolved = [r for r in history if r.get("outcome") is not None]

    if len(resolved) < 100:
        print(f"  [tune] {sport}: insufficient resolved predictions — skipping")
        return None

    games    = [r.get("game", {}) for r in resolved]
    outcomes = [r["outcome"] for r in resolved]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: _ensemble_objective(trial, games, outcomes),
        n_trials=ENSEMBLE_TRIALS,
        timeout=ENSEMBLE_TIMEOUT,
        show_progress_bar=False,
    )

    best = study.best_params
    total = sum(best.values())
    normalized = {k: round(v / total, 4) for k, v in best.items()}
    print(f"  [tune] {sport} ensemble best val accuracy: {study.best_value:.4f}")
    print(f"  [tune] {sport} optimal weights: {normalized}")
    return normalized


# ─────────────────────────────────────────────────────────────
# Section 4 — LSTM Architecture Search
# ─────────────────────────────────────────────────────────────

def _build_sequences_for_tune(data, sport, seq_len):
    """Reuses mind.py-style sequence building for validation."""
    team_histories = data.get("team_history", {}).get(sport, {})
    sequences, labels = [], []
    for team, games in team_histories.items():
        sorted_games = sorted(games, key=lambda g: g.get("date", ""))
        for i in range(seq_len, len(sorted_games) - 1):
            seq = np.array([
                [float(g.get("win", 0)),
                 float(g.get("margin", 0)) / 20.0,
                 float(g.get("home_flag", 0)),
                 float(g.get("opponent_elo", 1500)) / 2000.0,
                 float(g.get("scored", 0)) / 100.0,
                 float(g.get("allowed", 0)) / 100.0,
                 min(1.0, float(g.get("rest_days", 1)) / 7.0),
                 float(g.get("back_to_back", 0)),
                 float(g.get("cumulative_win_pct", 0.5)),
                 float(g.get("rolling_margin_5", 0)) / 20.0,
                 float(g.get("opp_win_pct", 0.5)),
                 float(g.get("is_playoff", 0))]
                for g in sorted_games[i - seq_len: i]
            ], dtype=np.float32)
            sequences.append(seq)
            labels.append(float(sorted_games[i + 1].get("win", 0)))
    return sequences, labels


def _lstm_objective(trial, sequences, labels):
    """Trains a small LSTM with the trial config and returns validation loss."""
    import torch
    import torch.nn as nn

    config = {
        "hidden_size":  trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
        "num_layers":   trial.suggest_int("num_layers", 1, 4),
        "dropout":      trial.suggest_float("dropout", 0.1, 0.5),
        "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "seq_len":      trial.suggest_categorical("seq_len", [5, 8, 10, 15, 20]),
        "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
    }

    seq_len = config["seq_len"]
    X = np.array([s[-seq_len:] for s in sequences if len(s) >= seq_len], dtype=np.float32)
    y = np.array([labels[i] for i, s in enumerate(sequences) if len(s) >= seq_len], dtype=np.float32)

    if len(X) < 50:
        return float("inf")

    split = int(len(X) * 0.80)
    X_tr = torch.from_numpy(X[:split])
    y_tr = torch.from_numpy(y[:split])
    X_vl = torch.from_numpy(X[split:])
    y_vl = torch.from_numpy(y[split:])

    class QuickLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=12,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                dropout=config["dropout"] if config["num_layers"] > 1 else 0,
                bidirectional=config["bidirectional"],
                batch_first=True,
            )
            out_size = config["hidden_size"] * (2 if config["bidirectional"] else 1)
            self.fc = nn.Sequential(nn.Linear(out_size, 1), nn.Sigmoid())

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model     = QuickLSTM()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for _ in range(30):  # Quick training
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_vl), y_vl).item()
    return val_loss


def tune_lstm(data, sport):
    """
    Optuna search over LSTM architecture for one sport.
    Returns best config dict or None.
    """
    if not OPTUNA_AVAILABLE:
        return None

    print(f"  [tune] LSTM tuning for {sport} ({LSTM_TRIALS} trials)...")

    sequences, labels = _build_sequences_for_tune(data, sport, seq_len=20)
    if len(sequences) < 200:
        print(f"  [tune] {sport}: insufficient sequence data — skipping LSTM tune")
        return None

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: _lstm_objective(trial, sequences, labels),
        n_trials=LSTM_TRIALS,
        timeout=LSTM_TIMEOUT // len(SPORTS),
        show_progress_bar=False,
    )

    best = study.best_params
    print(f"  [tune] {sport} LSTM best val loss: {study.best_value:.4f}")
    return best


# ─────────────────────────────────────────────────────────────
# Section 5 — Main Entry Point
# ─────────────────────────────────────────────────────────────

SPORTS = ["nfl", "nba", "mlb", "nhl", "ncaaf", "ncaabm", "ncaabw"]


def run_tune(data):
    """
    Called by brain.py --mode train --tune.
    Runs XGBoost, ensemble, and LSTM tuning for all sports.
    Saves best configs to data['tune']['best_configs'].
    """
    if not OPTUNA_AVAILABLE:
        print("[tune] Optuna not available. Install with: pip install optuna>=3.0.0")
        return

    print("[tune] Starting hyperparameter optimization (this will take ~2 hours)...")
    best_configs = {}
    validation_improvements = {}

    for sport in SPORTS:
        print(f"\n[tune] ─── {sport.upper()} ───")
        sport_config = {}

        # XGBoost
        xgb_best = tune_xgboost(data, sport)
        if xgb_best:
            sport_config["xgb"] = xgb_best

        # Ensemble weights
        ens_best = tune_ensemble_weights(data, sport)
        if ens_best:
            sport_config["ensemble_weights"] = ens_best

        # LSTM
        lstm_best = tune_lstm(data, sport)
        if lstm_best:
            sport_config["lstm"] = lstm_best

        best_configs[sport] = sport_config

        # Compute improvement estimate
        history = data.get("prediction_history", {}).get(sport, [])
        resolved = [r for r in history if r.get("outcome") is not None]
        if resolved:
            before_acc = float(np.mean([
                1 if (r.get("raw_win_probability", 0.5) > 0.5) == int(r["outcome"]) else 0
                for r in resolved
            ]))
            validation_improvements[sport] = {
                "before": round(before_acc, 4),
                "after": None,  # Will be populated when models are retrained with new configs
            }

    data.setdefault("tune", {}).update({
        "last_tuned": datetime.utcnow().isoformat(),
        "best_configs": best_configs,
        "validation_improvements": validation_improvements,
    })

    print("\n[tune] Optimization complete. Best configs saved to data['tune']['best_configs'].")
    print("[tune] Rerun foundation training to apply new configs.")
