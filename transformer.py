"""
transformer.py — EDGE Predict Transformer Momentum Model
Reads full game sequences with self-attention to find non-local momentum patterns.
Active sports: NBA, MLB, NHL, NCAABM, NCAABW (not NFL/NCAAF — too few games/season).
Called by: brain.py --mode train_models
"""

import os
import math
import pickle
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────
# Section 1 — Constants
# ─────────────────────────────────────────────────────────────

ACTIVE_SPORTS   = ["nba", "mlb", "nhl", "ncaabm", "ncaabw"]
INACTIVE_SPORTS = ["nfl", "ncaaf"]
MODELS_FILE     = "transformer_models.pkl"
SEQ_LENGTHS     = {"nba": 20, "mlb": 20, "nhl": 15, "ncaabm": 15, "ncaabw": 15}
FEATURE_DIM     = 12  # Same 12-feature vector as mind.py LSTM
D_MODEL         = 64
NUM_HEADS       = 4
NUM_LAYERS      = 3
D_FF            = 128
DROPOUT         = 0.2
EPOCHS          = 150
PATIENCE        = 20
BATCH_SIZE      = 32
LR              = 0.001

# The 12 game-level features (same as mind.py)
GAME_FEATURES = [
    "win",              # 1/0
    "margin",           # Normalized margin of victory/defeat
    "home_flag",        # Was this game at home?
    "opponent_elo",     # Opponent strength (normalized)
    "scored",           # Points scored (normalized by sport)
    "allowed",          # Points allowed (normalized)
    "rest_days",        # Days rest (normalized)
    "back_to_back",     # 1 if B2B
    "cumulative_win_pct",  # Running win% up to this game
    "rolling_margin_5", # 5-game rolling avg margin
    "opp_win_pct",      # Opponent's win% at time of game
    "is_playoff",       # Playoff game flag
]


# ─────────────────────────────────────────────────────────────
# Section 2 — Positional Encoding (Recency-Aware)
# ─────────────────────────────────────────────────────────────

def recency_positional_encoding(seq_len, d_model):
    """
    Positional encoding where position 0 (most recent game) gets highest weight.
    Earlier games get exponentially lower positional weight.
    """
    pe = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        recency_weight = 0.95 ** pos
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (2 * i / d_model))
            pe[pos, i]     = recency_weight * math.sin(angle)
            if i + 1 < d_model:
                pe[pos, i + 1] = recency_weight * math.cos(angle)
    return pe  # [seq_len, d_model]


# ─────────────────────────────────────────────────────────────
# Section 3 — Transformer Architecture
# ─────────────────────────────────────────────────────────────

class TransformerMomentum(nn.Module):
    """
    Input:  [batch, seq_len, FEATURE_DIM]
    Output: momentum score [0, 1] per team
    """
    def __init__(self, seq_len, feature_dim=FEATURE_DIM, d_model=D_MODEL,
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.seq_len = seq_len

        # Project input features up to d_model
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Register positional encoding as buffer (not a parameter)
        pe = recency_positional_encoding(seq_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, seq_len, d_model]

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [batch, seq_len, feature_dim]
        x = self.input_proj(x) + self.pe          # [batch, seq_len, d_model]
        x = self.transformer(x)                    # [batch, seq_len, d_model]
        x = x.mean(dim=1)                          # Global average pool: [batch, d_model]
        return self.head(x).squeeze(-1)            # [batch]


# ─────────────────────────────────────────────────────────────
# Section 4 — Data Preparation
# ─────────────────────────────────────────────────────────────

def _normalize_game_features(game, sport):
    """
    Extracts and normalizes the 12-feature vector for a single game entry.
    """
    score_scale = {"nba": 120.0, "mlb": 10.0, "nhl": 5.0, "ncaabm": 80.0, "ncaabw": 75.0}.get(sport, 100.0)
    margin_scale = score_scale / 2.0

    return np.array([
        float(game.get("win", 0)),
        float(game.get("margin", 0)) / margin_scale,
        float(game.get("home_flag", 0)),
        float(game.get("opponent_elo", 1500)) / 2000.0,
        float(game.get("scored", 0)) / score_scale,
        float(game.get("allowed", 0)) / score_scale,
        min(1.0, float(game.get("rest_days", 1)) / 7.0),
        float(game.get("back_to_back", 0)),
        float(game.get("cumulative_win_pct", 0.5)),
        float(game.get("rolling_margin_5", 0)) / margin_scale,
        float(game.get("opp_win_pct", 0.5)),
        float(game.get("is_playoff", 0)),
    ], dtype=np.float32)


def _build_team_histories_from_raw(raw_games, sport):
    """
    Converts a flat list of raw game dicts (from data['all_games_by_sport'])
    into per-team history dicts compatible with _build_sequences.

    Each resulting entry contains all 12 fields that _normalize_game_features
    expects. Fields computed here rather than defaulted:

      rest_days         — real calendar days since the team's previous game.
                          Defaults to 7 for a team's first game in the dataset.
      back_to_back      — 1.0 if rest_days <= 1, else 0.0.
      opp_win_pct       — opponent's running win% at the moment this game is
                          played (computed from concurrent records as we process
                          games in chronological order). Defaults to 0.5 until
                          the opponent has played at least one prior game.
      cumulative_win_pct — team's running win% up to (but not including) this
                          game. Defaults to 0.5 for the team's first game.
      rolling_margin_5  — average margin of the team's last 5 games.

    opponent_elo is left at the neutral default of 1500 because raw game dicts
    from fetch_season do not carry ELO values — those live in data['elo_ratings']
    which is keyed by current team state, not per historical game.
    """
    sorted_games = sorted(raw_games, key=lambda g: g.get("date", ""))

    # Per-team state tracked as we process games chronologically.
    team_histories:   dict = {}   # team -> list of feature dicts (output)
    team_wins:        dict = {}   # team -> int  (wins so far)
    team_played:      dict = {}   # team -> int  (games played so far)
    team_last_date:   dict = {}   # team -> str  (date of most recent game, YYYY-MM-DD)
    team_margins:     dict = {}   # team -> list (recent margins, capped at 5)

    for game in sorted_games:
        home = game.get("home", "")
        away = game.get("away", "")
        if not home or not away:
            continue

        home_score = game.get("home_score", 0) or 0
        away_score = game.get("away_score", 0) or 0
        home_won   = bool(game.get("home_won", home_score > away_score))
        margin     = home_score - away_score
        date_str   = game.get("date", "")
        is_playoff = float(game.get("is_playoff", 0))

        # Snapshot concurrent opponent win% BEFORE updating records for this game.
        # This gives the true opponent strength at game time, not post-game.
        home_opp_win_pct = (
            team_wins.get(away, 0) / team_played[away]
            if team_played.get(away, 0) > 0 else 0.5
        )
        away_opp_win_pct = (
            team_wins.get(home, 0) / team_played[home]
            if team_played.get(home, 0) > 0 else 0.5
        )

        # Process both perspectives (home team entry, away team entry).
        for is_home, team, scored, allowed, won, team_margin, opp_win_pct in [
            (True,  home, home_score, away_score,  home_won,  margin,  home_opp_win_pct),
            (False, away, away_score, home_score, not home_won, -margin, away_opp_win_pct),
        ]:
            # Initialise team state on first encounter.
            if team not in team_histories:
                team_histories[team]  = []
                team_wins[team]       = 0
                team_played[team]     = 0
                team_last_date[team]  = ""
                team_margins[team]    = []

            # Snapshot pre-game stats (before this game updates the record).
            wins   = team_wins[team]
            played = team_played[team]
            cum_win_pct = wins / played if played > 0 else 0.5

            # Rest days: days elapsed since last game for this team.
            prev_date = team_last_date[team]
            if prev_date and date_str and len(date_str) >= 10 and len(prev_date) >= 10:
                try:
                    delta = (
                        datetime.strptime(date_str[:10], "%Y-%m-%d")
                        - datetime.strptime(prev_date[:10], "%Y-%m-%d")
                    ).days
                    rest_days = max(0, delta)
                except ValueError:
                    rest_days = 7   # Unparseable date — use neutral default
            else:
                rest_days = 7       # First game in dataset — no prior date known

            back_to_back = 1.0 if rest_days <= 1 else 0.0

            rolling5 = (
                float(np.mean(team_margins[team][-5:]))
                if team_margins[team] else 0.0
            )

            team_histories[team].append({
                "date":               date_str,
                "win":                float(won),
                "margin":             float(team_margin),
                "home_flag":          float(is_home),
                "opponent_elo":       1500.0,           # No per-game ELO in raw data
                "scored":             float(scored),
                "allowed":            float(allowed),
                "rest_days":          float(rest_days),
                "back_to_back":       back_to_back,
                "cumulative_win_pct": cum_win_pct,
                "rolling_margin_5":   rolling5,
                "opp_win_pct":        opp_win_pct,
                "is_playoff":         is_playoff,
            })

            # Update state AFTER building the feature entry.
            team_wins[team]      = wins + (1 if won else 0)
            team_played[team]    = played + 1
            team_last_date[team] = date_str
            team_margins[team].append(float(team_margin))
            if len(team_margins[team]) > 5:
                team_margins[team].pop(0)

    return team_histories


def _build_sequences(data, sport):
    """
    Builds (sequence_tensor, label) pairs from team game histories.
    Label = 1 if team won their next game, 0 otherwise.
    Primary source: data['all_games_by_sport'] (raw game list, built by foundation training).
    Fallback: data['team_history'] (limited to MAX_TEAM_HISTORY=15 per team).
    """
    seq_len = SEQ_LENGTHS.get(sport, 15)

    # Primary: use raw game history stored by foundation training
    raw_games = data.get("all_games_by_sport", {}).get(sport, [])
    if raw_games:
        team_histories = _build_team_histories_from_raw(raw_games, sport)
    else:
        # Fallback to team_history (capped at 15 games/team — may be insufficient)
        team_histories = data.get("team_history", {}).get(sport, {})

    sequences, labels = [], []
    for team, games in team_histories.items():
        # games sorted oldest → newest
        game_list = sorted(games, key=lambda g: g.get("date", ""))
        feats = [_normalize_game_features(g, sport) for g in game_list]

        for i in range(seq_len, len(feats) - 1):
            # Sequence: last seq_len games (most recent = feats[i])
            seq = np.array(feats[i - seq_len + 1: i + 1])  # [seq_len, 12], oldest→newest
            seq = seq[::-1].copy()  # Flip so index 0 = most recent
            next_game_win = float(game_list[i + 1].get("win", 0))
            sequences.append(seq)
            labels.append(next_game_win)

    if len(sequences) < 100:
        return None, None

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Section 5 — Training Loop
# ─────────────────────────────────────────────────────────────

def _train_sport_model(X, y, sport):
    seq_len = SEQ_LENGTHS.get(sport, 15)
    n = len(X)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_tr, y_tr = X[:train_end], y[:train_end]
    X_vl, y_vl = X[train_end:val_end], y[train_end:val_end]

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_vl), torch.from_numpy(y_vl)),
        batch_size=BATCH_SIZE
    )

    model     = TransformerMomentum(seq_len=seq_len)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                val_losses.append(criterion(model(xb), yb).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)

    # Test accuracy
    X_test = torch.from_numpy(X[val_end:])
    y_test = y[val_end:]
    if len(X_test) > 0:
        model.eval()
        with torch.no_grad():
            preds = (model(X_test).numpy() > 0.5).astype(float)
        acc = float(np.mean(preds == y_test))
        print(f"  [transformer] {sport}: test accuracy = {acc:.4f} ({len(X_test)} sequences)")

    return model


# ─────────────────────────────────────────────────────────────
# Section 6 — Train Entry Point
# ─────────────────────────────────────────────────────────────

def train_transformer(data):
    """
    Called by brain.py --mode train_models.
    Trains one TransformerMomentum per active sport.
    Saves to transformer_models.pkl and updates data['transformer'].
    """
    print("[transformer] Starting training...")
    models = {}
    val_accuracy = {}

    for sport in ACTIVE_SPORTS:
        print(f"  [transformer] Building sequences for {sport}...")
        X, y = _build_sequences(data, sport)
        if X is None:
            print(f"  [transformer] {sport}: insufficient data — skipping")
            continue

        print(f"  [transformer] {sport}: {len(X)} sequences, seq_len={SEQ_LENGTHS[sport]}")
        model = _train_sport_model(X, y, sport)
        models[sport] = model

    with open(MODELS_FILE, "wb") as f:
        pickle.dump(models, f)
    print(f"[transformer] Models saved to {MODELS_FILE}")

    data["transformer"] = {
        "trained": True,
        "sports_active": ACTIVE_SPORTS,
        "sports_inactive": INACTIVE_SPORTS,
        "last_trained": datetime.utcnow().isoformat(),
        "validation_accuracy": val_accuracy,
    }
    print("[transformer] Training complete.")


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


def predict_transformer(team, sport, data):
    """
    Returns transformer momentum score [0, 1] for a team.
    Falls back to LSTM momentum score if transformer unavailable for this sport.
    """
    if sport in INACTIVE_SPORTS:
        # Not applicable — return LSTM score as pass-through
        return float(data.get("mind", {}).get("team_momentum", {}).get(team, 0.5))

    _load_models()
    model = _loaded_models.get(sport)
    if model is None:
        return float(data.get("mind", {}).get("team_momentum", {}).get(team, 0.5))

    seq_len = SEQ_LENGTHS.get(sport, 15)
    team_games = data.get("team_history", {}).get(sport, {}).get(team, [])
    if len(team_games) < seq_len:
        return float(data.get("mind", {}).get("team_momentum", {}).get(team, 0.5))

    recent = sorted(team_games, key=lambda g: g.get("date", ""))[-seq_len:]
    feats = np.array([_normalize_game_features(g, sport) for g in recent], dtype=np.float32)
    feats = feats[::-1].copy()  # Most recent first

    model.eval()
    x = torch.from_numpy(feats).unsqueeze(0)  # [1, seq_len, 12]
    with torch.no_grad():
        score = float(model(x).item())
    return max(0.0, min(1.0, score))
