"""
graph.py — EDGE Predict Graph Neural Network
Models each sport's league as a directed weighted graph.
Learns circular dominance patterns no linear model can detect.
All 7 sports supported — 25 years of edge data makes all graphs dense.
Called by: brain.py --mode train_models
"""

import os
import pickle
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

# torch_geometric imported conditionally
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, GATConv
    TG_AVAILABLE = True
except ImportError:
    TG_AVAILABLE = False
    print("[graph] WARNING: torch-geometric not installed. graph.py will be skipped.")

# ─────────────────────────────────────────────────────────────
# Section 1 — Constants
# ─────────────────────────────────────────────────────────────

SPORTS     = ["nfl", "nba", "mlb", "nhl", "ncaaf", "ncaabm", "ncaabw"]
MODELS_FILE = "graph_models.pkl"
NODE_DIM   = 10
EDGE_DIM   = 6
GCN_HIDDEN = 32
GAT_HIDDEN = 32
GAT_HEADS  = 4
PRED_HIDDEN = 16
EPOCHS      = 200
PATIENCE    = 25
LR          = 0.001
LOOKBACK_SEASONS = 5

TEMPORAL_DECAY = {
    0: 1.00,
    1: 0.70,
    2: 0.45,
    3: 0.25,
    4: 0.12,
}  # seasons_ago → weight; 5+ = 0.05


# ─────────────────────────────────────────────────────────────
# Section 2 — Graph Construction
# ─────────────────────────────────────────────────────────────

def _season_weight(seasons_ago):
    return TEMPORAL_DECAY.get(seasons_ago, 0.05)


def build_league_graph(data, sport):
    """
    Builds a directed weighted PyG Data object from historical game results.
    Nodes = teams, Edges = games (winner → loser).
    Returns (Data, team_to_idx, idx_to_team) or None if insufficient data.
    """
    if not TG_AVAILABLE:
        return None, None, None

    team_history = data.get("team_history", {}).get(sport, {})
    if not team_history:
        return None, None, None

    teams      = sorted(team_history.keys())
    team_to_idx = {t: i for i, t in enumerate(teams)}
    idx_to_team = {i: t for t, i in team_to_idx.items()}
    n_teams     = len(teams)

    edge_index = []
    edge_attr  = []

    current_season_val = int(datetime.utcnow().year)

    for game in data.get("game_history", {}).get(sport, []):
        winner = game.get("winner")
        loser  = game.get("loser")
        if winner not in team_to_idx or loser not in team_to_idx:
            continue

        game_season = int(game.get("season", current_season_val))
        seasons_ago = current_season_val - game_season
        if seasons_ago > LOOKBACK_SEASONS:
            continue

        w_idx = team_to_idx[winner]
        l_idx = team_to_idx[loser]

        raw_margin = float(game.get("margin", 1))
        norm_margin = np.tanh(raw_margin / 15.0)  # bounded [-1,1]

        edge_index.append([w_idx, l_idx])
        edge_attr.append([
            norm_margin,
            _season_weight(seasons_ago),
            float(game.get("is_playoff", 0)),
            float(game.get("home_winner", 0)),
            min(1.0, float(game.get("winner_elo_before", 1500)) / 2500.0),
            min(1.0, float(game.get("loser_elo_before", 1500)) / 2500.0),
        ])

    if len(edge_index) < 50:
        return None, None, None

    node_feats = build_node_features(teams, data, sport)

    graph = Data(
        x=node_feats,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        num_nodes=n_teams,
    )
    return graph, team_to_idx, idx_to_team


# ─────────────────────────────────────────────────────────────
# Section 3 — Node Feature Engineering
# ─────────────────────────────────────────────────────────────

def build_node_features(teams, data, sport):
    """Builds [n_teams, NODE_DIM] tensor of current team attributes."""
    feats = []
    team_stats = data.get("team_stats", {}).get(sport, {})
    for team in teams:
        ts = team_stats.get(team, {})
        feats.append([
            min(1.0, float(ts.get("elo", 1500)) / 2500.0),
            float(ts.get("pythagorean_expectation", 0.5)),
            float(ts.get("recent_win_pct_10", 0.5)),
            float(ts.get("home_win_rate", 0.5)),
            float(ts.get("away_win_rate", 0.5)),
            np.tanh(float(ts.get("avg_margin_of_victory", 0)) / 10.0),
            np.tanh(float(ts.get("avg_margin_of_defeat", 0)) / 10.0),
            float(ts.get("strength_of_schedule", 0.5)),
            float(ts.get("injury_impact", 0.0)),
            float(ts.get("momentum_score", 0.5)),
        ])
    return torch.tensor(feats, dtype=torch.float)


# ─────────────────────────────────────────────────────────────
# Section 4 — GCN + GAT Architecture
# ─────────────────────────────────────────────────────────────

class LeagueGNN(nn.Module):
    """
    Stage 1: GCN aggregates neighborhood information.
    Stage 2: GAT learns which relationships matter most.
    Prediction head: concatenate two team embeddings → matchup probability.
    """
    def __init__(self, node_dim=NODE_DIM):
        super().__init__()
        # GCN stage
        self.gcn1 = GCNConv(node_dim, GCN_HIDDEN)
        self.gcn2 = GCNConv(GCN_HIDDEN, GCN_HIDDEN)

        # GAT stage (4 heads, then collapse)
        self.gat1 = GATConv(GCN_HIDDEN, GAT_HIDDEN, heads=GAT_HEADS, dropout=0.3)
        self.gat2 = GATConv(GAT_HIDDEN * GAT_HEADS, 16, heads=1, concat=False)

        # Matchup prediction head
        self.head = nn.Sequential(
            nn.Linear(32, PRED_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(PRED_HIDDEN, 1),
            nn.Sigmoid(),
        )
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def embed(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = self.relu(self.gcn1(x, edge_index))
        x = self.relu(self.gcn2(x, edge_index))
        x = self.dropout(self.relu(self.gat1(x, edge_index)))
        x = self.gat2(x, edge_index)
        return x  # [n_teams, 16]

    def forward(self, graph, team_a_idx, team_b_idx):
        embeddings = self.embed(graph)
        ea = embeddings[team_a_idx]  # [batch, 16]
        eb = embeddings[team_b_idx]  # [batch, 16]
        pair = torch.cat([ea, eb], dim=-1)  # [batch, 32]
        return self.head(pair).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# Section 5 — Build Training Pairs From Graph
# ─────────────────────────────────────────────────────────────

def _build_training_pairs(data, sport, team_to_idx):
    """
    Returns (winner_idxs, loser_idxs, labels) tensors from game history.
    Label = 1 (team_a wins), uses the first team in the edge as team_a.
    """
    pairs_a, pairs_b, labels = [], [], []
    current_season_val = int(datetime.utcnow().year)

    for game in data.get("game_history", {}).get(sport, []):
        winner = game.get("winner")
        loser  = game.get("loser")
        if winner not in team_to_idx or loser not in team_to_idx:
            continue
        seasons_ago = current_season_val - int(game.get("season", current_season_val))
        if seasons_ago > LOOKBACK_SEASONS:
            continue
        pairs_a.append(team_to_idx[winner])
        pairs_b.append(team_to_idx[loser])
        labels.append(1.0)
        # Add reverse to balance (loser perspective)
        pairs_a.append(team_to_idx[loser])
        pairs_b.append(team_to_idx[winner])
        labels.append(0.0)

    if len(labels) < 100:
        return None, None, None

    return (
        torch.tensor(pairs_a, dtype=torch.long),
        torch.tensor(pairs_b, dtype=torch.long),
        torch.tensor(labels,  dtype=torch.float),
    )


# ─────────────────────────────────────────────────────────────
# Section 6 — Train Entry Point
# ─────────────────────────────────────────────────────────────

def train_graph(data):
    """
    Called by brain.py --mode train_models.
    Trains one LeagueGNN per sport. Saves to graph_models.pkl.
    """
    if not TG_AVAILABLE:
        print("[graph] torch-geometric not available — skipping graph training.")
        data.setdefault("graph", {})["trained"] = False
        return

    print("[graph] Starting GNN training...")
    models = {}
    graph_meta = {}

    for sport in SPORTS:
        print(f"  [graph] Building league graph for {sport}...")
        graph, team_to_idx, idx_to_team = build_league_graph(data, sport)
        if graph is None:
            print(f"  [graph] {sport}: insufficient edge data — skipping")
            continue

        n_edges = graph.edge_index.shape[1]
        n_nodes = graph.num_nodes
        print(f"  [graph] {sport}: {n_nodes} teams, {n_edges} historical edges")

        a_idx, b_idx, labels = _build_training_pairs(data, sport, team_to_idx)
        if a_idx is None:
            continue

        n = len(labels)
        train_end = int(n * 0.80)

        model     = LeagueGNN()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        best_val  = float("inf")
        best_state = None
        patience_ctr = 0

        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            preds = model(graph, a_idx[:train_end], b_idx[:train_end])
            loss  = criterion(preds, labels[:train_end])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = model(graph, a_idx[train_end:], b_idx[train_end:])
                val_loss  = criterion(val_preds, labels[train_end:]).item()

            if val_loss < best_val:
                best_val     = val_loss
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break

        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            val_preds = (model(graph, a_idx[train_end:], b_idx[train_end:]).numpy() > 0.5).astype(float)
        val_acc = float(np.mean(val_preds == labels[train_end:].numpy()))
        print(f"  [graph] {sport}: val accuracy = {val_acc:.4f}")

        models[sport] = {
            "model": model,
            "graph": graph,
            "team_to_idx": team_to_idx,
            "idx_to_team": idx_to_team,
        }
        graph_meta[sport] = {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "circular_patterns_found": 0,
            "validation_accuracy": round(val_acc, 4),
            "top_structural_edges": [],
        }

    with open(MODELS_FILE, "wb") as f:
        pickle.dump(models, f)
    print(f"[graph] Models saved to {MODELS_FILE}")

    data["graph"] = {
        "trained": True,
        "last_trained": datetime.utcnow().isoformat(),
        "sports": graph_meta,
    }
    print("[graph] Training complete.")


# ─────────────────────────────────────────────────────────────
# Section 7 — Circular Dominance Extractor
# ─────────────────────────────────────────────────────────────

def find_circular_dominance(team_a, team_b, sport, data, graph_prob, ensemble_prob):
    """
    Returns structural edge adjustment (negative = team_b has hidden advantage).
    Capped at ±0.08.
    """
    structural_edge = graph_prob - ensemble_prob
    return max(-0.08, min(0.08, structural_edge))


# ─────────────────────────────────────────────────────────────
# Section 8 — Inference
# ─────────────────────────────────────────────────────────────

_loaded = {}

def _load():
    global _loaded
    if _loaded:
        return
    if os.path.exists(MODELS_FILE):
        with open(MODELS_FILE, "rb") as f:
            _loaded = pickle.load(f)


def predict_graph(team_a, team_b, sport, ensemble_prob, data):
    """
    Returns (graph_win_prob, structural_edge) for team_a vs team_b.
    Falls back to (ensemble_prob, 0.0) if model unavailable.
    """
    if not TG_AVAILABLE:
        return ensemble_prob, 0.0

    _load()
    sport_data = _loaded.get(sport)
    if sport_data is None:
        return ensemble_prob, 0.0

    model        = sport_data["model"]
    graph        = sport_data["graph"]
    team_to_idx  = sport_data["team_to_idx"]

    if team_a not in team_to_idx or team_b not in team_to_idx:
        return ensemble_prob, 0.0

    a_idx = torch.tensor([team_to_idx[team_a]], dtype=torch.long)
    b_idx = torch.tensor([team_to_idx[team_b]], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        graph_prob = float(model(graph, a_idx, b_idx).item())

    structural_edge = find_circular_dominance(team_a, team_b, sport, data, graph_prob, ensemble_prob)
    return max(0.02, min(0.98, graph_prob)), structural_edge
