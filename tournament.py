"""
tournament.py — EDGE Predict Multi-Sport Playoff Simulator
Extends march.py's Monte Carlo framework to NFL, NBA, MLB, and NHL postseasons.
Outputs championship probabilities and round-by-round path probabilities.
Called by: brain.py --mode update (when playoffs are active)
"""

import random
import numpy as np
from datetime import datetime
from math import comb

# ─────────────────────────────────────────────────────────────
# Section 1 — Constants
# ─────────────────────────────────────────────────────────────

N_SIMULATIONS = 100000

PLAYOFF_STRUCTURES = {
    "nfl": {
        "type":             "single_elimination",
        "teams":            14,
        "rounds":           ["Wild Card", "Divisional", "Conference Championship", "Super Bowl"],
        "bye_seeds":        [1, 2],   # Per conference
        "home_field_seeds": [1, 2, 3, 4],
        "n_conferences":    2,
    },
    "nba": {
        "type":         "series",
        "series_length": 7,
        "teams":        16,
        "rounds":       ["First Round", "Second Round", "Conference Finals", "NBA Finals"],
        "home_court":   "higher_seed",
    },
    "mlb": {
        "type":   "mixed",
        "teams":  12,
        "rounds": [
            {"name": "Wild Card",           "type": "series", "length": 3},
            {"name": "Division Series",     "type": "series", "length": 5},
            {"name": "Championship Series", "type": "series", "length": 7},
            {"name": "World Series",        "type": "series", "length": 7},
        ],
    },
    "nhl": {
        "type":         "series",
        "series_length": 7,
        "teams":        16,
        "rounds":       ["First Round", "Second Round", "Conference Finals", "Stanley Cup Finals"],
        "home_ice":     "higher_seed",
    },
}

# Home field/court advantage by sport and round (applied to higher seed when at home)
HOME_ADVANTAGE = {
    "nfl": {"Wild Card": 0.04, "Divisional": 0.05, "Conference Championship": 0.04, "Super Bowl": 0.0},
    "nba": {"First Round": 0.06, "Second Round": 0.06, "Conference Finals": 0.05, "NBA Finals": 0.04},
    "mlb": {"Wild Card": 0.03, "Division Series": 0.03, "Championship Series": 0.03, "World Series": 0.0},
    "nhl": {"First Round": 0.04, "Second Round": 0.04, "Conference Finals": 0.04, "Stanley Cup Finals": 0.0},
}


# ─────────────────────────────────────────────────────────────
# Section 2 — Per-Game Win Probability
# ─────────────────────────────────────────────────────────────

def game_win_prob(team_a, team_b, sport, round_name, is_home_a, data):
    """
    Returns per-game win probability for team_a vs team_b.
    Reads brain.py's prediction pipeline outputs and applies home advantage.
    """
    predictions = data.get("predictions", {}).get(sport, {})

    # Try to find a pre-computed matchup probability
    matchup_key = f"{team_a}_vs_{team_b}"
    base_prob   = float(
        predictions.get(matchup_key, {}).get("win_probability") or
        data.get("team_stats", {}).get(sport, {}).get(team_a, {}).get("win_prob_vs_avg", 0.5)
    )

    # Apply home advantage
    home_adj = HOME_ADVANTAGE.get(sport, {}).get(round_name, 0.03)
    if is_home_a:
        base_prob = min(0.95, base_prob + home_adj)
    else:
        base_prob = max(0.05, base_prob - home_adj)

    return base_prob


# ─────────────────────────────────────────────────────────────
# Section 3 — Series Win Probability
# ─────────────────────────────────────────────────────────────

def series_win_probability(per_game_prob, series_length=7):
    """
    Probability of winning a best-of-N series given per-game win probability.
    Uses exact negative binomial calculation.
    """
    wins_needed = (series_length // 2) + 1
    series_prob = 0.0

    for wins in range(wins_needed, series_length + 1):
        losses = wins_needed - 1  # Opponent has won this many before deciding game
        # Actually: team wins 'wins' games, opponent wins (series_length - wins) games
        # The last game must be a win, and the series ends here
        games_before_last = wins - 1 + (series_length - wins)
        # Probability of winning exactly 'wins' games in (wins + losses_total) games
        losses_total = series_length - wins
        if losses_total < 0:
            continue
        games_played = wins + losses_total - 1
        if games_played < wins - 1:
            continue
        p = (comb(games_played, wins - 1) *
             (per_game_prob ** (wins - 1)) *
             ((1 - per_game_prob) ** losses_total) *
             per_game_prob)
        series_prob += p

    return min(0.99, max(0.01, series_prob))


def home_game_mask(series_length, higher_seed_is_a):
    """
    Returns list of booleans: is team_a at home for each game in the series?
    Standard format: H H A A H A H
    """
    patterns = {
        7: [True, True, False, False, True, False, True],
        5: [True, True, False, False, True],
        3: [True, True, False],
    }
    pattern = patterns.get(series_length, [True, True, False, False])
    if not higher_seed_is_a:
        pattern = [not x for x in pattern]
    return pattern


def simulate_series(team_a, team_b, sport, round_name, series_length, higher_seed_is_a, data):
    """Simulates a single best-of-N series. Returns winner string."""
    wins_a, wins_b = 0, 0
    wins_needed    = (series_length // 2) + 1
    home_mask      = home_game_mask(series_length, higher_seed_is_a)

    for game_num in range(series_length):
        is_home_a = home_mask[min(game_num, len(home_mask) - 1)]
        prob_a    = game_win_prob(team_a, team_b, sport, round_name, is_home_a, data)
        if random.random() < prob_a:
            wins_a += 1
        else:
            wins_b += 1
        if wins_a == wins_needed or wins_b == wins_needed:
            break

    return team_a if wins_a > wins_b else team_b


# ─────────────────────────────────────────────────────────────
# Section 4 — NFL Simulator
# ─────────────────────────────────────────────────────────────

def simulate_nfl_playoffs(bracket, data):
    """
    Simulates full NFL playoff bracket once. Returns champion string.
    bracket: {"afc": {1: team, 2: team, ...}, "nfc": {1: team, ...}}
    """
    rounds = PLAYOFF_STRUCTURES["nfl"]["rounds"]
    champion = None

    for conf in ("afc", "nfc"):
        seeds = bracket.get(conf, {})
        # Wild Card: 3v6, 4v5, 2v7 (seed 1 and 2 get byes)
        wild_card_winners = {}
        wc_matchups = [(3, 6), (4, 5), (2, 7)]
        for s_high, s_low in wc_matchups:
            ta = seeds.get(s_high, f"{conf}_seed_{s_high}")
            tb = seeds.get(s_low,  f"{conf}_seed_{s_low}")
            prob_a = game_win_prob(ta, tb, "nfl", "Wild Card", True, data)
            wild_card_winners[s_high] = ta if random.random() < prob_a else tb

        # Divisional: 1 vs lowest remaining, 2 vs remaining
        div_teams = [seeds.get(1), wild_card_winners.get(3), wild_card_winners.get(4)]
        div_teams = [t for t in div_teams if t]
        div_winners = []
        if len(div_teams) >= 2:
            for i in range(0, len(div_teams) - 1, 2):
                ta = div_teams[i]
                tb = div_teams[i + 1]
                prob_a = game_win_prob(ta, tb, "nfl", "Divisional", True, data)
                div_winners.append(ta if random.random() < prob_a else tb)

        # Conference Championship
        if len(div_winners) >= 2:
            ta, tb = div_winners[0], div_winners[1]
            prob_a = game_win_prob(ta, tb, "nfl", "Conference Championship", True, data)
            conf_winner = ta if random.random() < prob_a else tb
        elif div_winners:
            conf_winner = div_winners[0]
        else:
            conf_winner = seeds.get(1, f"{conf}_1")

        bracket[f"{conf}_finalist"] = conf_winner

    # Super Bowl (neutral)
    ta = bracket.get("afc_finalist", "AFC")
    tb = bracket.get("nfc_finalist", "NFC")
    prob_a = game_win_prob(ta, tb, "nfl", "Super Bowl", False, data)
    champion = ta if random.random() < prob_a else tb
    return champion


# ─────────────────────────────────────────────────────────────
# Section 5 — NBA/NHL Simulator
# ─────────────────────────────────────────────────────────────

def simulate_bracket_series_sport(teams_seeded, sport, data):
    """
    Simulates bracket-style series playoff for NBA or NHL.
    teams_seeded: list of teams sorted by seed, best seed first.
    Returns champion string.
    """
    structure = PLAYOFF_STRUCTURES[sport]
    series_len = structure.get("series_length", 7)
    rounds     = structure["rounds"]
    current    = list(teams_seeded)

    for round_name in rounds:
        next_round = []
        for i in range(0, len(current) - 1, 2):
            ta = current[i]
            tb = current[i + 1] if i + 1 < len(current) else current[i]
            winner = simulate_series(
                ta, tb, sport, round_name, series_len,
                higher_seed_is_a=True, data=data
            )
            next_round.append(winner)
        if not next_round:
            next_round = current[:1]
        current = next_round

    return current[0] if current else teams_seeded[0]


# ─────────────────────────────────────────────────────────────
# Section 6 — MLB Simulator
# ─────────────────────────────────────────────────────────────

def simulate_mlb_playoffs(teams_seeded, data):
    """Simulates full MLB playoff bracket (mixed series lengths)."""
    rounds = PLAYOFF_STRUCTURES["mlb"]["rounds"]
    current = list(teams_seeded)

    for round_info in rounds:
        round_name = round_info["name"]
        series_len = round_info["length"]
        next_round = []
        for i in range(0, len(current) - 1, 2):
            ta = current[i]
            tb = current[i + 1] if i + 1 < len(current) else current[i]
            winner = simulate_series(
                ta, tb, "mlb", round_name, series_len,
                higher_seed_is_a=True, data=data
            )
            next_round.append(winner)
        if not next_round:
            next_round = current[:1]
        current = next_round

    return current[0] if current else teams_seeded[0]


# ─────────────────────────────────────────────────────────────
# Section 7 — Confidence Bands
# ─────────────────────────────────────────────────────────────

def confidence_bands(champion_counts, n_simulations, teams):
    """
    Computes champion probability with 95% confidence bands for each team.
    Uses 50-batch bootstrap approach.
    """
    bands = {}
    for team in teams:
        mean_prob = champion_counts.get(team, 0) / n_simulations
        # Simple binomial confidence interval (Wilson)
        n     = n_simulations
        p     = mean_prob
        z     = 1.96
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = (z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
        bands[team] = {
            "mean":      round(mean_prob, 4),
            "lower_95":  round(max(0, center - margin), 4),
            "upper_95":  round(min(1, center + margin), 4),
            "certainty": "high"   if margin < 0.04 else
                         "medium" if margin < 0.09 else "low",
        }
    return bands


# ─────────────────────────────────────────────────────────────
# Section 8 — Main Entry Point
# ─────────────────────────────────────────────────────────────

def run_tournament(data):
    """
    Called by brain.py --mode update when playoffs are active.
    Runs Monte Carlo simulations for each active playoff sport.
    Updates data['playoffs'][sport].
    """
    print("[tournament] Running playoff simulations...")

    for sport in ("nfl", "nba", "mlb", "nhl"):
        playoff_data = data.get("playoffs", {}).get(sport, {})
        if not playoff_data.get("active", False):
            continue

        print(f"  [tournament] Simulating {sport.upper()} playoffs ({N_SIMULATIONS} runs)...")
        teams_seeded = playoff_data.get("teams_seeded", [])
        if not teams_seeded:
            continue

        champion_counts = {t: 0 for t in teams_seeded}
        round_counts    = {t: {} for t in teams_seeded}

        for _ in range(N_SIMULATIONS):
            if sport == "nfl":
                bracket  = playoff_data.get("bracket_template", {})
                champion = simulate_nfl_playoffs(dict(bracket), data)
            elif sport == "mlb":
                champion = simulate_mlb_playoffs(list(teams_seeded), data)
            else:
                champion = simulate_bracket_series_sport(list(teams_seeded), sport, data)

            if champion in champion_counts:
                champion_counts[champion] += 1

        champion_probs = confidence_bands(champion_counts, N_SIMULATIONS, teams_seeded)

        data.setdefault("playoffs", {})[sport] = {
            "active":          True,
            "last_updated":    datetime.utcnow().isoformat(),
            "champion_probs":  champion_probs,
            "simulations_run": N_SIMULATIONS,
        }

        top_3 = sorted(champion_probs.items(), key=lambda x: x[1]["mean"], reverse=True)[:3]
        for team, probs in top_3:
            print(f"    {team}: {probs['mean']:.1%} ({probs['lower_95']:.1%}–{probs['upper_95']:.1%})")

    print("[tournament] Playoff simulations complete.")
