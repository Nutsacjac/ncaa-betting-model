"""
Flask web frontend for the NBA Basketball Betting Model.
Serves a dashboard UI with auto-scan and manual analysis.
"""

import time
import numpy as np
from flask import Flask, jsonify, render_template, request

from nba_model import (
    BANKROLL_TIERS,
    BettingAnalyzer,
    ESPNDataProvider,
    NBABettingModel,
    TeamDatabase,
    TheOddsAPIProvider,
    _build_stats_dict,
    _fmt_ml,
    _resolve_team_stats,
    _spread_to_moneylines,
)
import picks_db

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Train model once at startup
print("  Training NBA betting model...")
model = NBABettingModel()
model.train()
print("  Model ready.")

analyzer = BettingAnalyzer(model)

# Initialize picks database
picks_db.init_db()
print("  Picks database ready.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

API_DELAY = 0.3


def _sanitize(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _bankroll_tier(edge):
    """Return (label, stake) for a given edge value."""
    for threshold, label, stake in BANKROLL_TIERS:
        if edge >= threshold:
            return {"label": label, "stake": stake}
    return {"label": "MODERATE VALUE", "stake": "1-2% of bankroll"}


def _build_parlay(analyses):
    """Build a 3-leg parlay from the strongest spread and O/U picks."""
    candidates = []

    for a in analyses:
        # Spread leg
        if a.get("spread_pick") and a.get("spread_edge") is not None:
            spread_impl = np.clip(0.5 + a["spread"] / 50, 0, 1)
            if a["spread_edge"] > 0:
                model_prob = spread_impl + a["spread_edge"]
            else:
                model_prob = (1 - spread_impl) + abs(a["spread_edge"])
            model_prob = float(np.clip(model_prob, 0.51, 0.95))
            candidates.append({
                "type": "Spread",
                "game": f"{a['away_team']} @ {a['home_team']}",
                "pick": a["spread_pick"],
                "edge": abs(a["spread_edge"]),
                "model_prob": model_prob,
            })

        # O/U leg
        if a.get("ou_pick") and a.get("ou_edge") is not None:
            model_prob = float(np.clip(0.5 + abs(a["ou_edge"]) * 0.02, 0.51, 0.95))
            candidates.append({
                "type": "O/U",
                "game": f"{a['away_team']} @ {a['home_team']}",
                "pick": f"{a['ou_pick']} {a['over_under']:.1f}",
                "edge": abs(a["ou_edge"]) / 100,
                "model_prob": model_prob,
            })

    # Sort by edge descending, take top 3
    candidates.sort(key=lambda c: c["edge"], reverse=True)
    legs = candidates[:3]

    if len(legs) < 2:
        return None

    # Each spread/O/U leg is at -110: decimal odds = 1.909
    leg_decimal = 1 + (100 / 110)  # 1.9091
    parlay_decimal = leg_decimal ** len(legs)
    parlay_american = round((parlay_decimal - 1) * 100)

    implied_prob = 1 / parlay_decimal
    combined_model_prob = 1.0
    for leg in legs:
        combined_model_prob *= leg["model_prob"]

    parlay_edge = combined_model_prob - implied_prob

    return {
        "legs": legs,
        "num_legs": len(legs),
        "decimal_odds": round(parlay_decimal, 2),
        "american_odds": f"+{parlay_american}",
        "implied_prob": round(implied_prob * 100, 1),
        "model_prob": round(combined_model_prob * 100, 1),
        "edge": round(parlay_edge * 100, 1),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/teams")
def api_teams():
    """Return known team names for autocomplete."""
    return jsonify(sorted(TeamDatabase.TEAMS.keys()))


@app.route("/api/scan")
def api_scan():
    """Fetch today's games, analyze all markets, return JSON.

    Query params:
      source  -- "espn" (default), "odds-api", or "fallback"
    """
    requested_source = request.args.get("source", "espn")

    games = None
    source = requested_source

    if requested_source == "espn":
        games = ESPNDataProvider.fetch_scoreboard()
        source = "espn"
        if not games:
            games = TeamDatabase.generate_fallback_games()
            source = "fallback"

        # If ESPN returned games but none have odds (in-progress/finished),
        # fall back so the dashboard isn't empty.
        has_odds = any(
            g.get("spread") is not None or g.get("over_under") is not None
            for g in games
        )
        if not has_odds:
            games = TeamDatabase.generate_fallback_games()
            source = "fallback"

    elif requested_source == "odds-api":
        odds_games = TheOddsAPIProvider.fetch_odds()
        if odds_games:
            espn_games = ESPNDataProvider.fetch_scoreboard()
            if espn_games:
                games = TheOddsAPIProvider.merge_odds_into_games(espn_games, odds_games)
            else:
                for og in odds_games:
                    og.setdefault("home_id", None)
                    og.setdefault("away_id", None)
                    og.setdefault("home_rank", 99)
                    og.setdefault("away_rank", 99)
                    og.setdefault("home_record", "")
                    og.setdefault("away_record", "")
                games = odds_games
            source = "odds-api"
        else:
            games = TeamDatabase.generate_fallback_games()
            source = "fallback"

    else:
        games = TeamDatabase.generate_fallback_games()
        source = "fallback"

    analyses = []
    for game in games:
        if game.get("spread") is None and game.get("over_under") is None:
            continue

        home_stats = _resolve_team_stats(
            game["home_team"], game.get("home_id"),
            game.get("home_record", ""), is_home=True,
        )
        away_stats = _resolve_team_stats(
            game["away_team"], game.get("away_id"),
            game.get("away_record", ""), is_home=False,
        )

        if home_stats and away_stats:
            result = analyzer.analyze_game(game, home_stats, away_stats)
            if result.get("best_edge", 0) > 0:
                result["bankroll"] = _bankroll_tier(result["best_edge"])
            analyses.append(result)

        if game.get("home_id"):
            time.sleep(API_DELAY)

    analyses.sort(key=lambda a: a["best_edge"], reverse=True)

    top_plays = [a for a in analyses if a["best_edge"] > 0][:5]

    # Build top parlay from strongest spread + O/U legs
    parlay = _build_parlay(analyses)

    return jsonify(_sanitize({
        "source": source,
        "total_games": len(games),
        "analyzed": len(analyses),
        "top_plays": top_plays,
        "all_games": analyses,
        "parlay": parlay,
    }))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Manual matchup analysis. Accepts JSON body with team names + optional lines."""
    data = request.get_json(force=True)
    home_name = data.get("home_team", "").strip()
    away_name = data.get("away_team", "").strip()

    if not home_name or not away_name:
        return jsonify({"error": "Both home_team and away_team are required."}), 400

    home_id = away_id = None
    home_display = home_name
    away_display = away_name

    espn_home = ESPNDataProvider.search_team(home_name)
    if espn_home:
        home_id, home_display = espn_home

    espn_away = ESPNDataProvider.search_team(away_name)
    if espn_away:
        away_id, away_display = espn_away

    home_stats = _resolve_team_stats(home_display, home_id, "", is_home=True)
    away_stats = _resolve_team_stats(away_display, away_id, "", is_home=False)

    spread = data.get("spread")
    over_under = data.get("over_under")
    home_ml = data.get("home_ml")
    away_ml = data.get("away_ml")

    if spread is not None:
        spread = float(spread)
    if over_under is not None:
        over_under = float(over_under)
    if home_ml is not None:
        home_ml = int(home_ml)
    if away_ml is not None:
        away_ml = int(away_ml)

    if home_ml is None and away_ml is None and spread is not None:
        home_ml, away_ml = _spread_to_moneylines(spread)

    game = {
        "home_team": home_display,
        "away_team": away_display,
        "home_id": home_id,
        "away_id": away_id,
        "spread": spread,
        "over_under": over_under,
        "home_ml": home_ml,
        "away_ml": away_ml,
        "time": "Manual",
        "home_rank": 99,
        "away_rank": 99,
        "home_record": "",
        "away_record": "",
    }

    result = analyzer.analyze_game(game, home_stats, away_stats)
    if result.get("best_edge", 0) > 0:
        result["bankroll"] = _bankroll_tier(result["best_edge"])

    return jsonify(_sanitize(result))


# ---------------------------------------------------------------------------
# Pick Tracking Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/picks/save", methods=["POST"])
def api_save_pick():
    """Save one or more analysis results as tracked picks.
    Rejects saves when source is 'fallback' (simulated data)."""
    source = request.args.get("source", "")
    if source == "fallback":
        return jsonify({"error": "Cannot save fallback/simulated picks to history."}), 400

    data = request.get_json(force=True)
    picks = data if isinstance(data, list) else [data]
    results = []
    for pick in picks:
        result = picks_db.save_pick(pick)
        results.append(result)
    return jsonify(_sanitize(results))


@app.route("/api/picks/history")
def api_picks_history():
    """Paginated pick history with filters."""
    history = picks_db.get_history(
        market=request.args.get("market"),
        status=request.args.get("status"),
        date_from=request.args.get("date_from"),
        date_to=request.args.get("date_to"),
        limit=int(request.args.get("limit", 50)),
        offset=int(request.args.get("offset", 0)),
    )
    return jsonify(_sanitize(history))


@app.route("/api/picks/stats")
def api_picks_stats():
    """Aggregate accuracy stats."""
    stats = picks_db.get_stats(
        market=request.args.get("market"),
        date_from=request.args.get("date_from"),
        date_to=request.args.get("date_to"),
    )
    return jsonify(_sanitize(stats))


@app.route("/api/picks/<int:pick_id>/resolve", methods=["POST"])
def api_resolve_pick(pick_id):
    """Manually resolve a pick with final scores."""
    data = request.get_json(force=True)
    home_score = data.get("home_score")
    away_score = data.get("away_score")
    if home_score is None or away_score is None:
        return jsonify({"error": "home_score and away_score are required."}), 400
    status = picks_db.resolve_pick(pick_id, int(home_score), int(away_score))
    if status is None:
        return jsonify({"error": "Pick not found."}), 404
    return jsonify({"id": pick_id, "status": status})


@app.route("/api/picks/resolve-pending", methods=["POST"])
def api_resolve_pending():
    """Auto-resolve all pending picks by querying ESPN for final scores."""
    final_games = ESPNDataProvider.fetch_final_scores()
    if not final_games:
        return jsonify({"resolved": 0, "message": "No final scores available from ESPN."})

    pending = picks_db.get_history(status="pending", limit=500)
    resolved_count = 0

    def _normalize(name):
        return name.lower().replace(".", "").replace("'", "").strip()

    def _match(a, b):
        na, nb = _normalize(a), _normalize(b)
        return na == nb or na in nb or nb in na

    for pick in pending["picks"]:
        for game in final_games:
            if (_match(pick["home_team"], game["home_team"])
                    and _match(pick["away_team"], game["away_team"])):
                result = picks_db.resolve_pick(
                    pick["id"], game["home_score"], game["away_score"]
                )
                if result and result != "pending":
                    resolved_count += 1
                break

    return jsonify({"resolved": resolved_count})


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5001)
