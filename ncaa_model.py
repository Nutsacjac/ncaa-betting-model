"""
NCAA Men's Basketball Betting Model
Combines auto-scan and interactive analysis across three betting markets:
  - Spread (ATS)
  - Over/Under
  - Moneyline

Fetches live data from ESPN's free public API with hardcoded fallback.
Optionally fetches live betting odds from The Odds API.

Usage:
  python ncaa_model.py                 # Auto-scan today's games (default)
  python ncaa_model.py --manual        # Interactive matchup input
  python ncaa_model.py --top 5         # Show top 5 plays (default: 3)
  python ncaa_model.py --all           # Show all games without prompting
  python ncaa_model.py --no-espn       # Force fallback data (skip ESPN)
  python ncaa_model.py --odds-api      # Use The Odds API for betting lines
  python ncaa_model.py --verbose       # Show model training output
"""

import argparse
import logging
import time
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/scoreboard"
)
ESPN_TEAM_STATS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/teams/{team_id}/statistics"
)

THE_ODDS_API_KEY = "92ec061401cb94a91a3a14203a482dc8"
THE_ODDS_API_URL = (
    "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
)

REQUEST_TIMEOUT = 10  # seconds
API_DELAY = 0.3       # seconds between ESPN calls
ESPN_CACHE_TTL = 300  # 5 minutes for ESPN data
ODDS_CACHE_TTL = 120  # 2 minutes for Odds API data (odds change faster)

EDGE_THRESHOLD_SPREAD = 0.03   # 3% min edge for spread
EDGE_THRESHOLD_ML = 0.03       # 3% min edge for moneyline
OU_POINTS_THRESHOLD = 3        # 3+ points for O/U

BANKROLL_TIERS = [
    (0.10, "STRONG VALUE",   "3-5% of bankroll"),
    (0.07, "SOLID VALUE",    "2-3% of bankroll"),
    (0.03, "MODERATE VALUE", "1-2% of bankroll"),
]

# ---------------------------------------------------------------------------
# ESPN Data Provider
# ---------------------------------------------------------------------------

class ESPNDataProvider:
    """Fetch live games, odds, and team stats from ESPN's free public API."""

    _cache = {}  # {url: (timestamp, data)}

    @staticmethod
    def _get_cached(url):
        """Return cached response data if still valid, else None."""
        if url in ESPNDataProvider._cache:
            ts, data = ESPNDataProvider._cache[url]
            if time.time() - ts < ESPN_CACHE_TTL:
                return data
            del ESPNDataProvider._cache[url]
        return None

    @staticmethod
    def _set_cache(url, data):
        ESPNDataProvider._cache[url] = (time.time(), data)

    @staticmethod
    def _request_with_retry(url, params=None, max_retries=3, base_delay=1.0):
        """HTTP GET with exponential backoff on timeout, connection error, 5xx."""
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 404:
                    logging.warning("ESPN 404 – resource not found: %s", url)
                    return None
                if resp.status_code >= 500:
                    logging.warning("ESPN %d on attempt %d/%d: %s",
                                    resp.status_code, attempt + 1, max_retries, url)
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2 ** attempt))
                        continue
                    return None
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout:
                logging.warning("ESPN timeout on attempt %d/%d: %s",
                                attempt + 1, max_retries, url)
            except requests.exceptions.ConnectionError:
                logging.warning("ESPN connection error on attempt %d/%d: %s",
                                attempt + 1, max_retries, url)
            except requests.exceptions.RequestException as e:
                logging.warning("ESPN request error: %s", e)
                return None
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
        logging.error("ESPN request failed after %d retries: %s", max_retries, url)
        return None

    @staticmethod
    def fetch_scoreboard():
        """Return today's D1 games with odds from the ESPN scoreboard API."""
        cache_key = ESPN_SCOREBOARD_URL + "?groups=50"
        cached = ESPNDataProvider._get_cached(cache_key)
        if cached is not None:
            logging.info("ESPN scoreboard served from cache")
            data = cached
        else:
            data = ESPNDataProvider._request_with_retry(
                ESPN_SCOREBOARD_URL, params={"groups": "50", "limit": "200"},
            )
            if data is None:
                print("  Could not reach ESPN scoreboard.")
                return None
            ESPNDataProvider._set_cache(cache_key, data)

        games = []
        for event in data.get("events", []):
            try:
                game = ESPNDataProvider._parse_event(event)
                if game:
                    games.append(game)
            except Exception:
                continue
        return games if games else None

    @staticmethod
    def _parse_event(event):
        """Parse a single ESPN event into a game dict."""
        competition = event["competitions"][0]
        competitors = competition["competitors"]

        home_comp = away_comp = None
        for comp in competitors:
            if comp["homeAway"] == "home":
                home_comp = comp
            else:
                away_comp = comp
        if not home_comp or not away_comp:
            return None

        home_team = home_comp["team"]["displayName"]
        away_team = away_comp["team"]["displayName"]
        home_id = home_comp["team"]["id"]
        away_id = away_comp["team"]["id"]
        home_rank = int(home_comp.get("curatedRank", {}).get("current", 99))
        away_rank = int(away_comp.get("curatedRank", {}).get("current", 99))

        # Records
        home_record = home_comp.get("records", [{}])[0].get("summary", "0-0") if home_comp.get("records") else "0-0"
        away_record = away_comp.get("records", [{}])[0].get("summary", "0-0") if away_comp.get("records") else "0-0"

        # Parse odds
        spread = over_under = home_ml = away_ml = None
        odds_list = competition.get("odds", [])
        if odds_list:
            odds = odds_list[0]
            # ESPN 'spread' is already from the home team perspective
            raw_spread = odds.get("spread")
            over_under = odds.get("overUnder")
            if raw_spread is not None:
                spread = float(raw_spread)
            if over_under is not None:
                over_under = float(over_under)
            # Moneylines live in odds.moneyline.home/away.close.odds
            ml_block = odds.get("moneyline", {})
            home_ml_str = (ml_block.get("home", {}).get("close", {}).get("odds")
                           or ml_block.get("home", {}).get("open", {}).get("odds"))
            away_ml_str = (ml_block.get("away", {}).get("close", {}).get("odds")
                           or ml_block.get("away", {}).get("open", {}).get("odds"))
            if home_ml_str and home_ml_str not in ("OFF", "N/A"):
                try:
                    home_ml = int(home_ml_str)
                except ValueError:
                    pass
            if away_ml_str and away_ml_str not in ("OFF", "N/A"):
                try:
                    away_ml = int(away_ml_str)
                except ValueError:
                    pass

        # Game time — prefer ESPN's short detail (includes status like "Final")
        status_detail = event.get("status", {}).get("type", {}).get("shortDetail", "")
        if status_detail:
            game_time = status_detail
        else:
            game_date = event.get("date", "")
            try:
                dt = datetime.fromisoformat(game_date.replace("Z", "+00:00"))
                game_time = dt.strftime("%-I:%M %p") + " ET"
            except Exception:
                game_time = "TBD"

        # Scores (available when game is Final)
        home_score = away_score = None
        status_name = event.get("status", {}).get("type", {}).get("name", "")
        if "Final" in game_time or status_name == "STATUS_FINAL":
            try:
                home_score = int(home_comp.get("score", {}) if isinstance(home_comp.get("score"), dict) else home_comp.get("score", 0))
            except (ValueError, TypeError):
                home_score = None
            try:
                away_score = int(away_comp.get("score", {}) if isinstance(away_comp.get("score"), dict) else away_comp.get("score", 0))
            except (ValueError, TypeError):
                away_score = None

        # Game status for filtering (pre-game vs live vs final)
        game_status = event.get("status", {}).get("type", {}).get("name", "STATUS_SCHEDULED")

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_id": home_id,
            "away_id": away_id,
            "home_rank": home_rank,
            "away_rank": away_rank,
            "home_record": home_record,
            "away_record": away_record,
            "spread": spread,
            "over_under": over_under,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "time": game_time,
            "home_score": home_score,
            "away_score": away_score,
            "game_status": game_status,
        }

    @staticmethod
    def fetch_team_stats(team_id):
        """Fetch per-game stats for a team. Returns dict or None."""
        url = ESPN_TEAM_STATS_URL.format(team_id=team_id)

        cached = ESPNDataProvider._get_cached(url)
        if cached is not None:
            data = cached
        else:
            data = ESPNDataProvider._request_with_retry(url)
            if data is None:
                return None
            ESPNDataProvider._set_cache(url, data)

        stats = {}
        try:
            # ESPN structure: results.stats.categories[].stats[]
            stats_obj = data.get("results", {}).get("stats", {})
            categories = stats_obj.get("categories", [])
            if not categories:
                # fallback: older structure
                categories = (data.get("statistics", {})
                              .get("splits", {}).get("categories", []))
            for cat in categories:
                for stat in cat.get("stats", []):
                    name = stat.get("name", "")
                    val = stat.get("value", 0)
                    if val is None:
                        val = 0
                    stats[name] = float(val)
        except Exception:
            return None

        if not stats:
            return None

        return {
            "ppg": stats.get("avgPoints", 75),
            "fg_pct": stats.get("fieldGoalPct", 45.0),
            "3p_pct": stats.get("threePointFieldGoalPct", 35.0),
            "ft_pct": stats.get("freeThrowPct", 72.0),
            "reb": stats.get("avgRebounds", 37.0),
            "ast": stats.get("avgAssists", 14.0),
            "to": stats.get("avgTurnovers", 13.0),
            "stl": stats.get("avgSteals", 7.0),
            "blk": stats.get("avgBlocks", 4.0),
        }

    @staticmethod
    def search_team(name):
        """Search ESPN for a team by name. Returns (team_id, display_name) or None."""
        url = (
            "https://site.api.espn.com/apis/site/v2/sports/basketball"
            "/mens-college-basketball/teams"
        )
        cache_key = url + "?limit=400"

        cached = ESPNDataProvider._get_cached(cache_key)
        if cached is not None:
            data = cached
        else:
            data = ESPNDataProvider._request_with_retry(url, params={"limit": 400})
            if data is None:
                return None
            ESPNDataProvider._set_cache(cache_key, data)

        try:
            name_lower = name.lower()
            for team in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
                t = team.get("team", {})
                candidates = [
                    t.get("displayName", "").lower(),
                    t.get("shortDisplayName", "").lower(),
                    t.get("name", "").lower(),
                    t.get("abbreviation", "").lower(),
                    t.get("nickname", "").lower(),
                ]
                if name_lower in candidates or any(name_lower in c for c in candidates):
                    return t["id"], t["displayName"]
        except Exception:
            pass
        return None

    @staticmethod
    def fetch_final_scores():
        """Return only completed games with final scores from today's scoreboard."""
        cache_key = ESPN_SCOREBOARD_URL + "?groups=50"
        cached = ESPNDataProvider._get_cached(cache_key)
        if cached is not None:
            data = cached
        else:
            data = ESPNDataProvider._request_with_retry(
                ESPN_SCOREBOARD_URL, params={"groups": "50", "limit": "200"},
            )
            if data is None:
                return []
            ESPNDataProvider._set_cache(cache_key, data)

        results = []
        for event in data.get("events", []):
            try:
                game = ESPNDataProvider._parse_event(event)
                if game and game.get("home_score") is not None and game.get("away_score") is not None:
                    results.append(game)
            except Exception:
                continue
        return results


# ---------------------------------------------------------------------------
# The Odds API Provider
# ---------------------------------------------------------------------------

class TheOddsAPIProvider:
    """Fetch live betting odds from The Odds API (the-odds-api.com)."""

    _cache = {}  # {url_key: (timestamp, data)}

    @staticmethod
    def fetch_odds():
        """Return a list of games with odds from The Odds API."""
        cache_key = THE_ODDS_API_URL

        # Check cache first
        if cache_key in TheOddsAPIProvider._cache:
            ts, cached_events = TheOddsAPIProvider._cache[cache_key]
            if time.time() - ts < ODDS_CACHE_TTL:
                logging.info("Odds API served from cache")
                games = []
                for event in cached_events:
                    try:
                        game = TheOddsAPIProvider._parse_event(event)
                        if game:
                            games.append(game)
                    except Exception:
                        continue
                return games if games else None

        try:
            params = {
                "apiKey": THE_ODDS_API_KEY,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
            }
            resp = requests.get(THE_ODDS_API_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            events = resp.json()
        except Exception as e:
            print(f"  Could not reach The Odds API: {e}")
            return None

        if not events:
            return None

        # Cache raw events
        TheOddsAPIProvider._cache[cache_key] = (time.time(), events)

        # Show remaining API usage from response headers
        remaining = resp.headers.get("x-requests-remaining")
        used = resp.headers.get("x-requests-used")
        if remaining is not None:
            print(f"  [Odds API] Requests remaining: {remaining}  (used: {used})")

        games = []
        for event in events:
            try:
                game = TheOddsAPIProvider._parse_event(event)
                if game:
                    games.append(game)
            except Exception:
                continue

        return games if games else None

    @staticmethod
    def _parse_event(event):
        """Parse a single Odds API event, selecting best lines across all bookmakers."""
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        commence = event.get("commence_time", "")

        try:
            dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            game_time = dt.strftime("%-I:%M %p") + " ET"
        except Exception:
            game_time = "TBD"

        # Collect all lines across bookmakers for best-line selection
        all_home_ml = []
        all_away_ml = []
        all_home_spreads = []
        all_totals = []

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                key = market.get("key")
                outcomes = market.get("outcomes", [])

                if key == "h2h":
                    for o in outcomes:
                        if o.get("name") == home_team:
                            all_home_ml.append(int(o["price"]))
                        elif o.get("name") == away_team:
                            all_away_ml.append(int(o["price"]))

                elif key == "spreads":
                    for o in outcomes:
                        if o.get("name") == home_team:
                            all_home_spreads.append(float(o.get("point", 0)))

                elif key == "totals":
                    for o in outcomes:
                        if o.get("name") == "Over":
                            all_totals.append(float(o.get("point", 0)))

        # Best moneyline: highest price for each side (most favorable to bettor)
        home_ml = max(all_home_ml) if all_home_ml else None
        away_ml = max(all_away_ml) if all_away_ml else None

        # Best spread: most points for the home team (highest value = most favorable)
        spread = max(all_home_spreads) if all_home_spreads else None

        # Consensus total: most common line across bookmakers
        if all_totals:
            total_counts = Counter(all_totals)
            over_under = total_counts.most_common(1)[0][0]
        else:
            over_under = None

        return {
            "home_team": home_team,
            "away_team": away_team,
            "spread": spread,
            "over_under": over_under,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "time": game_time,
        }

    @staticmethod
    def merge_odds_into_games(games, odds_games):
        """Overlay odds from The Odds API onto ESPN game dicts.

        Matches games by normalized team names. If a game from the odds
        feed has no match in the ESPN list, it is appended as a new entry.
        """
        if not odds_games:
            return games

        def _normalize(name):
            return name.lower().replace(".", "").replace("'", "").strip()

        def _team_match(espn_name, odds_name):
            en = _normalize(espn_name)
            on = _normalize(odds_name)
            return en == on or en in on or on in en

        updated = 0
        matched_odds = set()
        for game in games:
            for idx, og in enumerate(odds_games):
                if idx in matched_odds:
                    continue
                if (_team_match(game["home_team"], og["home_team"])
                        and _team_match(game["away_team"], og["away_team"])):
                    # Overlay odds (prefer Odds API data when available)
                    if og.get("spread") is not None:
                        game["spread"] = og["spread"]
                    if og.get("over_under") is not None:
                        game["over_under"] = og["over_under"]
                    if og.get("home_ml") is not None:
                        game["home_ml"] = og["home_ml"]
                    if og.get("away_ml") is not None:
                        game["away_ml"] = og["away_ml"]
                    matched_odds.add(idx)
                    updated += 1
                    break

        print(f"  [Odds API] Updated odds for {updated}/{len(games)} games.")
        return games


# ---------------------------------------------------------------------------
# Team Database (hardcoded fallback)
# ---------------------------------------------------------------------------

class TeamDatabase:
    """Hardcoded top-25-style team database used when ESPN is unreachable."""

    TEAMS = {
        'Duke': {'ppg': 82.5, 'fg_pct': 47.2, '3p_pct': 36.8, 'ft_pct': 75.4, 'reb': 38.2,
                 'ast': 16.3, 'to': 11.8, 'stl': 7.5, 'blk': 4.8, 'win_pct': 0.78,
                 'home_win': 0.88, 'away_win': 0.68, 'pace': 72, 'off_rtg': 116, 'def_rtg': 98, 'form': 0.8},
        'UConn': {'ppg': 80.3, 'fg_pct': 46.8, '3p_pct': 35.2, 'ft_pct': 73.1, 'reb': 39.5,
                  'ast': 15.8, 'to': 12.3, 'stl': 7.2, 'blk': 5.2, 'win_pct': 0.82,
                  'home_win': 0.90, 'away_win': 0.74, 'pace': 70, 'off_rtg': 118, 'def_rtg': 96, 'form': 0.8},
        'Kansas': {'ppg': 79.8, 'fg_pct': 46.5, '3p_pct': 37.1, 'ft_pct': 74.2, 'reb': 37.9,
                   'ast': 16.1, 'to': 12.1, 'stl': 7.8, 'blk': 4.3, 'win_pct': 0.76,
                   'home_win': 0.85, 'away_win': 0.67, 'pace': 71, 'off_rtg': 115, 'def_rtg': 99, 'form': 0.6},
        'Houston': {'ppg': 76.5, 'fg_pct': 45.9, '3p_pct': 34.5, 'ft_pct': 71.8, 'reb': 40.2,
                    'ast': 14.5, 'to': 11.5, 'stl': 8.5, 'blk': 5.8, 'win_pct': 0.80,
                    'home_win': 0.89, 'away_win': 0.71, 'pace': 68, 'off_rtg': 113, 'def_rtg': 94, 'form': 0.8},
        'Auburn': {'ppg': 81.2, 'fg_pct': 47.8, '3p_pct': 36.2, 'ft_pct': 72.5, 'reb': 38.5,
                   'ast': 15.2, 'to': 12.8, 'stl': 7.9, 'blk': 4.5, 'win_pct': 0.77,
                   'home_win': 0.87, 'away_win': 0.67, 'pace': 73, 'off_rtg': 117, 'def_rtg': 99, 'form': 0.6},
        'Purdue': {'ppg': 78.9, 'fg_pct': 46.2, '3p_pct': 35.8, 'ft_pct': 73.8, 'reb': 39.1,
                   'ast': 14.8, 'to': 11.2, 'stl': 6.8, 'blk': 5.5, 'win_pct': 0.75,
                   'home_win': 0.84, 'away_win': 0.66, 'pace': 69, 'off_rtg': 114, 'def_rtg': 97, 'form': 0.8},
        'North Carolina': {'ppg': 79.8, 'fg_pct': 45.9, '3p_pct': 34.2, 'ft_pct': 71.8, 'reb': 39.5,
                          'ast': 15.1, 'to': 12.5, 'stl': 6.8, 'blk': 3.9, 'win_pct': 0.72,
                          'home_win': 0.80, 'away_win': 0.64, 'pace': 72, 'off_rtg': 112, 'def_rtg': 100, 'form': 0.6},
        'Tennessee': {'ppg': 77.2, 'fg_pct': 45.5, '3p_pct': 33.8, 'ft_pct': 72.9, 'reb': 38.8,
                      'ast': 14.2, 'to': 11.8, 'stl': 7.5, 'blk': 4.8, 'win_pct': 0.74,
                      'home_win': 0.83, 'away_win': 0.65, 'pace': 68, 'off_rtg': 111, 'def_rtg': 96, 'form': 0.8},
        'Arizona': {'ppg': 80.5, 'fg_pct': 46.8, '3p_pct': 36.5, 'ft_pct': 73.2, 'reb': 37.5,
                    'ast': 15.8, 'to': 12.2, 'stl': 7.2, 'blk': 4.2, 'win_pct': 0.73,
                    'home_win': 0.82, 'away_win': 0.64, 'pace': 71, 'off_rtg': 114, 'def_rtg': 99, 'form': 0.6},
        'Kentucky': {'ppg': 81.8, 'fg_pct': 47.5, '3p_pct': 35.9, 'ft_pct': 74.5, 'reb': 38.9,
                     'ast': 16.5, 'to': 12.9, 'stl': 7.1, 'blk': 4.5, 'win_pct': 0.71,
                     'home_win': 0.79, 'away_win': 0.63, 'pace': 72, 'off_rtg': 115, 'def_rtg': 101, 'form': 0.4},
        'Marquette': {'ppg': 78.5, 'fg_pct': 45.8, '3p_pct': 35.5, 'ft_pct': 72.8, 'reb': 37.2,
                      'ast': 15.2, 'to': 11.9, 'stl': 7.0, 'blk': 3.8, 'win_pct': 0.70,
                      'home_win': 0.78, 'away_win': 0.62, 'pace': 70, 'off_rtg': 112, 'def_rtg': 100, 'form': 0.6},
        'Florida': {'ppg': 76.8, 'fg_pct': 44.9, '3p_pct': 34.8, 'ft_pct': 71.5, 'reb': 38.5,
                    'ast': 14.5, 'to': 12.5, 'stl': 6.9, 'blk': 4.5, 'win_pct': 0.69,
                    'home_win': 0.77, 'away_win': 0.61, 'pace': 69, 'off_rtg': 110, 'def_rtg': 98, 'form': 0.6},
        'Michigan State': {'ppg': 77.5, 'fg_pct': 45.2, '3p_pct': 34.2, 'ft_pct': 72.1, 'reb': 39.8,
                          'ast': 14.8, 'to': 11.8, 'stl': 7.2, 'blk': 4.9, 'win_pct': 0.68,
                          'home_win': 0.76, 'away_win': 0.60, 'pace': 68, 'off_rtg': 109, 'def_rtg': 97, 'form': 0.8},
        'Villanova': {'ppg': 75.2, 'fg_pct': 44.5, '3p_pct': 36.8, 'ft_pct': 73.5, 'reb': 36.5,
                      'ast': 14.2, 'to': 11.5, 'stl': 6.5, 'blk': 3.5, 'win_pct': 0.67,
                      'home_win': 0.75, 'away_win': 0.59, 'pace': 69, 'off_rtg': 108, 'def_rtg': 99, 'form': 0.4},
        'Alabama': {'ppg': 82.8, 'fg_pct': 47.2, '3p_pct': 37.5, 'ft_pct': 73.8, 'reb': 37.8,
                    'ast': 16.8, 'to': 13.5, 'stl': 7.8, 'blk': 4.2, 'win_pct': 0.72,
                    'home_win': 0.80, 'away_win': 0.64, 'pace': 74, 'off_rtg': 118, 'def_rtg': 102, 'form': 0.6},
        'Gonzaga': {'ppg': 83.5, 'fg_pct': 48.2, '3p_pct': 37.8, 'ft_pct': 74.9, 'reb': 39.2,
                    'ast': 17.2, 'to': 12.8, 'stl': 6.9, 'blk': 4.8, 'win_pct': 0.75,
                    'home_win': 0.84, 'away_win': 0.66, 'pace': 73, 'off_rtg': 119, 'def_rtg': 100, 'form': 0.8},
        'Texas': {'ppg': 78.2, 'fg_pct': 45.8, '3p_pct': 35.2, 'ft_pct': 72.5, 'reb': 38.2,
                  'ast': 15.5, 'to': 12.2, 'stl': 7.5, 'blk': 4.5, 'win_pct': 0.70,
                  'home_win': 0.78, 'away_win': 0.62, 'pace': 70, 'off_rtg': 113, 'def_rtg': 99, 'form': 0.6},
        'Creighton': {'ppg': 79.5, 'fg_pct': 46.5, '3p_pct': 37.2, 'ft_pct': 73.8, 'reb': 36.8,
                      'ast': 15.8, 'to': 11.5, 'stl': 6.8, 'blk': 3.8, 'win_pct': 0.69,
                      'home_win': 0.77, 'away_win': 0.61, 'pace': 71, 'off_rtg': 114, 'def_rtg': 100, 'form': 0.6},
        'UCLA': {'ppg': 76.8, 'fg_pct': 45.2, '3p_pct': 34.5, 'ft_pct': 71.9, 'reb': 38.5,
                 'ast': 14.8, 'to': 12.0, 'stl': 7.2, 'blk': 4.8, 'win_pct': 0.68,
                 'home_win': 0.76, 'away_win': 0.60, 'pace': 69, 'off_rtg': 111, 'def_rtg': 98, 'form': 0.4},
        'Baylor': {'ppg': 77.9, 'fg_pct': 45.9, '3p_pct': 35.8, 'ft_pct': 72.8, 'reb': 38.9,
                   'ast': 14.9, 'to': 11.8, 'stl': 7.5, 'blk': 5.2, 'win_pct': 0.71,
                   'home_win': 0.79, 'away_win': 0.63, 'pace': 70, 'off_rtg': 112, 'def_rtg': 97, 'form': 0.8},
    }

    @classmethod
    def lookup(cls, name):
        """Case-insensitive team lookup. Returns stats dict or None."""
        for team, stats in cls.TEAMS.items():
            if name.lower() == team.lower():
                return dict(stats)
        # partial match
        for team, stats in cls.TEAMS.items():
            if name.lower() in team.lower() or team.lower() in name.lower():
                return dict(stats)
        return None

    @classmethod
    def generate_fallback_games(cls, num_games=10):
        """Generate matchups from the hardcoded database (deterministic per day)."""
        teams = list(cls.TEAMS.keys())
        np.random.seed(int(datetime.now().strftime("%Y%m%d")))

        games = []
        used = set()
        for _ in range(num_games):
            avail = [t for t in teams if t not in used]
            if len(avail) < 2:
                break
            t1 = np.random.choice(avail)
            used.add(t1)
            avail.remove(t1)
            t2 = np.random.choice(avail)
            used.add(t2)

            if np.random.random() < 0.6:
                home, away = t1, t2
            else:
                home, away = t2, t1

            hs = cls.TEAMS[home]
            aws = cls.TEAMS[away]

            strength_diff = (hs['win_pct'] - aws['win_pct']) * 30
            spread = -(strength_diff + 3.5 + np.random.normal(0, 1.5))
            spread = round(spread * 2) / 2

            ou = round(hs['ppg'] + aws['ppg'] + np.random.normal(0, 3))

            hour = int(np.random.choice([12, 14, 16, 18, 19, 20, 21]))
            minute = int(np.random.choice([0, 30]))
            ampm = "PM" if hour >= 12 else "AM"
            disp_hour = hour if hour <= 12 else hour - 12
            game_time = f"{disp_hour}:{minute:02d} {ampm}"

            # Synthetic moneylines from spread
            home_ml, away_ml = _spread_to_moneylines(spread)

            games.append({
                "home_team": home,
                "away_team": away,
                "home_id": None,
                "away_id": None,
                "home_rank": 99,
                "away_rank": 99,
                "home_record": f"{int(hs['win_pct']*30)}-{int((1-hs['win_pct'])*30)}",
                "away_record": f"{int(aws['win_pct']*30)}-{int((1-aws['win_pct'])*30)}",
                "spread": spread,
                "over_under": float(ou),
                "home_ml": home_ml,
                "away_ml": away_ml,
                "time": game_time,
            })
        return games


def _spread_to_moneylines(spread):
    """Convert a point spread to approximate American moneylines."""
    # rough conversion: each point ~= 25 ML points near pick'em
    raw = spread * 25
    if raw < 0:
        home_ml = int(raw) if raw <= -110 else -110
        away_ml = int(-raw) if -raw >= 100 else 100
    elif raw > 0:
        home_ml = int(raw) if raw >= 100 else 100
        away_ml = int(-raw) if raw >= 110 else -110
    else:
        home_ml = -110
        away_ml = -110
    return home_ml, away_ml


# ---------------------------------------------------------------------------
# NCAA Betting Model (MLP Neural Network)
# ---------------------------------------------------------------------------

class NCAABettingModel:
    """MLPClassifier (128-64-32-16) trained on synthetic data."""

    FEATURE_NAMES = [
        'team_ppg', 'opp_ppg', 'team_fg_pct', 'opp_fg_pct',
        'team_3p_pct', 'opp_3p_pct', 'team_ft_pct', 'opp_ft_pct',
        'team_reb_pg', 'opp_reb_pg', 'team_ast_pg', 'opp_ast_pg',
        'team_to_pg', 'opp_to_pg', 'team_stl_pg', 'opp_stl_pg',
        'team_blk_pg', 'opp_blk_pg', 'team_win_pct', 'opp_win_pct',
        'team_home_win_pct', 'opp_away_win_pct', 'is_home',
        'team_pace', 'opp_pace', 'team_off_rating', 'opp_off_rating',
        'team_def_rating', 'opp_def_rating', 'team_recent_form', 'opp_recent_form',
    ]

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    # -- synthetic data generation ------------------------------------------

    @staticmethod
    def _generate_synthetic_data(n_samples=5000):
        np.random.seed(42)
        data = []
        for _ in range(n_samples):
            team_ppg = np.random.normal(75, 8)
            opp_ppg = np.random.normal(75, 8)
            team_fg_pct = np.random.normal(0.45, 0.05)
            opp_fg_pct = np.random.normal(0.45, 0.05)
            team_3p_pct = np.random.normal(0.35, 0.06)
            opp_3p_pct = np.random.normal(0.35, 0.06)
            team_ft_pct = np.random.normal(0.72, 0.06)
            opp_ft_pct = np.random.normal(0.72, 0.06)
            team_reb_pg = np.random.normal(37, 5)
            opp_reb_pg = np.random.normal(37, 5)
            team_ast_pg = np.random.normal(14, 3)
            opp_ast_pg = np.random.normal(14, 3)
            team_to_pg = np.random.normal(13, 3)
            opp_to_pg = np.random.normal(13, 3)
            team_stl_pg = np.random.normal(7, 2)
            opp_stl_pg = np.random.normal(7, 2)
            team_blk_pg = np.random.normal(4, 1.5)
            opp_blk_pg = np.random.normal(4, 1.5)
            team_win_pct = np.random.uniform(0.3, 0.9)
            opp_win_pct = np.random.uniform(0.3, 0.9)
            team_home_win_pct = team_win_pct + np.random.uniform(0, 0.15)
            opp_away_win_pct = opp_win_pct - np.random.uniform(0, 0.10)
            is_home = np.random.choice([0, 1])
            team_pace = np.random.normal(70, 5)
            opp_pace = np.random.normal(70, 5)
            team_off_rating = np.random.normal(110, 10)
            opp_off_rating = np.random.normal(110, 10)
            team_def_rating = np.random.normal(100, 10)
            opp_def_rating = np.random.normal(100, 10)
            team_recent_form = np.random.uniform(0, 1)
            opp_recent_form = np.random.uniform(0, 1)

            score_diff = (team_ppg - opp_ppg) * 1.5
            efficiency_diff = ((team_off_rating - team_def_rating)
                               - (opp_off_rating - opp_def_rating))
            shooting_diff = (team_fg_pct - opp_fg_pct) * 100
            win_pct_diff = (team_win_pct - opp_win_pct) * 50
            home_advantage = 3 if is_home == 1 else 0
            form_diff = (team_recent_form - opp_recent_form) * 10

            predicted_margin = (score_diff + efficiency_diff * 0.3
                                + shooting_diff * 0.5 + win_pct_diff
                                + home_advantage + form_diff)
            win_prob = 1 / (1 + np.exp(-predicted_margin / 10))
            win_prob = np.clip(win_prob + np.random.normal(0, 0.1), 0, 1)

            features = [
                team_ppg, opp_ppg, team_fg_pct, opp_fg_pct,
                team_3p_pct, opp_3p_pct, team_ft_pct, opp_ft_pct,
                team_reb_pg, opp_reb_pg, team_ast_pg, opp_ast_pg,
                team_to_pg, opp_to_pg, team_stl_pg, opp_stl_pg,
                team_blk_pg, opp_blk_pg, team_win_pct, opp_win_pct,
                team_home_win_pct, opp_away_win_pct, is_home,
                team_pace, opp_pace, team_off_rating, opp_off_rating,
                team_def_rating, opp_def_rating, team_recent_form, opp_recent_form,
            ]
            data.append(features + [win_prob])

        columns = NCAABettingModel.FEATURE_NAMES + ['win_probability']
        return pd.DataFrame(data, columns=columns)

    # -- training -----------------------------------------------------------

    def train(self, verbose=0):
        data = self._generate_synthetic_data(n_samples=5000)
        X = data[self.FEATURE_NAMES].values
        y = (data['win_probability'].values > 0.5).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32, 16),
            activation='relu', solver='adam', alpha=0.001,
            batch_size=32, learning_rate_init=0.001, max_iter=200,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, random_state=42, verbose=verbose,
        )
        self.model.fit(X_train_scaled, y_train)

        if verbose:
            X_test_scaled = self.scaler.transform(X_test)
            from sklearn.metrics import accuracy_score, roc_auc_score
            y_pred = self.model.predict(X_test_scaled)
            y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            print(f"  Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print(f"  Test AUC:      {roc_auc_score(y_test, y_proba):.4f}")

    # -- prediction ---------------------------------------------------------

    def predict(self, feature_vector):
        """Return P(home win) given a 31-element feature vector."""
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[0][1]


# ---------------------------------------------------------------------------
# Betting Analyzer
# ---------------------------------------------------------------------------

class BettingAnalyzer:
    """Calculate edges across spread, O/U, and moneyline markets."""

    def __init__(self, model: NCAABettingModel):
        self.model = model

    def build_feature_vector(self, home_stats, away_stats, is_home=1):
        """Build the 31-element feature vector from team stat dicts."""
        def _pct(val):
            """Normalise percentage: if > 1 assume it's 0-100 scale."""
            return val / 100 if val > 1 else val

        return [
            home_stats['ppg'],
            away_stats['ppg'],
            _pct(home_stats.get('fg_pct', 45)),
            _pct(away_stats.get('fg_pct', 45)),
            _pct(home_stats.get('3p_pct', 35)),
            _pct(away_stats.get('3p_pct', 35)),
            _pct(home_stats.get('ft_pct', 72)),
            _pct(away_stats.get('ft_pct', 72)),
            home_stats.get('reb', 37),
            away_stats.get('reb', 37),
            home_stats.get('ast', 14),
            away_stats.get('ast', 14),
            home_stats.get('to', 13),
            away_stats.get('to', 13),
            home_stats.get('stl', 7),
            away_stats.get('stl', 7),
            home_stats.get('blk', 4),
            away_stats.get('blk', 4),
            home_stats.get('win_pct', 0.5),
            away_stats.get('win_pct', 0.5),
            home_stats.get('home_win', home_stats.get('win_pct', 0.5) + 0.05),
            away_stats.get('away_win', away_stats.get('win_pct', 0.5) - 0.05),
            is_home,
            home_stats.get('pace', 70),
            away_stats.get('pace', 70),
            home_stats.get('off_rtg', 110),
            away_stats.get('off_rtg', 110),
            home_stats.get('def_rtg', 100),
            away_stats.get('def_rtg', 100),
            home_stats.get('form', 0.5),
            away_stats.get('form', 0.5),
        ]

    def analyze_game(self, game, home_stats, away_stats):
        """Full three-market analysis for one game. Returns analysis dict."""
        features = self.build_feature_vector(home_stats, away_stats)
        home_win_prob = self.model.predict(features)
        away_win_prob = 1 - home_win_prob
        predicted_margin = (home_win_prob - 0.5) * 20

        result = {
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "time": game.get("time", "TBD"),
            "home_rank": game.get("home_rank", 99),
            "away_rank": game.get("away_rank", 99),
            "home_record": game.get("home_record", ""),
            "away_record": game.get("away_record", ""),
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
            "predicted_margin": predicted_margin,
            "confidence": abs(home_win_prob - 0.5) * 200,
            # Market results filled below
            "spread": None, "spread_edge": None, "spread_pick": None,
            "over_under": None, "predicted_total": None, "ou_edge": None, "ou_pick": None,
            "home_ml": None, "away_ml": None, "ml_edge": None, "ml_pick": None,
            "best_edge": 0, "best_market": None, "best_pick": None,
        }

        edges = []

        # --- Spread ATS ---
        if game.get("spread") is not None:
            spread = float(game["spread"])
            result["spread"] = spread
            spread_impl_prob = np.clip(0.5 + spread / 50, 0, 1)
            spread_edge = home_win_prob - spread_impl_prob
            result["spread_edge"] = spread_edge
            if abs(spread_edge) >= EDGE_THRESHOLD_SPREAD:
                if spread_edge > 0:
                    result["spread_pick"] = f"{game['home_team']} {spread:+.1f}"
                else:
                    result["spread_pick"] = f"{game['away_team']} {-spread:+.1f}"
                edges.append(("Spread", abs(spread_edge), result["spread_pick"]))

        # --- Over / Under ---
        if game.get("over_under") is not None:
            ou_line = float(game["over_under"])
            result["over_under"] = ou_line
            avg_pace = (home_stats.get('pace', 70) + away_stats.get('pace', 70)) / 2
            pace_factor = avg_pace / 70
            pred_total = (home_stats['ppg'] + away_stats['ppg']) * pace_factor
            result["predicted_total"] = pred_total
            ou_diff = pred_total - ou_line
            result["ou_edge"] = ou_diff
            if abs(ou_diff) >= OU_POINTS_THRESHOLD:
                result["ou_pick"] = "OVER" if ou_diff > 0 else "UNDER"
                edges.append(("O/U", abs(ou_diff) / 100, result["ou_pick"]))

        # --- Moneyline ---
        if game.get("home_ml") is not None and game.get("away_ml") is not None:
            home_ml = game["home_ml"]
            away_ml = game["away_ml"]
            result["home_ml"] = home_ml
            result["away_ml"] = away_ml
            home_impl = _ml_to_implied_prob(home_ml)
            away_impl = _ml_to_implied_prob(away_ml)
            home_ml_edge = home_win_prob - home_impl
            away_ml_edge = away_win_prob - away_impl
            if home_ml_edge > away_ml_edge and home_ml_edge >= EDGE_THRESHOLD_ML:
                result["ml_edge"] = home_ml_edge
                result["ml_pick"] = f"{game['home_team']} ML ({_fmt_ml(home_ml)})"
                edges.append(("ML", home_ml_edge, result["ml_pick"]))
            elif away_ml_edge >= EDGE_THRESHOLD_ML:
                result["ml_edge"] = away_ml_edge
                result["ml_pick"] = f"{game['away_team']} ML ({_fmt_ml(away_ml)})"
                edges.append(("ML", away_ml_edge, result["ml_pick"]))

        # Best edge overall
        if edges:
            edges.sort(key=lambda e: e[1], reverse=True)
            result["best_edge"] = edges[0][1]
            result["best_market"] = edges[0][0]
            result["best_pick"] = edges[0][2]

        return result


def _ml_to_implied_prob(ml):
    """Convert American moneyline to implied probability."""
    if ml < 0:
        return -ml / (-ml + 100)
    else:
        return 100 / (ml + 100)


def _fmt_ml(ml):
    """Format moneyline for display."""
    return f"{ml:+d}"


# ---------------------------------------------------------------------------
# Output Formatter
# ---------------------------------------------------------------------------

class OutputFormatter:
    """All CLI display logic."""

    @staticmethod
    def header():
        print(f"\n{'='*80}")
        print(f"  NCAA MEN'S BASKETBALL BETTING MODEL")
        print(f"  {datetime.now().strftime('%A, %B %d, %Y')}")
        print(f"{'='*80}")

    @staticmethod
    def game_list(games):
        print(f"\n{'Game':<45} {'Time':<12} {'Spread':<12} {'O/U':<8} {'ML'}")
        print("-" * 80)
        for g in games:
            matchup = f"{g['away_team']} @ {g['home_team']}"
            sp = f"{g['spread']:+.1f}" if g.get('spread') is not None else "N/A"
            ou = f"{g['over_under']:.1f}" if g.get('over_under') is not None else "N/A"
            ml = ""
            if g.get("home_ml") is not None:
                ml = f"{_fmt_ml(g['home_ml'])}/{_fmt_ml(g['away_ml'])}"
            print(f"{matchup:<45} {g['time']:<12} {sp:<12} {ou:<8} {ml}")
        print()

    @staticmethod
    def top_plays(plays, label="TOP PLAYS"):
        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}\n")
        if not plays:
            print("  No strong betting advantages found for today's games.")
            print("  All lines appear efficient. Consider passing.\n")
            return

        for i, p in enumerate(plays, 1):
            OutputFormatter._print_play(i, p)

    @staticmethod
    def _print_play(rank, p):
        away_label = f"#{p['away_rank']} " if p.get('away_rank', 99) <= 25 else ""
        home_label = f"#{p['home_rank']} " if p.get('home_rank', 99) <= 25 else ""
        matchup = f"{away_label}{p['away_team']} @ {home_label}{p['home_team']}"

        print(f"#{rank}  {matchup}")
        print("-" * 80)
        print(f"  Time: {p['time']}   "
              f"Records: {p.get('away_record','')} / {p.get('home_record','')}")

        # Win probabilities
        print(f"\n  MODEL PREDICTION:")
        print(f"    {p['home_team']:<25} {p['home_win_prob']*100:5.1f}%")
        print(f"    {p['away_team']:<25} {p['away_win_prob']*100:5.1f}%")
        print(f"    Predicted margin: {p['home_team']} {p['predicted_margin']:+.1f}")
        print(f"    Confidence: {p['confidence']:.0f}%")

        # Spread
        if p.get("spread") is not None:
            spread_impl = np.clip(0.5 + p['spread'] / 50, 0, 1)
            print(f"\n  SPREAD (ATS):  Line {p['spread']:+.1f}")
            print(f"    Implied prob: {spread_impl*100:.1f}%  |  Model: {p['home_win_prob']*100:.1f}%"
                  f"  |  Edge: {abs(p['spread_edge'])*100:.1f}%")
            if p.get("spread_pick"):
                print(f"    >> Pick: {p['spread_pick']}")

        # Over/Under
        if p.get("over_under") is not None:
            print(f"\n  OVER/UNDER:  Line {p['over_under']:.1f}")
            print(f"    Predicted total: {p['predicted_total']:.1f}  |  Diff: {p['ou_edge']:+.1f} pts")
            if p.get("ou_pick"):
                print(f"    >> Pick: {p['ou_pick']}")

        # Moneyline
        if p.get("home_ml") is not None:
            print(f"\n  MONEYLINE:  {p['home_team']} {_fmt_ml(p['home_ml'])}  /  "
                  f"{p['away_team']} {_fmt_ml(p['away_ml'])}")
            if p.get("ml_edge") is not None:
                print(f"    Edge: {p['ml_edge']*100:.1f}%")
            if p.get("ml_pick"):
                print(f"    >> Pick: {p['ml_pick']}")

        # Bankroll recommendation
        if p.get("best_edge", 0) > 0:
            tier_label, tier_stake = "MODERATE VALUE", "1-2% of bankroll"
            for threshold, label, stake in BANKROLL_TIERS:
                if p["best_edge"] >= threshold:
                    tier_label, tier_stake = label, stake
                    break
            print(f"\n  RECOMMENDATION:  {tier_label}")
            print(f"    Best market: {p['best_market']}  ->  {p['best_pick']}")
            print(f"    Suggested stake: {tier_stake}")
        print("\n" + "=" * 80 + "\n")

    @staticmethod
    def disclaimer():
        print("=" * 80)
        print("  DISCLAIMER: For educational and entertainment purposes only.")
        print("  Always bet responsibly. Never wager more than you can afford to lose.")
        print("  Past performance does not guarantee future results.")
        print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Stat helpers
# ---------------------------------------------------------------------------

def _record_to_win_pct(record_str):
    """Parse '20-5' into 0.80 win percentage."""
    try:
        parts = record_str.split("-")
        wins, losses = int(parts[0]), int(parts[1])
        total = wins + losses
        return wins / total if total > 0 else 0.5
    except Exception:
        return 0.5


def _estimate_advanced(stats, win_pct):
    """Estimate pace / off_rtg / def_rtg / form from basic stats."""
    ppg = stats.get('ppg', 75)
    fg = stats.get('fg_pct', 45)
    # pace: fast teams score more
    pace = 65 + (ppg - 65) * 0.3
    off_rtg = 90 + fg * 0.5 + (ppg - 65) * 0.2
    def_rtg = 110 - win_pct * 20
    form = min(1.0, win_pct + 0.1)
    return {
        'pace': round(pace, 1),
        'off_rtg': round(off_rtg, 1),
        'def_rtg': round(def_rtg, 1),
        'form': round(form, 2),
    }


def _build_stats_dict(espn_stats, win_pct, is_home):
    """Merge ESPN basic stats with estimated advanced stats."""
    if espn_stats is None:
        return None
    out = dict(espn_stats)
    out['win_pct'] = win_pct
    advanced = _estimate_advanced(out, win_pct)
    out.update(advanced)
    out['home_win'] = min(1.0, win_pct + 0.08) if is_home else win_pct
    out['away_win'] = max(0.0, win_pct - 0.05) if not is_home else win_pct
    return out


# ---------------------------------------------------------------------------
# Auto-scan mode
# ---------------------------------------------------------------------------

def run_auto_scan(args):
    """Fetch today's games from ESPN (or fallback), analyze, display top plays."""

    OutputFormatter.header()

    # 1. Train model
    print("\n  [1/3] Training neural network model...")
    model = NCAABettingModel()
    model.train(verbose=1 if args.verbose else 0)
    print("  Model trained.\n")

    # 2. Fetch games
    print("  [2/3] Fetching today's schedule...")
    games = None
    if not args.no_espn:
        games = ESPNDataProvider.fetch_scoreboard()
        if games:
            print(f"  Loaded {len(games)} games from ESPN.\n")
        else:
            print("  ESPN unavailable, using fallback database.\n")

    if games is None:
        games = TeamDatabase.generate_fallback_games()
        print(f"  Generated {len(games)} fallback matchups.\n")

    # 2b. Optionally overlay odds from The Odds API
    if args.odds_api:
        print("  Fetching live odds from The Odds API...")
        odds_games = TheOddsAPIProvider.fetch_odds()
        if odds_games:
            print(f"  Loaded odds for {len(odds_games)} games from The Odds API.")
            games = TheOddsAPIProvider.merge_odds_into_games(games, odds_games)
        else:
            print("  No odds returned from The Odds API.\n")

    OutputFormatter.game_list(games)

    # 3. Analyze
    print("  [3/3] Analyzing games across Spread / O-U / ML markets...\n")
    analyzer = BettingAnalyzer(model)
    analyses = []

    for game in games:
        # Skip games without odds
        if game.get("spread") is None and game.get("over_under") is None:
            continue

        home_stats = _resolve_team_stats(
            game["home_team"], game.get("home_id"), game.get("home_record", ""), is_home=True,
            use_espn=(not args.no_espn),
        )
        away_stats = _resolve_team_stats(
            game["away_team"], game.get("away_id"), game.get("away_record", ""), is_home=False,
            use_espn=(not args.no_espn),
        )

        if home_stats and away_stats:
            result = analyzer.analyze_game(game, home_stats, away_stats)
            analyses.append(result)
        if not args.no_espn and game.get("home_id"):
            time.sleep(API_DELAY)

    # Sort by best edge
    analyses.sort(key=lambda a: a["best_edge"], reverse=True)

    # Display
    top_n = args.top
    top = [a for a in analyses if a["best_edge"] > 0][:top_n]
    OutputFormatter.top_plays(top, label=f"TOP {len(top)} PLAYS")
    OutputFormatter.disclaimer()

    # Show all?
    if args.all:
        _print_all_analyses(analyses)
    elif analyses:
        try:
            resp = input("  Show all game analyses? (y/n): ").strip().lower()
            if resp in ('y', 'yes'):
                _print_all_analyses(analyses)
        except (EOFError, KeyboardInterrupt):
            print()


def _print_all_analyses(analyses):
    print(f"\n{'='*80}")
    print("  COMPLETE GAME ANALYSIS")
    print(f"{'='*80}\n")
    print(f"  {'Game':<42} {'Best Market':<10} {'Edge':>8}  {'Pick'}")
    print("  " + "-" * 76)
    for a in analyses:
        matchup = f"{a['away_team']} @ {a['home_team']}"
        mkt = a.get("best_market", "-") or "-"
        edge_val = a["best_edge"] * 100 if a["best_market"] != "O/U" else (abs(a.get("ou_edge", 0)))
        pick = a.get("best_pick", "PASS") or "PASS"
        print(f"  {matchup:<42} {mkt:<10} {edge_val:>7.1f}%  {pick}")
    print()


def _resolve_team_stats(team_name, team_id, record, is_home, use_espn=True):
    """Try ESPN stats, then fallback database."""
    win_pct = _record_to_win_pct(record) if record else 0.5

    # Try ESPN
    if use_espn and team_id:
        espn_stats = ESPNDataProvider.fetch_team_stats(team_id)
        if espn_stats:
            return _build_stats_dict(espn_stats, win_pct, is_home)

    # Fallback to hardcoded DB
    fb = TeamDatabase.lookup(team_name)
    if fb:
        return fb

    # Last resort: generic stats
    return {
        'ppg': 72, 'fg_pct': 44, '3p_pct': 33, 'ft_pct': 70,
        'reb': 35, 'ast': 13, 'to': 13, 'stl': 6, 'blk': 3,
        'win_pct': win_pct,
        'home_win': min(1.0, win_pct + 0.08),
        'away_win': max(0.0, win_pct - 0.05),
        'pace': 68, 'off_rtg': 105, 'def_rtg': 102, 'form': 0.5,
    }


# ---------------------------------------------------------------------------
# Manual / interactive mode
# ---------------------------------------------------------------------------

def run_manual_mode(args):
    """Interactive matchup entry with ESPN auto-fill."""

    OutputFormatter.header()

    print("\n  Training neural network model...")
    model = NCAABettingModel()
    model.train(verbose=1 if args.verbose else 0)
    print("  Model trained.\n")

    analyzer = BettingAnalyzer(model)

    while True:
        print("=" * 80)
        print("  ENTER MATCHUP")
        print("=" * 80)

        home_name = input("\n  Home team: ").strip()
        if not home_name:
            break
        away_name = input("  Away team: ").strip()
        if not away_name:
            break

        home_stats = _interactive_stats(home_name, is_home=True, use_espn=(not args.no_espn))
        away_stats = _interactive_stats(away_name, is_home=False, use_espn=(not args.no_espn))

        # Betting lines — try The Odds API first if enabled
        spread = ou = home_ml = away_ml = None
        if args.odds_api:
            odds_match = _lookup_odds_api(home_name, away_name)
            if odds_match:
                spread = odds_match.get("spread")
                ou = odds_match.get("over_under")
                home_ml = odds_match.get("home_ml")
                away_ml = odds_match.get("away_ml")
                print(f"\n  --- Odds from The Odds API ---")
                print(f"    Spread: {spread}  |  O/U: {ou}  |  ML: {home_ml}/{away_ml}")
            else:
                print("\n  Could not find odds for this matchup on The Odds API.")

        if spread is None and ou is None:
            print("\n  --- Betting Lines (press Enter to skip) ---")
            spread = _input_float(f"  Spread for {home_name} (e.g. -5.5): ")
            ou = _input_float("  Over/Under total: ")
            home_ml = _input_int(f"  {home_name} moneyline (e.g. -150): ")
            away_ml = _input_int(f"  {away_name} moneyline (e.g. +130): ")

        game = {
            "home_team": home_name, "away_team": away_name,
            "spread": spread, "over_under": ou,
            "home_ml": home_ml, "away_ml": away_ml,
            "time": "Manual", "home_rank": 99, "away_rank": 99,
            "home_record": "", "away_record": "",
        }

        result = analyzer.analyze_game(game, home_stats, away_stats)
        OutputFormatter.top_plays([result], label="ANALYSIS")
        OutputFormatter.disclaimer()

        try:
            again = input("  Analyze another game? (y/n): ").strip().lower()
            if again not in ('y', 'yes'):
                break
        except (EOFError, KeyboardInterrupt):
            break

    print("\n  Thanks for using the NCAA Basketball Betting Model!\n")


def _interactive_stats(team_name, is_home, use_espn=True):
    """Try ESPN lookup, then fallback DB, then manual entry."""
    # ESPN lookup
    if use_espn:
        result = ESPNDataProvider.search_team(team_name)
        if result:
            team_id, display_name = result
            print(f"  Found on ESPN: {display_name}")
            espn_stats = ESPNDataProvider.fetch_team_stats(team_id)
            if espn_stats:
                print(f"    PPG: {espn_stats['ppg']:.1f}  FG%: {espn_stats['fg_pct']:.1f}  "
                      f"3P%: {espn_stats['3p_pct']:.1f}  RPG: {espn_stats['reb']:.1f}")
                win_pct = 0.5  # unknown from stats alone
                return _build_stats_dict(espn_stats, win_pct, is_home)

    # Fallback DB
    fb = TeamDatabase.lookup(team_name)
    if fb:
        print(f"  Using fallback database for {team_name}")
        return fb

    # Manual entry
    print(f"  Team not found. Enter stats for {team_name}:")
    ppg = _input_float("    Points per game [75]: ", 75)
    fg = _input_float("    FG% [45]: ", 45)
    tp = _input_float("    3P% [35]: ", 35)
    ft = _input_float("    FT% [72]: ", 72)
    reb = _input_float("    Rebounds per game [37]: ", 37)
    ast = _input_float("    Assists per game [14]: ", 14)
    to = _input_float("    Turnovers per game [13]: ", 13)
    stl = _input_float("    Steals per game [7]: ", 7)
    blk = _input_float("    Blocks per game [4]: ", 4)
    wp = _input_float("    Win % (0-100) [50]: ", 50) / 100

    stats = {
        'ppg': ppg, 'fg_pct': fg, '3p_pct': tp, 'ft_pct': ft,
        'reb': reb, 'ast': ast, 'to': to, 'stl': stl, 'blk': blk,
        'win_pct': wp,
    }
    advanced = _estimate_advanced(stats, wp)
    stats.update(advanced)
    stats['home_win'] = min(1.0, wp + 0.08)
    stats['away_win'] = max(0.0, wp - 0.05)
    return stats


def _input_float(prompt, default=None):
    try:
        val = input(prompt).strip()
        if not val:
            return default
        return float(val)
    except (ValueError, EOFError):
        return default


def _input_int(prompt, default=None):
    try:
        val = input(prompt).strip()
        if not val:
            return default
        return int(val)
    except (ValueError, EOFError):
        return default


def _lookup_odds_api(home_name, away_name):
    """Search The Odds API for a specific matchup and return its odds."""
    odds_games = TheOddsAPIProvider.fetch_odds()
    if not odds_games:
        return None

    def _normalize(name):
        return name.lower().replace(".", "").replace("'", "").strip()

    def _match(a, b):
        na, nb = _normalize(a), _normalize(b)
        return na == nb or na in nb or nb in na

    for og in odds_games:
        if _match(home_name, og["home_team"]) and _match(away_name, og["away_team"]):
            return og
    return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NCAA Men's Basketball Betting Model",
    )
    parser.add_argument("--manual", action="store_true",
                        help="Interactive matchup input mode")
    parser.add_argument("--top", type=int, default=3,
                        help="Number of top plays to display (default: 3)")
    parser.add_argument("--all", action="store_true",
                        help="Show all game analyses without prompting")
    parser.add_argument("--no-espn", action="store_true",
                        help="Skip ESPN API, use fallback data only")
    parser.add_argument("--odds-api", action="store_true",
                        help="Fetch live odds from The Odds API (the-odds-api.com)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show model training details")

    args = parser.parse_args()

    if args.manual:
        run_manual_mode(args)
    else:
        run_auto_scan(args)


if __name__ == "__main__":
    main()
