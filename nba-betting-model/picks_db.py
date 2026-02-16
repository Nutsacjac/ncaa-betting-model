"""
SQLite database layer for tracking NBA betting picks and outcomes.
"""

import json
import sqlite3
from datetime import datetime

DB_PATH = "nba_picks.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create the picks table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            game_date TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_id TEXT,
            away_id TEXT,
            home_rank INTEGER,
            away_rank INTEGER,
            home_record TEXT,
            away_record TEXT,
            home_win_prob REAL,
            away_win_prob REAL,
            predicted_margin REAL,
            confidence REAL,
            spread REAL,
            over_under REAL,
            home_ml INTEGER,
            away_ml INTEGER,
            spread_edge REAL,
            spread_pick TEXT,
            ou_edge REAL,
            ou_pick TEXT,
            ml_edge REAL,
            ml_pick TEXT,
            best_market TEXT,
            best_pick TEXT,
            best_edge REAL,
            status TEXT NOT NULL DEFAULT 'pending',
            home_score INTEGER,
            away_score INTEGER,
            actual_margin REAL,
            actual_total REAL,
            resolved_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_pick(analysis):
    """Insert a pick from a BettingAnalyzer result dict. Dedup by date+teams+best_market."""
    conn = _get_conn()
    today = datetime.now().strftime("%Y-%m-%d")

    # Check for duplicate
    existing = conn.execute(
        "SELECT id FROM picks WHERE game_date=? AND home_team=? AND away_team=? AND best_market=?",
        (today, analysis["home_team"], analysis["away_team"], analysis.get("best_market")),
    ).fetchone()
    if existing:
        conn.close()
        return {"id": existing["id"], "duplicate": True}

    cursor = conn.execute("""
        INSERT INTO picks (
            game_date, home_team, away_team, home_id, away_id,
            home_rank, away_rank, home_record, away_record,
            home_win_prob, away_win_prob, predicted_margin, confidence,
            spread, over_under, home_ml, away_ml,
            spread_edge, spread_pick, ou_edge, ou_pick, ml_edge, ml_pick,
            best_market, best_pick, best_edge
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        today,
        analysis["home_team"],
        analysis["away_team"],
        analysis.get("home_id"),
        analysis.get("away_id"),
        analysis.get("home_rank"),
        analysis.get("away_rank"),
        analysis.get("home_record"),
        analysis.get("away_record"),
        analysis.get("home_win_prob"),
        analysis.get("away_win_prob"),
        analysis.get("predicted_margin"),
        analysis.get("confidence"),
        analysis.get("spread"),
        analysis.get("over_under"),
        analysis.get("home_ml"),
        analysis.get("away_ml"),
        analysis.get("spread_edge"),
        analysis.get("spread_pick"),
        analysis.get("ou_edge"),
        analysis.get("ou_pick"),
        analysis.get("ml_edge"),
        analysis.get("ml_pick"),
        analysis.get("best_market"),
        analysis.get("best_pick"),
        analysis.get("best_edge"),
    ))
    pick_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"id": pick_id, "duplicate": False}


def resolve_pick(pick_id, home_score, away_score):
    """Resolve a pending pick given final scores. Returns the updated status."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM picks WHERE id=?", (pick_id,)).fetchone()
    if not row:
        conn.close()
        return None

    actual_margin = home_score - away_score
    actual_total = home_score + away_score
    market = row["best_market"]
    pick_text = row["best_pick"] or ""
    status = "pending"

    if market == "Spread":
        spread = row["spread"]
        if spread is not None:
            covered_margin = actual_margin + spread
            if abs(covered_margin) < 0.5:
                status = "push"
            else:
                if row["home_team"] in pick_text:
                    status = "won" if (actual_margin + spread) > 0 else "lost"
                else:
                    status = "won" if (actual_margin + spread) < 0 else "lost"
        else:
            status = "cancelled"

    elif market == "O/U":
        ou_line = row["over_under"]
        if ou_line is not None:
            if abs(actual_total - ou_line) < 0.5:
                status = "push"
            elif "OVER" in pick_text.upper():
                status = "won" if actual_total > ou_line else "lost"
            else:
                status = "won" if actual_total < ou_line else "lost"
        else:
            status = "cancelled"

    elif market == "ML":
        if row["home_team"] in pick_text:
            status = "won" if actual_margin > 0 else "lost"
        else:
            status = "won" if actual_margin < 0 else "lost"
        if actual_margin == 0:
            status = "push"

    else:
        status = "cancelled"

    now = datetime.now().isoformat()
    conn.execute("""
        UPDATE picks SET status=?, home_score=?, away_score=?,
        actual_margin=?, actual_total=?, resolved_at=?
        WHERE id=?
    """, (status, home_score, away_score, actual_margin, actual_total, now, pick_id))
    conn.commit()
    conn.close()
    return status


def get_history(market=None, status=None, date_from=None, date_to=None, limit=50, offset=0):
    """Paginated pick history with optional filters."""
    conn = _get_conn()
    clauses = []
    params = []

    if market:
        clauses.append("best_market=?")
        params.append(market)
    if status:
        clauses.append("status=?")
        params.append(status)
    if date_from:
        clauses.append("game_date>=?")
        params.append(date_from)
    if date_to:
        clauses.append("game_date<=?")
        params.append(date_to)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    total = conn.execute(f"SELECT COUNT(*) as cnt FROM picks {where}", params).fetchone()["cnt"]

    rows = conn.execute(
        f"SELECT * FROM picks {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    conn.close()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "picks": [dict(r) for r in rows],
    }


def get_stats(market=None, status=None, date_from=None, date_to=None):
    """Aggregate accuracy stats with optional filters."""
    conn = _get_conn()
    clauses = []
    params = []

    if market:
        clauses.append("best_market=?")
        params.append(market)
    if date_from:
        clauses.append("game_date>=?")
        params.append(date_from)
    if date_to:
        clauses.append("game_date<=?")
        params.append(date_to)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    all_picks = conn.execute(f"SELECT * FROM picks {where} ORDER BY created_at ASC", params).fetchall()
    conn.close()

    total = len(all_picks)
    resolved = [p for p in all_picks if p["status"] in ("won", "lost", "push")]
    wins = sum(1 for p in resolved if p["status"] == "won")
    losses = sum(1 for p in resolved if p["status"] == "lost")
    pushes = sum(1 for p in resolved if p["status"] == "push")
    pending = sum(1 for p in all_picks if p["status"] == "pending")
    win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0

    profit = 0.0
    total_wagered = 0
    for p in resolved:
        if p["status"] == "push":
            continue
        total_wagered += 1
        if p["best_market"] == "ML":
            ml_odds = None
            pick_text = p["best_pick"] or ""
            if p["home_team"] in pick_text and p["home_ml"] is not None:
                ml_odds = p["home_ml"]
            elif p["away_ml"] is not None:
                ml_odds = p["away_ml"]

            if p["status"] == "won" and ml_odds is not None:
                if ml_odds > 0:
                    profit += ml_odds / 100
                else:
                    profit += 100 / abs(ml_odds)
            elif p["status"] == "lost":
                profit -= 1
        else:
            if p["status"] == "won":
                profit += 100 / 110
            else:
                profit -= 1

    roi = (profit / total_wagered * 100) if total_wagered > 0 else 0

    market_stats = {}
    for mkt in ("Spread", "O/U", "ML"):
        mkt_picks = [p for p in resolved if p["best_market"] == mkt]
        mkt_wins = sum(1 for p in mkt_picks if p["status"] == "won")
        mkt_losses = sum(1 for p in mkt_picks if p["status"] == "lost")
        mkt_total_wager = mkt_wins + mkt_losses
        mkt_profit = 0.0
        for p in mkt_picks:
            if p["status"] == "push":
                continue
            if mkt == "ML":
                ml_odds = None
                pick_text = p["best_pick"] or ""
                if p["home_team"] in pick_text and p["home_ml"] is not None:
                    ml_odds = p["home_ml"]
                elif p["away_ml"] is not None:
                    ml_odds = p["away_ml"]
                if p["status"] == "won" and ml_odds is not None:
                    if ml_odds > 0:
                        mkt_profit += ml_odds / 100
                    else:
                        mkt_profit += 100 / abs(ml_odds)
                elif p["status"] == "lost":
                    mkt_profit -= 1
            else:
                if p["status"] == "won":
                    mkt_profit += 100 / 110
                else:
                    mkt_profit -= 1

        market_stats[mkt] = {
            "total": len(mkt_picks),
            "wins": mkt_wins,
            "losses": mkt_losses,
            "win_rate": (mkt_wins / mkt_total_wager * 100) if mkt_total_wager > 0 else 0,
            "profit": round(mkt_profit, 2),
            "roi": (mkt_profit / mkt_total_wager * 100) if mkt_total_wager > 0 else 0,
        }

    buckets = [
        ("0-3%", 0, 0.03),
        ("3-5%", 0.03, 0.05),
        ("5-7%", 0.05, 0.07),
        ("7-10%", 0.07, 0.10),
        ("10%+", 0.10, 999),
    ]
    edge_calibration = []
    for label, lo, hi in buckets:
        bucket_picks = [p for p in resolved if lo <= abs(p["best_edge"] or 0) < hi]
        bucket_wins = sum(1 for p in bucket_picks if p["status"] == "won")
        bucket_decided = sum(1 for p in bucket_picks if p["status"] in ("won", "lost"))
        edge_calibration.append({
            "label": label,
            "total": len(bucket_picks),
            "wins": bucket_wins,
            "win_rate": (bucket_wins / bucket_decided * 100) if bucket_decided > 0 else 0,
        })

    current_streak = _calc_streak(resolved, current=True)
    best_streak = _calc_best_streak(resolved, "won")
    worst_streak = _calc_best_streak(resolved, "lost")

    last_10 = resolved[-10:] if len(resolved) >= 10 else resolved
    l10_wins = sum(1 for p in last_10 if p["status"] == "won")
    l10_decided = sum(1 for p in last_10 if p["status"] in ("won", "lost"))

    return {
        "total": total,
        "pending": pending,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": round(win_rate, 1),
        "profit": round(profit, 2),
        "roi": round(roi, 1),
        "market_stats": market_stats,
        "edge_calibration": edge_calibration,
        "current_streak": current_streak,
        "best_streak": best_streak,
        "worst_streak": worst_streak,
        "last_10": f"{l10_wins}-{l10_decided - l10_wins}" if l10_decided > 0 else "N/A",
    }


def _calc_streak(picks, current=False):
    """Calculate current win/loss streak from resolved picks."""
    if not picks:
        return {"type": "none", "count": 0}
    streak_type = None
    count = 0
    for p in reversed(picks):
        s = p["status"]
        if s == "push":
            continue
        if streak_type is None:
            streak_type = s
            count = 1
        elif s == streak_type:
            count += 1
        else:
            break
    return {"type": streak_type or "none", "count": count}


def _calc_best_streak(picks, target_status):
    """Find longest streak of target_status."""
    best = 0
    current = 0
    for p in picks:
        if p["status"] == target_status:
            current += 1
            best = max(best, current)
        elif p["status"] != "push":
            current = 0
    return best
