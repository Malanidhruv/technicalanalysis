"""
Persistence layer for the breakout -> retest watchlist.

Why this exists: a retest is inherently a multi-day pattern (breaks out on
Day 0, should pull back and hold within the next ~12 trading days). A
single EOD scan can't see that on its own - it needs to remember what
broke out yesterday/last week and check it again today. This module is
that memory.

Lifecycle of a watchlist row:
  active -> retested      (entry signal fires, trade_plan attached)
  active -> invalidated   (closed meaningfully below the breakout level)
  active -> expired       (too old without retesting, OR ran away too far
                            above the breakout level without ever pulling back)

NOTE: On Streamlit Community Cloud the local `data/` SQLite file is ephemeral
(lost on reboot/redeploy). Use export_active_csv / import_from_csv as backup.
"""

import csv
import io
import os
import sqlite3
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict, Union, BinaryIO, TextIO

_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = str(_ROOT / "data" / "breakout_watchlist.db")

# Scanner uses ThreadPoolExecutor; sqlite connections aren't thread-safe by default.
_write_lock = threading.Lock()

_CSV_COLUMNS = [
    "symbol", "exchange", "tier", "breakout_level", "breakout_date",
    "base_low", "breakout_volume", "status", "added_at", "updated_at",
    "retest_date", "invalidation_reason",
]


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            tier TEXT NOT NULL,               -- 'ATH' | '52W_HIGH' | 'VCP_BASE' | 'BASE'
            breakout_level REAL NOT NULL,
            breakout_date TEXT NOT NULL,       -- ISO date of the breakout day
            base_low REAL,                     -- only set for BASE/VCP tier (measured-move target sizing)
            breakout_volume REAL,
            status TEXT NOT NULL DEFAULT 'active',
            added_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            retest_date TEXT,
            invalidation_reason TEXT,
            PRIMARY KEY (symbol, exchange, breakout_date)
        )
    """)
    conn.commit()


def add_candidate(conn, symbol: str, exchange: str, tier: str, breakout_level: float,
                   breakout_date: str, base_low: Optional[float] = None,
                   breakout_volume: Optional[float] = None) -> bool:
    """Insert a new breakout candidate. Returns False if it already exists
    (idempotent - safe to call every day the breakout condition still holds
    on the FIRST breakout day; scanner should only call this on first-break)."""
    now = datetime.now().isoformat()
    with _write_lock:
        try:
            conn.execute("""
                INSERT INTO watchlist
                    (symbol, exchange, tier, breakout_level, breakout_date,
                     base_low, breakout_volume, status, added_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
            """, (symbol, exchange, tier, breakout_level, breakout_date,
                  base_low, breakout_volume, now, now))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # already tracked for this exact breakout day


def get_active(conn) -> List[Dict]:
    rows = conn.execute("SELECT * FROM watchlist WHERE status = 'active'").fetchall()
    return [dict(r) for r in rows]


def mark_status(conn, symbol: str, exchange: str, breakout_date: str, status: str,
                reason: Optional[str] = None, retest_date: Optional[str] = None) -> None:
    with _write_lock:
        conn.execute("""
            UPDATE watchlist
            SET status = ?, invalidation_reason = ?, retest_date = ?, updated_at = ?
            WHERE symbol = ? AND exchange = ? AND breakout_date = ?
        """, (status, reason, retest_date, datetime.now().isoformat(),
              symbol, exchange, breakout_date))
        conn.commit()


def already_tracked_recently(conn, symbol: str, exchange: str, within_days: int = 20) -> bool:
    """Avoid re-flagging a symbol as a 'new' breakout every day it keeps climbing -
    check if it already has an active or recently-resolved entry."""
    row = conn.execute("""
        SELECT 1 FROM watchlist
        WHERE symbol = ? AND exchange = ? AND status = 'active'
        LIMIT 1
    """, (symbol, exchange)).fetchone()
    return row is not None


def expire_stale(conn, max_age_trading_days: int = 12) -> int:
    """
    Expire active candidates that are too old without retesting.
    Trading-day count is approximated as calendar days * (7/5) since we don't
    carry a market calendar here - close enough for a ~12-trading-day window.
    Returns number of rows expired.
    """
    max_age_calendar_days = int(round(max_age_trading_days * 7 / 5))
    active = get_active(conn)
    count = 0
    today = date.today()
    for row in active:
        breakout_date = date.fromisoformat(row['breakout_date'])
        age_days = (today - breakout_date).days
        if age_days > max_age_calendar_days:
            mark_status(conn, row['symbol'], row['exchange'], row['breakout_date'],
                        'expired', reason=f"no retest within {max_age_trading_days} trading days")
            count += 1
    return count


def export_active_csv(conn) -> str:
    """Return CSV text of all active watchlist rows (Cloud backup safety net)."""
    rows = get_active(conn)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in _CSV_COLUMNS})
    return buf.getvalue()


def import_from_csv(conn, source: Union[str, bytes, BinaryIO, TextIO]) -> int:
    """
    Restore watchlist rows from a prior export (INSERT OR REPLACE).
    Returns number of rows written. Designed for post-Cloud-reboot recovery.
    """
    if isinstance(source, bytes):
        text = source.decode("utf-8-sig")
    elif hasattr(source, "read"):
        raw = source.read()
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    else:
        text = str(source)

    reader = csv.DictReader(io.StringIO(text))
    now = datetime.now().isoformat()
    count = 0
    with _write_lock:
        for row in reader:
            symbol = (row.get("symbol") or "").strip().upper()
            exchange = (row.get("exchange") or "").strip().upper()
            breakout_date = (row.get("breakout_date") or "").strip()
            tier = (row.get("tier") or "").strip()
            if not symbol or not exchange or not breakout_date or not tier:
                continue
            try:
                level = float(row["breakout_level"])
            except (KeyError, TypeError, ValueError):
                continue

            def _opt_float(key):
                v = row.get(key)
                if v is None or v == "":
                    return None
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None

            status = (row.get("status") or "active").strip() or "active"
            conn.execute("""
                INSERT OR REPLACE INTO watchlist
                    (symbol, exchange, tier, breakout_level, breakout_date,
                     base_low, breakout_volume, status, added_at, updated_at,
                     retest_date, invalidation_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, exchange, tier, level, breakout_date,
                _opt_float("base_low"), _opt_float("breakout_volume"), status,
                row.get("added_at") or now, row.get("updated_at") or now,
                row.get("retest_date") or None,
                row.get("invalidation_reason") or None,
            ))
            count += 1
        conn.commit()
    return count
