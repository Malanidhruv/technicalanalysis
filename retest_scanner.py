"""
Stage 2: retest scanner.

Runs only against the (small) active watchlist populated by
breakout_scanner.py - NOT the full universe, so this stays fast even run
right before 3:15pm. For each active candidate, checks today's price action
against its breakout_level:

    RETESTED     - pulled back into the retest zone and held (closed at/above
                   the level, ideally on lighter volume) -> generates the
                   actual entry/SL/target you act on.
    INVALIDATED  - closed meaningfully below the breakout level -> failed
                   breakout, drop it.
    RAN_AWAY     - extended too far above the level without ever pulling
                   back -> different trade than "buy the retest", drop it
                   rather than chase.
    STILL_WATCHING - none of the above yet, keep tracking.

Trade plan sizing (your requested defaults - 2% stop, 2-5% target):
  - Stop defaults to 2%, but widens automatically if the stock's ATR%
    is close to or bigger than that, so the stop isn't sitting inside
    normal daily noise.
  - Target is tier-aware: ATH breakouts have no overhead resistance ("blue
    sky") so default toward the top of your 2-5% band; 52W-high breakouts
    are capped by distance to the actual all-time high above; base
    breakouts use a measured-move (base height) estimate. All clamped to
    your 2-5% band either way.
"""

import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Callable

from watchlist_store import get_active, mark_status, expire_stale

# --- tunables (your stated defaults) ---
STOP_PCT_DEFAULT = 0.02
MIN_TARGET_PCT = 0.02
MAX_TARGET_PCT = 0.05
ATR_STOP_MULT = 1.3          # required stop >= ATR_STOP_MULT * ATR% to avoid noise stop-outs
RETEST_TOLERANCE_PCT = 0.015  # price low within 1.5% of breakout_level counts as "touched the retest zone"
INVALIDATION_BUFFER_PCT = 0.01  # close this far below breakout_level = failed breakout
RUNAWAY_PCT = 0.10             # extended >10% above breakout_level with no retest yet = don't chase
MAX_WATCH_TRADING_DAYS = 12


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    last_close = close.iloc[-1]
    if pd.isna(atr) or last_close <= 0:
        return None
    return float(atr / last_close * 100)


def compute_trade_plan(df: pd.DataFrame, breakout_level: float, tier: str,
                        base_low: Optional[float] = None) -> Optional[Dict]:
    entry = float(df['close'].iloc[-1])
    if entry <= 0:
        return None

    atr_pct = compute_atr_pct(df)
    stop_pct = STOP_PCT_DEFAULT
    stop_note = f"{STOP_PCT_DEFAULT*100:.1f}% (default)"
    if atr_pct is not None:
        min_required_stop_pct = (ATR_STOP_MULT * atr_pct) / 100
        if min_required_stop_pct > stop_pct:
            stop_pct = round(min_required_stop_pct, 4)
            stop_note = (f"{stop_pct*100:.2f}% (widened from {STOP_PCT_DEFAULT*100:.1f}% - "
                         f"stock's ATR is {atr_pct:.2f}%, 2% was inside noise band)")

    sl = entry * (1 - stop_pct)

    # tier-aware target sizing, clamped to [MIN_TARGET_PCT, MAX_TARGET_PCT]
    if tier == 'ATH':
        target_pct = MAX_TARGET_PCT  # no overhead resistance - aim for the top of the band
        target_note = "ATH breakout - no overhead resistance, targeting top of band"
    elif tier == '52W_HIGH':
        all_time_high = df['high'].max()
        if all_time_high > entry:
            headroom_pct = (all_time_high - entry) / entry
            target_pct = min(MAX_TARGET_PCT, max(MIN_TARGET_PCT, headroom_pct * 0.8))
            target_note = f"52W high breakout - {headroom_pct*100:.1f}% headroom to ATH above"
        else:
            target_pct = MAX_TARGET_PCT
            target_note = "52W high breakout - already at/near ATH too, treating as blue sky"
    else:  # BASE or VCP_BASE - same measured-move logic, VCP just arrived with a stricter filter
        if base_low is not None and breakout_level > 0:
            measured_move_pct = (breakout_level - base_low) / breakout_level
            target_pct = min(MAX_TARGET_PCT, max(MIN_TARGET_PCT, measured_move_pct))
            label = "VCP base" if tier == 'VCP_BASE' else "base"
            target_note = f"{label} breakout - measured move from {measured_move_pct*100:.1f}% base height"
        else:
            target_pct = MIN_TARGET_PCT
            target_note = "base breakout - no base_low on record, using conservative minimum target"

    target = entry * (1 + target_pct)
    rr = round(target_pct / stop_pct, 2)

    return {
        'entry': round(entry, 2),
        'stop_loss': round(sl, 2),
        'target': round(target, 2),
        'stop_pct': round(stop_pct * 100, 2),
        'target_pct': round(target_pct * 100, 2),
        'rr': rr,
        'atr_pct': round(atr_pct, 2) if atr_pct is not None else None,
        'stop_note': stop_note,
        'target_note': target_note,
    }


def evaluate_candidate(df: pd.DataFrame, candidate: Dict) -> Dict:
    """
    Returns {'action': 'retested'|'invalidated'|'ran_away'|'still_watching',
             'trade_plan': {...} or None, 'reason': str}
    """
    breakout_level = candidate['breakout_level']
    tier = candidate['tier']
    base_low = candidate.get('base_low')

    today_low = float(df['low'].iloc[-1])
    today_close = float(df['close'].iloc[-1])
    today_high = float(df['high'].iloc[-1])

    invalidation_line = breakout_level * (1 - INVALIDATION_BUFFER_PCT)
    if today_close < invalidation_line:
        return {'action': 'invalidated', 'trade_plan': None,
                'reason': f"closed {today_close} below invalidation line {invalidation_line:.2f}"}

    retest_zone_low = breakout_level * (1 - RETEST_TOLERANCE_PCT)
    touched_retest_zone = today_low <= breakout_level and today_low >= retest_zone_low
    held_above_level = today_close >= breakout_level

    if touched_retest_zone and held_above_level:
        # optional: lighter volume on the pullback than the original breakout day is a nice-to-have,
        # not required, since 'held above the level on the retest day' is the core signal.
        trade_plan = compute_trade_plan(df, breakout_level, tier, base_low)
        if trade_plan is None:
            return {'action': 'still_watching', 'trade_plan': None, 'reason': 'trade plan calc failed'}
        return {'action': 'retested', 'trade_plan': trade_plan, 'reason': 'retest held'}

    runaway_line = breakout_level * (1 + RUNAWAY_PCT)
    if today_high > runaway_line:
        return {'action': 'ran_away', 'trade_plan': None,
                'reason': f"extended past {runaway_line:.2f} ({RUNAWAY_PCT*100:.0f}% above breakout) without retesting"}

    return {'action': 'still_watching', 'trade_plan': None, 'reason': 'no retest yet'}


def scan_watchlist_for_retests(alice, conn, exchange_map: Dict[str, str],
                                get_historical_data_fn: Callable,
                                fundamental_score_fn: Optional[Callable] = None) -> List[Dict]:
    """
    exchange_map: {symbol: token} lookup so we know what to fetch for each
                  active watchlist symbol (build this from your existing
                  symbol_lookup.py).
    fundamental_score_fn: OPTIONAL callable(symbol) -> (score_0_100, breakdown)
                  or None if unavailable. Purely informational - never
                  filters or blocks a retest signal. Wrap your existing
                  fundamental_score.py + cached fetcher call here if you
                  want the side-column; leave as None to skip it entirely.
    """
    expire_stale(conn, MAX_WATCH_TRADING_DAYS)

    active = get_active(conn)
    signals = []

    for candidate in active:
        symbol = candidate['symbol']
        exchange = candidate['exchange']
        token = exchange_map.get(symbol)
        if token is None:
            continue

        try:
            instrument, df = get_historical_data_fn(
                alice, token,
                datetime.now() - timedelta(days=60),
                datetime.now(), "D", exchange
            )
            if df is None or len(df) < 5:
                continue

            result = evaluate_candidate(df, candidate)

            if result['action'] == 'retested':
                mark_status(conn, symbol, exchange, candidate['breakout_date'],
                            'retested', reason=result['reason'],
                            retest_date=date.today().isoformat())

                fund_score, fund_breakdown = (None, None)
                if fundamental_score_fn is not None:
                    try:
                        fund_score, fund_breakdown = fundamental_score_fn(symbol)
                    except Exception:
                        pass  # fundamentals are optional context - never block a signal on this

                signals.append({
                    'Name': symbol,
                    'Exchange': exchange,
                    'Tier': candidate['tier'],
                    'Breakout_Level': candidate['breakout_level'],
                    'Breakout_Date': candidate['breakout_date'],
                    'Fundamental_Score': fund_score,   # None if unavailable/not requested
                    'Fundamental_Breakdown': fund_breakdown,
                    **result['trade_plan'],
                })

            elif result['action'] in ('invalidated', 'ran_away'):
                mark_status(conn, symbol, exchange, candidate['breakout_date'],
                            result['action'], reason=result['reason'])
            # 'still_watching' - leave as active, no DB update needed

        except Exception as e:
            print(f"Error evaluating {symbol} for retest: {e}")

    signals.sort(key=lambda s: {'ATH': 0, '52W_HIGH': 1, 'VCP_BASE': 2, 'BASE': 3}.get(s['Tier'], 4))
    return signals
