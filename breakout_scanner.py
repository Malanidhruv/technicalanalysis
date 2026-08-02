"""
Stage 1: EOD breakout detector.

Runs across your full universe (NSE + BSE Top 500) once a day and flags
stocks that broke out TODAY across three tiers:

    ATH        - closed above its entire available price history
    52W_HIGH   - closed above its 52-week high (but not an all-time high)
    BASE       - broke out of a tight (<15%) 15-day consolidation range

A stock only gets added to the watchlist on its FIRST breakout day (checked
via `already_tracked_recently` + a "wasn't already broken out yesterday"
condition) - this stage does NOT decide entries, it just starts tracking.
Entries happen in retest_scanner.py once a tracked stock pulls back and holds.

Wire your own historical-data fetcher in - this module takes it as a
parameter (`get_historical_data_fn`) rather than importing your alice_client
directly, so it stays testable and decoupled from your app's session/cache
plumbing. Signature expected:
    get_historical_data_fn(alice, token, from_date, to_date, interval, exchange)
        -> (instrument, df)   # df has columns: open, high, low, close, volume
"""

import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Callable, List
from scipy.signal import argrelextrema
import numpy as np

from watchlist_store import add_candidate, already_tracked_recently

# --- tunables ---
MIN_AVG_TURNOVER = 1_00_00_000     # min ~20-day avg daily turnover in Rs (1 crore) - tune to your capital/slippage tolerance
BREAKOUT_VOLUME_MULT = 1.3          # today's volume must be >= this x the 20-day avg to count as a real breakout, not drift
BASE_RANGE_MAX_PCT = 15.0            # 15-day high/low range must be tighter than this % to count as a "base"
BASE_LOOKBACK_DAYS = 15

# --- VCP tunables ---
VCP_LOOKBACK_DAYS = 50               # how far back to look for contraction legs
VCP_SWING_ORDER = 3                   # bars on each side to confirm a local swing high/low
VCP_MIN_LEGS = 2                       # need at least this many pullback legs to call it a contraction sequence
VCP_CONTRACTION_TOLERANCE = 1.05        # each leg's depth must be <= prior leg's depth * this (small noise allowance)
VCP_FINAL_LEG_MAX_PCT = 10.0              # the final (tightest) leg must be under this % depth
VCP_VOLUME_CONTRACTION_TOLERANCE = 1.15    # final leg's avg volume must be <= first leg's avg volume * this
VCP_MIN_LEG_DEPTH_PCT = 2.5                 # ignore wiggles shallower than this - noise, not a real pullback leg


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def avg_turnover(df: pd.DataFrame, days: int = 20) -> float:
    if 'volume' not in df.columns or len(df) < days:
        return 0.0
    turnover = (df['close'] * df['volume']).tail(days)
    return float(turnover.mean())


def _volume_confirmed(df: pd.DataFrame) -> bool:
    if 'volume' not in df.columns or len(df) < 21:
        return False
    avg_vol_20 = df['volume'].iloc[-21:-1].mean()
    if avg_vol_20 <= 0:
        return False
    return df['volume'].iloc[-1] >= BREAKOUT_VOLUME_MULT * avg_vol_20


def check_ath_breakout(df: pd.DataFrame) -> Optional[Dict]:
    """Closed above the entire available price history today, and wasn't
    already above it yesterday (first-break-day filter)."""
    if len(df) < 30:
        return None
    prior_ath = df['high'].iloc[:-1].max()
    today_close = df['close'].iloc[-1]
    if today_close <= prior_ath:
        return None
    if len(df) >= 2:
        prior_ath_yesterday = df['high'].iloc[:-2].max() if len(df) > 2 else prior_ath
        if df['close'].iloc[-2] > prior_ath_yesterday:
            return None  # already broke out yesterday, not a fresh signal
    if not _volume_confirmed(df):
        return None
    return {'tier': 'ATH', 'breakout_level': round(float(prior_ath), 2), 'base_low': None}


def check_52w_breakout(df: pd.DataFrame) -> Optional[Dict]:
    """Closed above the 52-week high (but this is NOT also an ATH - that's
    the stronger signal and takes priority via classify_breakout)."""
    if len(df) < 252:
        return None
    window = df.tail(253)  # 252 prior + today
    prior_52w_high = window['high'].iloc[:-1].max()
    today_close = window['close'].iloc[-1]
    if today_close <= prior_52w_high:
        return None
    if len(df) >= 254:
        prior_52w_high_yday = df['high'].iloc[-254:-2].max()
        if df['close'].iloc[-2] > prior_52w_high_yday:
            return None
    if not _volume_confirmed(df):
        return None
    return {'tier': '52W_HIGH', 'breakout_level': round(float(prior_52w_high), 2), 'base_low': None}


def _find_swing_points(window: pd.DataFrame, order: int = VCP_SWING_ORDER) -> List[Dict]:
    """
    Find local swing highs/lows in a price window, then collapse into a clean
    alternating high-low-high-low sequence (real swings, not every local wiggle).
    Returns chronologically ordered list of {'type': 'high'|'low', 'idx': i, 'price': p}.
    """
    highs = window['high'].values
    lows = window['low'].values

    max_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
    min_idx = argrelextrema(lows, np.less_equal, order=order)[0]

    points = [{'type': 'high', 'idx': int(i), 'price': float(highs[i])} for i in max_idx]
    points += [{'type': 'low', 'idx': int(i), 'price': float(lows[i])} for i in min_idx]
    points.sort(key=lambda p: p['idx'])

    if not points:
        return []

    # collapse consecutive same-type points to the most extreme one, so we get
    # a clean alternating sequence of real swings rather than noisy clusters
    swings = [points[0]]
    for p in points[1:]:
        if p['type'] == swings[-1]['type']:
            if (p['type'] == 'high' and p['price'] > swings[-1]['price']) or \
               (p['type'] == 'low' and p['price'] < swings[-1]['price']):
                swings[-1] = p
        else:
            swings.append(p)
    return swings


def _extract_contraction_legs(swings: List[Dict], window: pd.DataFrame) -> List[Dict]:
    """
    Turn an alternating swing sequence into a list of decline "legs"
    (high -> subsequent low), each with a depth% and average volume during
    the decline - the raw material for checking if pullbacks are contracting.
    """
    legs = []
    for a, b in zip(swings, swings[1:]):
        if a['type'] == 'high' and b['type'] == 'low' and a['price'] > 0:
            depth_pct = (a['price'] - b['price']) / a['price'] * 100
            vol_slice = window['volume'].iloc[a['idx']:b['idx'] + 1]
            avg_vol = float(vol_slice.mean()) if len(vol_slice) > 0 else 0.0
            legs.append({'start_idx': a['idx'], 'end_idx': b['idx'],
                        'depth_pct': depth_pct, 'avg_volume': avg_vol})
    return legs


def check_vcp_base_breakout(df: pd.DataFrame) -> Optional[Dict]:
    """
    VCP (Volatility Contraction Pattern): a sequence of pullback legs, each
    tighter in both price range and volume than the last, culminating in a
    tight final contraction before the breakout. This is a materially higher
    quality signal than a flat consolidation range - it's evidence of sellers
    progressively drying up rather than just "price has been quiet."
    """
    if len(df) < VCP_LOOKBACK_DAYS + 5:
        return None

    window = df.iloc[-(VCP_LOOKBACK_DAYS + 1):-1].reset_index(drop=True)  # exclude today
    swings = _find_swing_points(window)
    all_legs = _extract_contraction_legs(swings, window)

    # drop micro-wiggles (noise) - a real VCP leg is a meaningful pullback,
    # not a 1% random fluctuation. Only real legs count toward the pattern.
    legs = [l for l in all_legs if l['depth_pct'] >= VCP_MIN_LEG_DEPTH_PCT]

    if len(legs) < VCP_MIN_LEGS:
        return None

    # check the legs are contracting: each successive leg tighter (with small tolerance)
    # in both price depth and volume than the one before it
    for prev_leg, curr_leg in zip(legs, legs[1:]):
        if curr_leg['depth_pct'] > prev_leg['depth_pct'] * VCP_CONTRACTION_TOLERANCE:
            return None  # not tightening - this leg is as wide or wider than the last
        if prev_leg['avg_volume'] > 0 and curr_leg['avg_volume'] > prev_leg['avg_volume'] * VCP_VOLUME_CONTRACTION_TOLERANCE:
            return None  # volume isn't drying up on the pullbacks

    final_leg = legs[-1]
    if final_leg['depth_pct'] > VCP_FINAL_LEG_MAX_PCT:
        return None  # final contraction still too wide to call it "tight"

    base_high = window['high'].max()
    base_low = window['low'].min()

    today_close = df['close'].iloc[-1]
    yesterday_close = df['close'].iloc[-2]
    if today_close <= base_high or yesterday_close > base_high:
        return None

    if not _volume_confirmed(df):
        return None

    return {'tier': 'VCP_BASE', 'breakout_level': round(float(base_high), 2),
            'base_low': round(float(base_low), 2),
            'num_contractions': len(legs),
            'final_leg_depth_pct': round(final_leg['depth_pct'], 1)}


def check_base_breakout(df: pd.DataFrame) -> Optional[Dict]:
    """Broke out of a tight consolidation range today. base_low is recorded
    for measured-move target sizing later in retest_scanner.py."""
    if len(df) < BASE_LOOKBACK_DAYS + 5:
        return None

    base_window = df.iloc[-(BASE_LOOKBACK_DAYS + 1):-1]  # the base, excluding today
    base_high = base_window['high'].max()
    base_low = base_window['low'].min()
    if base_low <= 0:
        return None

    range_pct = (base_high - base_low) / base_low * 100
    if range_pct > BASE_RANGE_MAX_PCT:
        return None  # not tight enough to call it a base

    today_close = df['close'].iloc[-1]
    if today_close <= base_high:
        return None

    yesterday_close = df['close'].iloc[-2]
    if yesterday_close > base_high:
        return None  # already broke out yesterday

    if not _volume_confirmed(df):
        return None

    return {'tier': 'BASE', 'breakout_level': round(float(base_high), 2),
            'base_low': round(float(base_low), 2)}


def classify_breakout(df: pd.DataFrame) -> Optional[Dict]:
    """Check tiers in priority order - ATH beats 52W beats BASE, since a
    stock can technically satisfy more than one at once (e.g. an ATH is
    almost always also breaking a base)."""
    for check_fn in (check_ath_breakout, check_52w_breakout, check_vcp_base_breakout, check_base_breakout):
        result = check_fn(df)
        if result is not None:
            return result
    return None


def scan_symbol_for_breakout(alice, token, symbol, exchange, conn,
                              get_historical_data_fn: Callable) -> Optional[str]:
    """Returns the tier string if a new breakout was added, else None."""
    try:
        if already_tracked_recently(conn, symbol, exchange):
            return None  # already being watched, don't duplicate

        instrument, df = get_historical_data_fn(
            alice, token,
            datetime.now() - timedelta(days=400),  # >252 trading days for 52W/ATH checks
            datetime.now(), "D", exchange
        )
        if df is None or len(df) < 30:
            return None

        if avg_turnover(df) < MIN_AVG_TURNOVER:
            return None  # too illiquid for a clean 3:15pm entry

        result = classify_breakout(df)
        if result is None:
            return None

        breakout_date = df.index[-1].date().isoformat() if hasattr(df.index[-1], 'date') \
            else datetime.now().date().isoformat()
        breakout_volume = float(df['volume'].iloc[-1]) if 'volume' in df.columns else None

        added = add_candidate(
            conn, symbol, exchange, result['tier'], result['breakout_level'],
            breakout_date, base_low=result['base_low'], breakout_volume=breakout_volume
        )
        return result['tier'] if added else None

    except Exception as e:
        print(f"Error scanning {symbol} for breakout: {e}")
        return None


def scan_universe_for_breakouts(alice, tokens_with_symbols, exchange, conn,
                                 get_historical_data_fn: Callable, max_workers: int = 16,
                                 on_progress: Optional[Callable] = None) -> Dict:
    """
    tokens_with_symbols: list of (token, symbol) tuples for one exchange.
    Returns summary stats including new_vcp for VCP_BASE tier.
    on_progress(done, total) is called from the coordinating thread as futures complete.
    """
    stats = {
        'scanned': 0, 'new_ath': 0, 'new_52w': 0, 'new_vcp': 0, 'new_base': 0, 'errors': 0,
    }
    tier_key = {
        'ATH': 'new_ath', '52W_HIGH': 'new_52w', 'VCP_BASE': 'new_vcp', 'BASE': 'new_base',
    }
    total = len(tokens_with_symbols)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(scan_symbol_for_breakout, alice, token, symbol, exchange,
                             conn, get_historical_data_fn): symbol
            for token, symbol in tokens_with_symbols
        }
        for future in as_completed(futures):
            stats['scanned'] += 1
            try:
                tier = future.result()
                if tier and tier in tier_key:
                    stats[tier_key[tier]] += 1
            except Exception:
                stats['errors'] += 1
            if on_progress is not None:
                try:
                    on_progress(stats['scanned'], total)
                except Exception:
                    pass

    return stats
