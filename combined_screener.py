"""
Combined technical + fundamental swing-trade screener.

Pipeline
--------
1. Run your existing technical screener (analyze_all_tokens_advanced or
   analyze_all_tokens from your current codebase) restricted to your
   Nifty500 + BSE500 token universe. It returns candidates with a
   'Name' and 'Strength' (technical) score - typically your top ~10.

2. For each candidate, fetch fundamentals (wire your Screener.in scraper
   here) into a CompanyFundamentals object, and score it with
   fundamental_score.py. Hard filters (pledge / leverage / cash quality)
   drop a stock outright, they don't just lower its score.

3. Composite = W_TECHNICAL * normalized_technical + W_FUNDAMENTAL * fundamental.

4. Rank by composite, take the top N (default 2).

5. For each pick, compute Entry / Stop Loss / Target off the SAME daily
   OHLC dataframe your technical screener already pulled, using ATR +
   recent swing structure, and force the trade into a 1:2-1:3 R:R band.

This file assumes you already have the OHLC dataframe per symbol from your
technical scan (you fetch it anyway to compute EMAs/RSI/patterns) - reuse
it rather than re-fetching, to save API calls.
"""

import pandas as pd

from fundamental_score import CompanyFundamentals, fundamental_score, hard_filters

# Weight of technical vs fundamental score in the final ranking.
# For a breakout STRATEGY, technicals should still lead the entry decision -
# fundamentals here work as a quality gate, tune if you disagree.
W_TECHNICAL = 0.55
W_FUNDAMENTAL = 0.45

MIN_RR = 2.0
MAX_RR = 3.0
ATR_MULT = 1.3          # stop distance = ATR_MULT * ATR(14), before comparing to swing low
SWING_LOOKBACK = 12      # bars used to find recent swing low for the stop
RESISTANCE_LOOKBACK = 60  # bars used to check headroom to decide 1:2 vs 1:3


def normalize(value, lo, hi):
    if value is None:
        return 0.0
    if hi == lo:
        return 50.0
    return max(0.0, min(100.0, (value - lo) / (hi - lo) * 100.0))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_entry_sl_target(df: pd.DataFrame):
    """
    Entry: latest close (fits your 3:15pm decision / next-morning entry workflow).
    Stop: tighter of (recent swing low, entry - ATR_MULT*ATR), but never closer
          than 0.5*ATR to entry (so the stop isn't unrealistically tight on a
          low-volatility day).
    Target: risk * RR, where RR is 3.0 if there's enough headroom to the last
            60-day high, else 2.0. This is what keeps every pick inside your
            requested 1:2-1:3 band, never below or above it.
    Returns None if the structure doesn't support a valid stop (e.g. flat data).
    """
    if df is None or len(df) < 20:
        return None

    entry = float(df['close'].iloc[-1])

    atr_series = compute_atr(df)
    atr = atr_series.iloc[-1]
    if pd.isna(atr) or atr <= 0:
        atr = (df['high'] - df['low']).tail(14).mean()
    if pd.isna(atr) or atr <= 0:
        return None

    swing_low = df['low'].tail(SWING_LOOKBACK).min()
    atr_stop = entry - ATR_MULT * atr

    # tighter (higher) of the two stop candidates, but not closer than 0.5*ATR
    sl = max(swing_low, atr_stop)
    sl = min(sl, entry - 0.5 * atr)

    risk = entry - sl
    if risk <= 0:
        return None

    lookback_high = df['high'].tail(RESISTANCE_LOOKBACK).max()
    room = lookback_high - entry
    rr = MAX_RR if room >= risk * MAX_RR else MIN_RR

    target = entry + risk * rr

    return {
        'entry': round(entry, 2),
        'stop_loss': round(sl, 2),
        'target': round(target, 2),
        'risk_per_share': round(risk, 2),
        'reward_per_share': round(target - entry, 2),
        'rr': rr,
    }


def build_watchlist(technical_results, fundamentals_lookup, price_data_lookup, top_n=2):
    """
    technical_results  : list of dicts from your analyze_all_tokens_advanced /
                          analyze_all_tokens (must contain 'Name' and 'Strength')
    fundamentals_lookup: dict[symbol] -> CompanyFundamentals
    price_data_lookup  : dict[symbol] -> OHLCV DataFrame (reuse the one your
                          technical screener already fetched)
    top_n              : how many final picks to return (default 2)

    Returns a list of dicts (best first), each ready to log/alert/automate,
    or an empty list if nothing survives the filters.
    """
    tech_scores = [r.get('Strength', 0) for r in technical_results]
    t_lo, t_hi = (min(tech_scores), max(tech_scores)) if tech_scores else (0, 100)

    candidates = []
    dropped = []

    for r in technical_results:
        name = r['Name']
        cf = fundamentals_lookup.get(name)
        if cf is None:
            dropped.append((name, "no fundamental data available"))
            continue

        passed, reasons = hard_filters(cf)
        if not passed:
            dropped.append((name, "; ".join(reasons)))
            continue

        f_score, f_breakdown = fundamental_score(cf)
        t_score_norm = normalize(r.get('Strength', 0), t_lo, t_hi)
        composite = W_TECHNICAL * t_score_norm + W_FUNDAMENTAL * f_score

        df = price_data_lookup.get(name)
        trade_plan = compute_entry_sl_target(df)
        if trade_plan is None:
            dropped.append((name, "no valid entry/stop structure"))
            continue

        candidates.append({
            'Name': name,
            'Technical_Score': round(t_score_norm, 1),
            'Fundamental_Score': f_score,
            'Fundamental_Breakdown': f_breakdown,
            'Composite_Score': round(composite, 1),
            'Pattern': r.get('Pattern'),
            'statement_type': getattr(cf, 'statement_type', None),
            'Exchange': r.get('Exchange'),
            **trade_plan,
        })

    candidates.sort(key=lambda x: x['Composite_Score'], reverse=True)
    return candidates[:top_n], dropped


def print_watchlist(watchlist, dropped=None):
    if not watchlist:
        print("No candidates survived the technical + fundamental filters today.")
    for i, c in enumerate(watchlist, 1):
        print(f"\n#{i}  {c['Name']}  |  Composite {c['Composite_Score']}  "
              f"(Tech {c['Technical_Score']} / Fund {c['Fundamental_Score']})")
        print(f"    Pattern: {c.get('Pattern')}")
        print(f"    Entry {c['entry']}  SL {c['stop_loss']}  "
              f"Target {c['target']}  R:R 1:{c['rr']}")
    if dropped:
        print("\nDropped candidates:")
        for name, reason in dropped:
            print(f"  {name}: {reason}")


# ----------------------------------------------------------------------
# Example wiring (adapt to your actual code) - not executed on import
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # from advanced_strategies import analyze_all_tokens_advanced
    # from your_fundamental_fetcher import fetch_fundamentals_for_symbol
    #
    # tokens = load_nifty500_bse500_tokens()
    # technical_results = analyze_all_tokens_advanced(
    #     alice, tokens, strategy="Price Action Breakout"
    # )
    #
    # fundamentals_lookup = {}
    # price_data_lookup = {}
    # for r in technical_results:
    #     name = r['Name']
    #     fundamentals_lookup[name] = fetch_fundamentals_for_symbol(name)
    #     _, price_data_lookup[name] = get_historical_data(alice, token_for(name), ...)
    #
    # watchlist, dropped = build_watchlist(
    #     technical_results, fundamentals_lookup, price_data_lookup, top_n=2
    # )
    # print_watchlist(watchlist, dropped)
    pass
