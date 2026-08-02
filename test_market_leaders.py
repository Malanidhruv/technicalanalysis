"""Quick self-check: Market Leaders must not hard-fail on <252 bars."""
import numpy as np
import pandas as pd

from stock_analysis import analyze_market_leaders
from alice_client import Instrument


def _make_uptrend(n=200):
    # Smooth uptrend finishing near highs — ADX/EMA friendly
    t = np.arange(n, dtype=float)
    close = 100 + t * 0.4 + np.sin(t / 8) * 0.5
    high = close + 0.8
    low = close - 0.8
    vol = np.full(n, 2000.0)
    vol[-20:] = 2800.0
    return pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    })


def main():
    inst = Instrument("1", "NSE", symbol="TEST")
    out = analyze_market_leaders(_make_uptrend(200), inst)
    assert out is not None, f"expected leader match, got {out}"
    assert out["Strength"] >= 55
    # Deep drawdown should fail distance/performance gates
    df = _make_uptrend(200)
    df.loc[df.index[-1], "close"] = df["close"].iloc[-1] * 0.5
    assert analyze_market_leaders(df, inst) is None
    print("ok: market leaders accepts <252 bars and scores leaders")


if __name__ == "__main__":
    main()
