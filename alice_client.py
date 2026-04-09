import pandas as pd
import requests
import streamlit as st
from datetime import datetime

BASE_URL = "https://ant.aliceblueonline.com/open-api/od/v1"

# ✅ Instrument class to match screener
class Instrument:
    def __init__(self, symbol, exchange="NSE"):
        self.symbol = symbol
        self.exchange = exchange


def initialize_alice():
    session = st.session_state.get("session")
    if not session:
        raise Exception("Login required")
    return Aliceblue(session)


class Aliceblue:
    def __init__(self, session):
        self.session = session
        self.headers = {
            "Authorization": f"Bearer {session}",
            "Content-Type": "application/json"
        }

    def get_session_id(self):
        return True

    def get_instrument_by_token(self, exchange, token):
        # Normalize exchange name
        if exchange in ("BSE (1)", "BSE"):
            exch = "BSE"
        else:
            exch = "NSE"
        return Instrument(str(token), exch)

    def get_historical(self, instrument, from_date, to_date, interval="D", exchange=None):
        # Use instrument's exchange if not explicitly passed
        exch = exchange if exchange else instrument.exchange

        # Convert datetime → milliseconds
        if isinstance(from_date, datetime):
            from_ts = int(from_date.timestamp() * 1000)
        else:
            from_ts = int(from_date)

        if isinstance(to_date, datetime):
            to_ts = int(to_date.timestamp() * 1000)
        else:
            to_ts = int(to_date)

        resolution = "1D" if interval == "D" else "1"

        payload = {
            "token": str(instrument.symbol),
            "resolution": resolution,
            "from": str(from_ts),
            "to": str(to_ts),
            "exchange": exch
        }

        url = "https://ant.aliceblueonline.com/open-api/od/ChartAPIService/api/chart/history"

        print(f"🔍 Fetching TOKEN: {instrument.symbol} | Exchange: {exch} | Resolution: {resolution}")

        try:
            res = requests.post(url, json=payload, headers=self.headers, timeout=15)
            res.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed for token {instrument.symbol}: {e}")

        try:
            data = res.json()
        except Exception:
            raise Exception(f"Invalid JSON response: {res.text[:200]}")

        print(f"📦 API Response stat: {data.get('stat')} | Token: {instrument.symbol}")

        if data.get("stat") != "Ok":
            raise Exception(f"API Error for {instrument.symbol}: {data.get('emsg', data)}")

        candles = data.get("result", [])

        if not candles:
            print(f"❌ No candle data for token {instrument.symbol}")
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candles)

        # Rename columns robustly
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower == "time":
                rename_map[col] = "datetime"
            elif col_lower in ("open", "high", "low", "close", "volume"):
                rename_map[col] = col_lower

        df = df.rename(columns=rename_map)

        # Parse datetime from the correct column
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", errors="coerce")
        else:
            print(f"⚠️ No 'time'/'datetime' column found. Columns: {df.columns.tolist()}")

        # Cast OHLCV columns to float safely
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"])
        df = df.sort_values("datetime").reset_index(drop=True)

        return df


# ✅ Cache using string keys (lru_cache needs hashable args)
_historical_cache = {}


def get_cached_historical_data(alice, token, from_date, to_date, interval="D", exchange="NSE"):
    """
    Cached wrapper for historical data fetching.
    Uses a dict-based cache since lru_cache cannot handle class instances.
    """
    # Build a string cache key
    from_str = from_date.strftime("%Y%m%d") if isinstance(from_date, datetime) else str(from_date)
    to_str = to_date.strftime("%Y%m%d") if isinstance(to_date, datetime) else str(to_date)
    cache_key = f"{token}_{exchange}_{from_str}_{to_str}_{interval}"

    if cache_key in _historical_cache:
        print(f"✅ Cache HIT: {cache_key}")
        return _historical_cache[cache_key]

    instrument = alice.get_instrument_by_token(exchange, token)
    df = alice.get_historical(instrument, from_date, to_date, interval, exchange)

    result = (instrument, df)
    _historical_cache[cache_key] = result
    return result


def clear_cache():
    """Clear the historical data cache."""
    global _historical_cache
    _historical_cache = {}
    print("🗑️ Cache cleared")
