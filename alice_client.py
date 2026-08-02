import pandas as pd
import requests
import streamlit as st
from datetime import datetime
from threading import Lock

from symbol_lookup import token_to_symbol

# Prefer the primary ANT chart endpoint; fallback only if needed.
CHART_URLS = [
    "https://ant.aliceblueonline.com/open-api/od/ChartAPIService/api/chart/history",
    "https://ant.aliceblueonline.com/rest/AliceBlueAPIService/api/chart/history",
]
DAILY_RESOLUTIONS = ("1D", "D")

# Remember what worked so later stocks skip failed URL/resolution combos.
_endpoint_lock = Lock()
_preferred_url = CHART_URLS[0]
_preferred_resolution = "1D"

# Parallelism vs AliceBlue rate limits — sweet spot for Cloud free tier.
DEFAULT_WORKERS = 16
REQUEST_TIMEOUT = 8  # fail fast; hung calls were the main stall


class Instrument:
    def __init__(self, token, exchange="NSE", symbol=None):
        self.token = str(token)
        self.exchange = exchange
        self.symbol = symbol or token_to_symbol(token, exchange)


def initialize_alice():
    session = st.session_state.get("session")
    user_id = st.session_state.get("user_id")
    if not session:
        raise Exception("Login required")
    if not user_id:
        raise Exception("User ID missing — please log out and log in again.")
    return Aliceblue(session, user_id)


class Aliceblue:
    def __init__(self, session, user_id):
        self.session = session
        self.user_id = user_id
        self.headers = {
            "Authorization": f"Bearer {user_id} {session}",
            "Content-Type": "application/json",
        }
        # One Session per client: connection reuse across threads is OK for requests.Session
        # with a lock around adapter use; simpler: each call uses requests.post (thread-safe).
        self._http = requests.Session()
        self._http.headers.update(self.headers)

    def get_session_id(self):
        return True

    def get_instrument_by_token(self, exchange, token):
        if exchange in ("BSE (1)", "BSE"):
            exch = "BSE"
        else:
            exch = "NSE"
        return Instrument(token, exch)

    def get_historical(self, instrument, from_date, to_date, interval="D", exchange=None):
        global _preferred_url, _preferred_resolution
        exch = exchange if exchange else instrument.exchange
        api_token = getattr(instrument, "token", None) or instrument.symbol

        if isinstance(from_date, datetime):
            from_ts = int(from_date.timestamp() * 1000)
        else:
            from_ts = int(from_date)

        if isinstance(to_date, datetime):
            to_ts = int(to_date.timestamp() * 1000)
        else:
            to_ts = int(to_date)

        with _endpoint_lock:
            preferred_url = _preferred_url
            preferred_res = _preferred_resolution

        if interval == "D":
            # Try preferred combo first, then remaining options
            urls = [preferred_url] + [u for u in CHART_URLS if u != preferred_url]
            resolutions = [preferred_res] + [r for r in DAILY_RESOLUTIONS if r != preferred_res]
        else:
            urls = [preferred_url] + [u for u in CHART_URLS if u != preferred_url]
            resolutions = ("1",)

        last_error = "No data available"

        for url in urls:
            for resolution in resolutions:
                payload = {
                    "token": str(api_token),
                    "resolution": resolution,
                    "from": str(from_ts),
                    "to": str(to_ts),
                    "exchange": exch,
                }

                try:
                    res = self._http.post(url, json=payload, timeout=REQUEST_TIMEOUT)
                    res.raise_for_status()
                    data = res.json()
                except requests.exceptions.RequestException as exc:
                    last_error = str(exc)
                    continue
                except Exception as exc:
                    last_error = str(exc)
                    continue

                if data.get("stat") != "Ok":
                    last_error = data.get("emsg") or data.get("message") or str(data)
                    low = last_error.lower()
                    if "session" in low or "auth" in low or "unauthorized" in low:
                        raise Exception(f"Auth error: {last_error}")
                    # Don't burn more endpoints on "no data"
                    if "no data" in low:
                        return pd.DataFrame(
                            columns=["datetime", "open", "high", "low", "close", "volume"]
                        )
                    continue

                candles = data.get("result", [])
                if not candles:
                    continue

                with _endpoint_lock:
                    _preferred_url = url
                    _preferred_resolution = resolution

                return self._candles_to_dataframe(candles)

        if "no data" in last_error.lower():
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        raise Exception(f"API Error for {instrument.symbol}: {last_error}")

    @staticmethod
    def _candles_to_dataframe(candles):
        df = pd.DataFrame(candles)

        rename_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower == "time":
                rename_map[col] = "datetime"
            elif col_lower in ("open", "high", "low", "close", "volume", "vol", "v"):
                rename_map[col] = "volume" if col_lower in ("volume", "vol", "v") else col_lower

        df = df.rename(columns=rename_map)

        if "datetime" in df.columns:
            if pd.api.types.is_numeric_dtype(df["datetime"]):
                df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", errors="coerce")
            else:
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "volume" not in df.columns:
            df["volume"] = 0.0
        else:
            df["volume"] = df["volume"].fillna(0.0)

        df = df.dropna(subset=["close"])
        return df.sort_values("datetime").reset_index(drop=True) if "datetime" in df.columns else df


_historical_cache = {}
_cache_lock = Lock()


def get_cached_historical_data(alice, token, from_date, to_date, interval="D", exchange="NSE"):
    # Day-granularity key so same-day strategy switches reuse candles
    from_str = from_date.strftime("%Y%m%d") if isinstance(from_date, datetime) else str(from_date)[:8]
    to_str = to_date.strftime("%Y%m%d") if isinstance(to_date, datetime) else str(to_date)[:8]
    cache_key = f"{token}_{exchange}_{from_str}_{to_str}_{interval}"

    with _cache_lock:
        if cache_key in _historical_cache:
            return _historical_cache[cache_key]

    instrument = alice.get_instrument_by_token(exchange, token)
    df = alice.get_historical(instrument, from_date, to_date, interval, exchange)
    result = (instrument, df)

    with _cache_lock:
        _historical_cache[cache_key] = result
    return result


def clear_cache():
    global _historical_cache
    with _cache_lock:
        _historical_cache = {}
