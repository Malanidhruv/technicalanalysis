import pandas as pd
import requests
import streamlit as st
from datetime import datetime

CHART_URLS = [
    "https://ant.aliceblueonline.com/open-api/od/ChartAPIService/api/chart/history",
    "https://ant.aliceblueonline.com/rest/AliceBlueAPIService/api/chart/history",
]
DAILY_RESOLUTIONS = ("1D", "D")


class Instrument:
    def __init__(self, symbol, exchange="NSE"):
        self.symbol = symbol
        self.exchange = exchange


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

    def get_session_id(self):
        return True

    def get_instrument_by_token(self, exchange, token):
        if exchange in ("BSE (1)", "BSE"):
            exch = "BSE"
        else:
            exch = "NSE"
        return Instrument(str(token), exch)

    def get_historical(self, instrument, from_date, to_date, interval="D", exchange=None):
        exch = exchange if exchange else instrument.exchange

        if isinstance(from_date, datetime):
            from_ts = int(from_date.timestamp() * 1000)
        else:
            from_ts = int(from_date)

        if isinstance(to_date, datetime):
            to_ts = int(to_date.timestamp() * 1000)
        else:
            to_ts = int(to_date)

        resolutions = DAILY_RESOLUTIONS if interval == "D" else ("1",)
        last_error = "No data available"

        for url in CHART_URLS:
            for resolution in resolutions:
                payload = {
                    "token": str(instrument.symbol),
                    "resolution": resolution,
                    "from": str(from_ts),
                    "to": str(to_ts),
                    "exchange": exch,
                }

                try:
                    res = requests.post(url, json=payload, headers=self.headers, timeout=15)
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
                    if "no data" in last_error.lower():
                        continue
                    if "session" in last_error.lower() or "auth" in last_error.lower():
                        raise Exception(f"Auth error: {last_error}")
                    continue

                candles = data.get("result", [])
                if not candles:
                    continue

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


def get_cached_historical_data(alice, token, from_date, to_date, interval="D", exchange="NSE"):
    from_str = from_date.strftime("%Y%m%d") if isinstance(from_date, datetime) else str(from_date)
    to_str = to_date.strftime("%Y%m%d") if isinstance(to_date, datetime) else str(to_date)
    cache_key = f"{token}_{exchange}_{from_str}_{to_str}_{interval}"

    if cache_key in _historical_cache:
        return _historical_cache[cache_key]

    instrument = alice.get_instrument_by_token(exchange, token)
    df = alice.get_historical(instrument, from_date, to_date, interval, exchange)
    result = (instrument, df)
    _historical_cache[cache_key] = result
    return result


def clear_cache():
    global _historical_cache
    _historical_cache = {}
