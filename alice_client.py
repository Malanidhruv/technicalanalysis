import pandas as pd
import requests
import streamlit as st
from functools import lru_cache

BASE_URL = "https://ant.aliceblueonline.com/open-api/od/v1"


# ✅ fake instrument to match your screener
class Instrument:
    def __init__(self, symbol):
        self.symbol = symbol


def initialize_alice():
    session = st.session_state.get("session")

    if not session:
        raise Exception("Login required")

    return Aliceblue(session)


class Aliceblue:
    def __init__(self, session):
        self.headers = {
            "Authorization": f"Bearer {session}",
            "Content-Type": "application/json"
        }

    def get_session_id(self):
        return True

    # ✅ IMPORTANT: returns object (not string)
    def get_instrument_by_token(self, exchange, token):
        return Instrument(token)

    # ✅ IMPORTANT: returns DataFrame (same as before)
    def get_historical(self, instrument, from_date, to_date, interval="D"):

        payload = {
            "symbol": instrument.symbol,
            "fromDate": from_date.strftime("%Y-%m-%d"),
            "toDate": to_date.strftime("%Y-%m-%d"),
            "interval": interval
        }

        url = f"{BASE_URL}/market/getHistoricalData"

        res = requests.post(url, json=payload, headers=self.headers)
        data = res.json()

        if data.get("stat") != "Ok":
            raise Exception(data)

        candles = data.get("data", [])

        df = pd.DataFrame(candles, columns=[
            "datetime", "open", "high", "low", "close", "volume"
        ])

        if df.empty:
            return df

        df["datetime"] = pd.to_datetime(df["datetime"])

        df = df.astype({
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float
        })

        return df


# ✅ your cache (unchanged)
@lru_cache(maxsize=1000)
def get_cached_historical_data(alice, token, from_date, to_date, interval="D", exchange='NSE'):
    instrument = alice.get_instrument_by_token(exchange, token)
    df = alice.get_historical(instrument, from_date, to_date, interval)
    return instrument, df


def clear_cache():
    get_cached_historical_data.cache_clear()
