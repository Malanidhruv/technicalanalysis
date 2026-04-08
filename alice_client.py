import datetime
from functools import lru_cache
import pandas as pd
import requests
import streamlit as st

BASE_URL = "https://ant.aliceblueonline.com/open-api/od/v1"


# 🔹 NEW: No credentials needed anymore
def initialize_alice():
    """Initialize Alice client using session from Streamlit."""
    session = st.session_state.get("session")

    if not session:
        raise Exception("Not logged in. Please login first.")

    return Aliceblue(session)


class Aliceblue:
    def __init__(self, session):
        self.session = session
        self.headers = {
            "Authorization": f"Bearer {session}",
            "Content-Type": "application/json"
        }

    # keep compatibility (your code expects this)
    def get_session_id(self):
        return True

    # 🔹 Replace token → symbol logic
    def get_instrument_by_token(self, exchange, token):
        """
        Your old code used token-based instruments.
        For now, we assume token itself is usable symbol.
        If not, we will fix mapping later.
        """
        return token

    # 🔹 MAIN FIX: historical data
    def get_historical(self, instrument, from_date, to_date, interval="D"):

        payload = {
            "symbol": instrument,
            "fromDate": str(from_date),
            "toDate": str(to_date),
            "interval": interval
        }

        url = f"{BASE_URL}/market/getHistoricalData"

        response = requests.post(url, json=payload, headers=self.headers)
        data = response.json()

        if data.get("stat") != "Ok":
            raise Exception(f"API Error: {data}")

        candles = data.get("data", [])

        # Convert to DataFrame (IMPORTANT for your screener)
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


# 🔹 KEEP YOUR CACHE FUNCTION (unchanged logic)
@lru_cache(maxsize=1000)
def get_cached_historical_data(alice, token, from_date, to_date, interval="D", exchange='NSE'):
    """Cached version of historical data fetching."""
    instrument = alice.get_instrument_by_token(exchange, token)

    df = alice.get_historical(instrument, from_date, to_date, interval)

    return instrument, df


def clear_cache():
    """Clear the historical data cache."""
    get_cached_historical_data.cache_clear()
