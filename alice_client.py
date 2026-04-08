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

        

    def get_historical(self, instrument, from_date, to_date, interval="D", exchange="NSE"):

        # Convert datetime → milliseconds
        from_ts = int(from_date.timestamp() * 1000)
        to_ts = int(to_date.timestamp() * 1000)
    
        payload = {
            "token": str(instrument.symbol),   # IMPORTANT → token, not symbol
            "resolution": "1D" if interval == "D" else "1",
            "from": str(from_ts),
            "to": str(to_ts),
            "exchange": exchange
        }
    
        url = "https://ant.aliceblueonline.com/open-api/od/ChartAPIService/api/chart/history"
    
        print("🔍 Fetching TOKEN:", instrument.symbol)
    
        res = requests.post(url, json=payload, headers=self.headers)
    
        try:
            data = res.json()
        except:
            raise Exception(f"Invalid response: {res.text}")
    
        print("📦 API Response:", data)
    
        if data.get("stat") != "Ok":
            raise Exception(f"API Error: {data}")
    
        candles = data.get("result", [])
    
        if not candles:
            print(f"❌ No data for token {instrument.symbol}")
            return pd.DataFrame()
    
        df = pd.DataFrame(candles)
    
        # Rename to match your system
        df = df.rename(columns={
            "time": "datetime",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        })
    
        df["datetime"] = pd.to_datetime(df["time"])
    
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
